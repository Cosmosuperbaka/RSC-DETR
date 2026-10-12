# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Dual-stream transform pipeline for RGB+IR DroneVehicle detection.

Copyright(c) 2025. All Rights Reserved.

Strategy: synchronized random parameters via RNG fork
=====================================================

torchvision v2 transforms are stateful in that each forward draws fresh
random params from the global torch RNG. To apply the EXACT same geometric
transform to both RGB and IR images, we:

    1. Snapshot the RNG state before transforming RGB.
    2. Run transform(rgb, target) to get (new_rgb, new_target).
    3. Restore the RNG state.
    4. Run transform(ir, dummy_target) to get (new_ir, _) — same random
       params consumed in the same order, so the geometry is identical.
    5. Discard ir's dummy target; keep new_target from step 2.

Per-modality ops (RandomPhotometricDistort, ConvertPILImage, Normalize) are
applied INDEPENDENTLY — RGB and IR have different sensor characteristics,
so we want different colour augmentation for each modality. This is
controlled by the `_PER_MODALITY_OPS` set below.

This keeps the YAML transform list 100% identical to the original
RT-DETRv2 single-stream config — you just change `Compose` to `DualCompose`.
"""

import copy
from typing import Any

import torch
import torch.nn as nn
import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from ._transforms import EmptyTransform
from ...core import register, GLOBAL_CONFIG


__all__ = ['DualCompose']


# Ops applied INDEPENDENTLY to each modality (different colour/stats per sensor).
_PER_MODALITY_OPS = {
    'RandomPhotometricDistort',
    'ConvertPILImage',
    'Normalize',
}

# Ops applied ONLY to IR; RGB is passed through unchanged.
# Used for thermal-sensor-specific augmentations.
_IR_ONLY_OPS = {
    'IRChannelNoise',
    'IRGlobalBrightness',
}

# Ops applied ONLY to RGB; IR is passed through unchanged.
# Used for nighttime-degradation augmentations that train IR reliance.
_RGB_ONLY_OPS = {
    'RGBNightSimulation',
}

# Ops that need BOTH streams simultaneously and implement forward_dual(rgb, ir, target).
# DualCompose calls transform.forward_dual() instead of the standard torchvision forward.
_DUAL_STREAM_OPS = {
    'ModalityDropout',
    'CrossModalCutOut',
    'SmallVehicleCopyPaste',
    'RandomRotate90',
}


def _snapshot_rng():
    """Snapshot torch CPU RNG state only.

    NOTE: We deliberately do NOT touch CUDA RNG. DataLoader workers run as
    forked subprocesses; calling torch.cuda.* inside them triggers
    "Cannot re-initialize CUDA in forked subprocess". Image transforms
    use CPU random ops only, so CPU RNG is sufficient.
    """
    return torch.get_rng_state()


def _restore_rng(state):
    torch.set_rng_state(state)


@register()
class DualCompose(T.Compose):
    """Drop-in replacement for `Compose` that processes paired (rgb, ir).

    Forward expects:
        ((rgb, ir), target, dataset)
    and returns the same shape with all transforms applied identically (for
    geometric ops) or per-modality (for colour ops).
    """

    def __init__(self, ops, policy=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transform = getattr(
                        GLOBAL_CONFIG[name]['_pymodule'],
                        GLOBAL_CONFIG[name]['_name']
                    )(**op)
                    transforms.append(transform)
                    op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError(f'Unsupported transform spec: {type(op)}')
        else:
            transforms = [EmptyTransform()]

        super().__init__(transforms=transforms)
        self.policy = policy if policy is not None else {'name': 'default'}
        self.global_samples = 0

    # ------------------------------------------------------------------

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name: str):
        return {
            'default':     self.default_forward,
            'stop_epoch':  self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }[name]

    # ------------------------------------------------------------------

    def _apply_transform(self, transform, rgb, ir, target, dataset):
        """Apply one transform to (rgb, ir, target). See module docstring."""
        op_name = type(transform).__name__

        # ─── per-modality ops (colour / conversion) ──────────────────
        if op_name in _PER_MODALITY_OPS:
            rgb_out = transform(rgb, target, dataset)
            ir_out  = transform(ir,  target, dataset)

            new_rgb    = rgb_out[0] if isinstance(rgb_out, (tuple, list)) else rgb_out
            new_ir     = ir_out[0]  if isinstance(ir_out,  (tuple, list)) else ir_out
            new_target = rgb_out[1] if isinstance(rgb_out, (tuple, list)) and len(rgb_out) > 1 else target
            return new_rgb, new_ir, new_target

        # ─── IR-only ops (thermal sensor simulation) ─────────────────
        if op_name in _IR_ONLY_OPS:
            ir_out = transform(ir, target, dataset)
            new_ir = ir_out[0] if isinstance(ir_out, (tuple, list)) else ir_out
            return rgb, new_ir, target

        # ─── RGB-only ops (nighttime / lighting degradation) ─────────
        if op_name in _RGB_ONLY_OPS:
            rgb_out = transform(rgb, target, dataset)
            new_rgb    = rgb_out[0] if isinstance(rgb_out, (tuple, list)) else rgb_out
            new_target = rgb_out[1] if isinstance(rgb_out, (tuple, list)) and len(rgb_out) > 1 else target
            return new_rgb, ir, new_target

        # ─── dual-stream ops (need both rgb+ir, call forward_dual) ───
        if op_name in _DUAL_STREAM_OPS:
            return transform.forward_dual(rgb, ir, target)

        # ─── geometric / shared ops: synchronize RNG ─────────────────
        rng_state = _snapshot_rng()
        rgb_out = transform(rgb, target, dataset)
        if isinstance(rgb_out, (tuple, list)):
            new_rgb = rgb_out[0]
            new_target = rgb_out[1] if len(rgb_out) > 1 else target
        else:
            new_rgb, new_target = rgb_out, target

        # Re-run with identical RNG state on IR. Use a deepcopy of the
        # original target so the IR pass doesn't double-modify boxes.
        _restore_rng(rng_state)
        ir_target_dummy = copy.deepcopy(target)
        ir_out = transform(ir, ir_target_dummy, dataset)
        new_ir = ir_out[0] if isinstance(ir_out, (tuple, list)) else ir_out

        return new_rgb, new_ir, new_target

    # ------------------------------------------------------------------

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        (rgb, ir), target, dataset = sample
        for t in self.transforms:
            rgb, ir, target = self._apply_transform(t, rgb, ir, target, dataset)
        return (rgb, ir), target, dataset

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        (rgb, ir), target, dataset = sample

        cur_epoch    = dataset.epoch
        policy_ops   = self.policy['ops']
        policy_epoch = self.policy['epoch']

        for t in self.transforms:
            if type(t).__name__ in policy_ops and cur_epoch >= policy_epoch:
                continue
            rgb, ir, target = self._apply_transform(t, rgb, ir, target, dataset)

        return (rgb, ir), target, dataset

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        (rgb, ir), target, dataset = sample

        policy_ops    = self.policy['ops']
        policy_sample = self.policy['sample']

        for t in self.transforms:
            if type(t).__name__ in policy_ops and self.global_samples >= policy_sample:
                continue
            rgb, ir, target = self._apply_transform(t, rgb, ir, target, dataset)

        self.global_samples += 1
        return (rgb, ir), target, dataset
