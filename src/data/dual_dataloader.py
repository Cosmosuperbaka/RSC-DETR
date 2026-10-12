# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Dual-stream collate function for RGB+IR batched detection.

Copyright(c) 2025. All Rights Reserved.

After `DroneVehicleDetection.__getitem__` and `DualCompose`, each sample is:

    sample[0] = (rgb_tensor, ir_tensor)   # both (3, H, W), float32, normalised
    sample[1] = target dict

We collate into a dict-of-batches:

    samples = {'rgb': (B, 3, H, W), 'ir': (B, 3, H, W)}
    targets = list of B target dicts

Multi-scale training (random size resize per batch) is also supported here,
mirroring `BatchImageCollateFunction`.

The downstream `det_engine.train_one_epoch` will check whether `samples` is
a dict and route to the dual-stream model accordingly.
"""

import random

import torch
import torch.nn.functional as F

from .dataloader import BaseCollateFunction
from ..core import register


__all__ = [
    'DualBatchImageCollateFunction',
    'DualBatchImagePadCollateFunction',
    'dual_batch_image_collate_fn',
]


@register()
def dual_batch_image_collate_fn(items):
    """Stateless dual-stream collate (no multi-scale)."""
    rgb = torch.cat([s[0][0][None] for s in items], dim=0)
    ir  = torch.cat([s[0][1][None] for s in items], dim=0)
    targets = [s[1] for s in items]
    return {'rgb': rgb, 'ir': ir}, targets


@register()
class DualBatchImageCollateFunction(BaseCollateFunction):
    """Dual-stream collate with optional multi-scale resizing.

    Args:
        scales (List[int] | None): if provided, randomly pick one and resize
            both rgb and ir to that size each batch (until `stop_epoch`).
        stop_epoch (int | None):   epoch at which to stop multi-scale.
    """

    def __init__(self, scales=None, stop_epoch=None) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100_000_000

    def __call__(self, items):
        rgb = torch.cat([s[0][0][None] for s in items], dim=0)
        ir  = torch.cat([s[0][1][None] for s in items], dim=0)
        targets = [s[1] for s in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            rgb = F.interpolate(rgb, size=sz)
            ir  = F.interpolate(ir,  size=sz)
            if 'masks' in targets[0]:
                # Not supported for DroneVehicle (detection only)
                raise NotImplementedError('Mask resizing not supported for dual stream.')

        return {'rgb': rgb, 'ir': ir}, targets


@register()
class DualBatchImagePadCollateFunction(BaseCollateFunction):
    """Dual-stream native-resolution collate with right/bottom padding.

    Images keep their per-sample geometry until collation.  The batch tensor is
    padded on the right and bottom to the batch maximum H/W, optionally rounded
    up to a divisor.  Boxes produced by ``ConvertBoxes(fmt='cxcywh',
    normalize=True)`` are re-normalized from each original image size to the
    padded tensor size so model training remains in the input tensor coordinate
    system.  ``orig_size`` is preserved for COCO evaluation, while ``pad_size``
    is used by the postprocessor to map predictions back from padded pixels.
    """

    def __init__(self, size_divisibility=32, fill=0.0, adjust_normalized_boxes=True, scales=None, stop_epoch=None) -> None:
        super().__init__()
        self.size_divisibility = int(size_divisibility or 1)
        self.fill = float(fill)
        self.adjust_normalized_boxes = bool(adjust_normalized_boxes)

    @staticmethod
    def _round_up(value, divisor):
        if divisor <= 1:
            return int(value)
        return int((int(value) + divisor - 1) // divisor * divisor)

    def __call__(self, items):
        rgbs = [s[0][0] for s in items]
        irs = [s[0][1] for s in items]
        targets = [s[1] for s in items]

        shapes = [tuple(rgb.shape[-2:]) for rgb in rgbs]
        ir_shapes = [tuple(ir.shape[-2:]) for ir in irs]
        if shapes != ir_shapes:
            raise ValueError(f'RGB/IR shapes differ before padding: rgb={shapes}, ir={ir_shapes}')

        max_h = self._round_up(max(h for h, _ in shapes), self.size_divisibility)
        max_w = self._round_up(max(w for _, w in shapes), self.size_divisibility)

        def _new_batch(tensors):
            first = tensors[0]
            batch = first.new_full((len(tensors), first.shape[0], max_h, max_w), self.fill)
            for idx, tensor in enumerate(tensors):
                h, w = tensor.shape[-2:]
                batch[idx, :, :h, :w] = tensor
            return batch

        rgb = _new_batch(rgbs)
        ir = _new_batch(irs)
        valid_mask = torch.zeros((len(items), max_h, max_w), dtype=torch.bool)

        for target, (h, w) in zip(targets, shapes):
            target['pad_size'] = torch.as_tensor([max_w, max_h], dtype=torch.int64)
            target['valid_size'] = torch.as_tensor([w, h], dtype=torch.int64)
            if self.adjust_normalized_boxes and 'boxes' in target and target['boxes'].numel() > 0:
                scale = target['boxes'].new_tensor([w / max_w, h / max_h, w / max_w, h / max_h])
                target['boxes'] = target['boxes'] * scale

        for idx, (h, w) in enumerate(shapes):
            valid_mask[idx, :h, :w] = True

        return {'rgb': rgb, 'ir': ir, 'valid_mask': valid_mask}, targets
