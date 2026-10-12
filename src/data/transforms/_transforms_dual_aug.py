# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Domain-targeted augmentations for RGB+IR drone vehicle detection.

Literature basis
----------------
UA-CMDet (DroneVehicle, IEEE TGRS 2022):
    Modality uncertainty weighting based on IoU and illumination.

DMDTM (DroneVehicle, Sensors 2025):
    Dynamic Modality Dropout + Threshold Masking:  10% region masking per
    modality independently. Prevents static fusion from over-relying on
    one stream under varying scene conditions.

Modality Dropout survey (MICCAI/NeurIPS multimodal):
    p=0.3 per-modality dropout with 10-epoch curriculum warmup consistently
    improves RGB+IR fusion by forcing models to learn IR-independent and
    RGB-independent representations.

DroneVehicle mosaic augmentation (UA-CMDet supplementary):
    IR images: random edge enhancement + random blur (no hue/saturation).
    RGB images: mosaic + random brightness (minimal color jitter).

Data analysis (this project):
    AP@small = 0.30 vs AP@medium = 0.60 is the largest gap.
    val/test distribution is similar (no night shift), gap is likely
    caused by sparse small-vehicle training signal and val over-fitting.

Augmentation categories and DualCompose integration
----------------------------------------------------
Geometric (synchronized):
    RandomVerticalFlip      — aerial top-down view is vertically symmetric

Per-modality (independent colour/noise for each stream):
    IRChannelNoise          — thermal sensor Gaussian noise
    IRGlobalBrightness      — temperature-drift multiplicative shift

RGB-only (degrades RGB, keeps IR clean):
    RGBNightSimulation      — darken + noise + desaturation, teaches fusion
                              to rely on IR when RGB quality drops

Full-stream (applied to both, for distribution robustness):
    ModalityDropout         — zero entire RGB or IR with prob p each;
                              guarantees at least one stream remains.
                              Addresses static fusion over-reliance.
    CrossModalCutOut        — mask the SAME spatial region in BOTH streams
                              independently, forcing each modality to
                              reconstruct context from complementary stream.

Small-object (geometry, synchronized):
    SmallVehicleCopyPaste   — paste small-vehicle crops from within the
                              same batch to increase small-instance density.
                              Targets AP@small=0.30 bottleneck.
"""

import random
import copy
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from .._misc import Image, BoundingBoxes, convert_to_tv_tensor
from ...core import register

__all__ = [
    'RandomVerticalFlip',
    'IRChannelNoise',
    'IRGlobalBrightness',
    'RGBNightSimulation',
    'ModalityDropout',
    'CrossModalCutOut',
    'SmallVehicleCopyPaste',
]


# ---------------------------------------------------------------------------
# Geometric (synchronized) — DualCompose handles automatically
# ---------------------------------------------------------------------------

@register()
class RandomVerticalFlip(T.RandomVerticalFlip):
    """Vertical flip synchronized across RGB and IR.

    Aerial drone imagery is captured looking straight down; vertical flip
    is geometrically valid and adds orientation diversity missing from
    horizontal-flip-only pipelines.
    """
    pass


# ---------------------------------------------------------------------------
# IR-only  →  listed in _IR_ONLY_OPS in _transforms_dual.py
# ---------------------------------------------------------------------------

@register()
class IRChannelNoise(T.Transform):
    """Additive Gaussian noise on the IR tensor only.

    Simulates thermal sensor noise and cross-capture calibration variance.
    Applied only to the IR stream; RGB passes through unchanged.

    Args:
        std_range: (min_std, max_std) noise std in normalized [0,1] range.
        p: probability of applying noise per image.
    """
    _transformed_types = (Image, torch.Tensor)

    def __init__(self, std_range=(0.005, 0.025), p=0.5) -> None:
        super().__init__()
        self.std_min, self.std_max = std_range
        self.p = p

    def _get_params(self, flat_inputs):
        apply = torch.rand(1).item() < self.p
        std = (torch.rand(1).item() * (self.std_max - self.std_min) + self.std_min) if apply else 0.
        return {'apply': apply, 'std': std}

    def make_params(self, flat_inputs):
        return self._get_params(flat_inputs)

    def _transform(self, inpt, params):
        if not params['apply']:
            return inpt
        return (inpt + torch.randn_like(inpt) * params['std']).clamp(0., 1.)

    def transform(self, inpt, params):
        return self._transform(inpt, params)


@register()
class IRGlobalBrightness(T.Transform):
    """Multiplicative brightness shift on the IR tensor only.

    Thermal cameras exhibit global temperature-dependent intensity drift
    across scenes and times of day. A uniform scale in [scale_range]
    simulates this, complementing IRChannelNoise's local variance.

    Args:
        scale_range: (min, max) multiplicative factor.
        p: probability of applying per image.
    """
    _transformed_types = (Image, torch.Tensor)

    def __init__(self, scale_range=(0.7, 1.3), p=0.4) -> None:
        super().__init__()
        self.scale_min, self.scale_max = scale_range
        self.p = p

    def _get_params(self, flat_inputs):
        apply = torch.rand(1).item() < self.p
        scale = (torch.rand(1).item() * (self.scale_max - self.scale_min) + self.scale_min) if apply else 1.
        return {'apply': apply, 'scale': scale}

    def make_params(self, flat_inputs):
        return self._get_params(flat_inputs)

    def _transform(self, inpt, params):
        if not params['apply']:
            return inpt
        return (inpt * params['scale']).clamp(0., 1.)

    def transform(self, inpt, params):
        return self._transform(inpt, params)


# ---------------------------------------------------------------------------
# RGB-only  →  listed in _RGB_ONLY_OPS in _transforms_dual.py
# ---------------------------------------------------------------------------

@register()
class RGBNightSimulation(T.Transform):
    """Simulate nighttime RGB degradation on the RGB tensor only.

    In drone night missions the RGB camera is underexposed and noisy while
    the IR thermal camera remains clear. Training with this augmentation
    teaches the fusion module to fall back on IR when RGB quality drops.

    Three degradation steps:
        1. Darken: multiply by gamma sampled from gamma_range (< 1 = darker).
        2. Noise:  Gaussian with std = (1 - gamma) * noise_scale.
        3. Desaturate: blend toward grayscale with probability desaturate_p,
           simulating color information loss in low light.

    IR passes through unchanged, so the model must learn to rely on IR.

    Args:
        gamma_range: (min, max) brightness multiplier. Night: 0.2–0.75.
        noise_scale: noise std = (1 - gamma) * noise_scale.
        desaturate_p: probability of partial desaturation.
        p: probability of applying this augmentation at all.
    """
    _transformed_types = (Image, torch.Tensor)

    def __init__(self, gamma_range=(0.2, 0.75), noise_scale=0.04,
                 desaturate_p=0.4, p=0.3) -> None:
        super().__init__()
        self.gamma_min, self.gamma_max = gamma_range
        self.noise_scale  = noise_scale
        self.desaturate_p = desaturate_p
        self.p = p

    def _get_params(self, flat_inputs):
        if torch.rand(1).item() >= self.p:
            return {'apply': False}
        gamma     = torch.rand(1).item() * (self.gamma_max - self.gamma_min) + self.gamma_min
        noise_std = (1. - gamma) * self.noise_scale
        desat     = torch.rand(1).item() < self.desaturate_p
        desat_alpha = (torch.rand(1).item() * 0.7 + 0.3) if desat else 0.
        return {'apply': True, 'gamma': gamma, 'noise_std': noise_std,
                'desat': desat, 'desat_alpha': desat_alpha}

    def make_params(self, flat_inputs):
        return self._get_params(flat_inputs)

    def _transform(self, inpt, params):
        if not params['apply']:
            return inpt
        x = inpt * params['gamma']
        x = x + torch.randn_like(x) * params['noise_std']
        if params['desat'] and x.shape[0] == 3:
            gray = x.mean(dim=0, keepdim=True).expand_as(x)
            x = x * (1. - params['desat_alpha']) + gray * params['desat_alpha']
        return x.clamp(0., 1.)

    def transform(self, inpt, params):
        return self._transform(inpt, params)


# ---------------------------------------------------------------------------
# Full-stream augmentations — handled specially by DualCompose via
# _DUAL_STREAM_OPS set; DualCompose calls transform.forward_dual(rgb, ir, target)
# ---------------------------------------------------------------------------

@register()
class ModalityDropout:
    """Randomly zero entire RGB or IR stream during training.

    Based on:
        DMDTM (DroneVehicle, Sensors 2025): Dynamic Modality Dropout
        Multimodal dropout literature: p=0.3 per modality with curriculum warmup

    Forces the fusion module to learn representations that do not degenerate
    when one stream is missing, directly addressing static-fusion over-reliance.

    At most one stream is dropped per sample (i.e., if p_rgb fires, p_ir is
    skipped, and vice versa), ensuring at least one modality remains active.

    Args:
        p_rgb: probability of zeroing the RGB stream.
        p_ir:  probability of zeroing the IR stream.
               If p_rgb + p_ir > 1, they are renormalized.
        fill:  fill value for zeroed stream (0.0 = black, 0.5 = grey).
        curriculum_epochs: linearly ramp both probs from 0 to their
               target values over this many epochs (0 = no ramp).
    """

    def __init__(self, p_rgb: float = 0.15, p_ir: float = 0.15,
                 fill: float = 0.0, curriculum_epochs: int = 10) -> None:
        self.p_rgb = p_rgb
        self.p_ir  = p_ir
        self.fill  = fill
        self.curriculum_epochs = curriculum_epochs
        self._current_epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def _effective_p(self) -> Tuple[float, float]:
        if self.curriculum_epochs <= 0:
            return self.p_rgb, self.p_ir
        scale = min(1.0, self._current_epoch / self.curriculum_epochs)
        return self.p_rgb * scale, self.p_ir * scale

    def forward_dual(self, rgb: torch.Tensor, ir: torch.Tensor, target):
        """Called by DualCompose for _DUAL_STREAM_OPS."""
        p_rgb, p_ir = self._effective_p()
        r = torch.rand(1).item()
        if r < p_rgb:
            rgb = torch.full_like(rgb, self.fill)
        elif r < p_rgb + p_ir:
            ir = torch.full_like(ir, self.fill)
        return rgb, ir, target

    def __call__(self, *inputs):
        # Fallback for single-stream usage (should not normally be called).
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class RandomRotate90:
    """Randomly rotate both RGB and IR streams by 90 / 180 / 270 degrees.

    Rotation is exact for square images (no bbox approximation).  Applied
    synchronously to both modalities via DualCompose's forward_dual path.

    Args:
        p: probability of applying any rotation (0 = never, 1 = always).
           When triggered, one of {90°, 180°, 270°} is chosen uniformly.
    """

    def __init__(self, p: float = 0.75):
        self.p = p

    def forward_dual(self, rgb: torch.Tensor, ir: torch.Tensor, target: dict):
        if random.random() > self.p:
            return rgb, ir, target

        k = random.randint(1, 3)          # 90° / 180° / 270° CCW

        rgb = torch.rot90(rgb, k, dims=[-2, -1])
        ir  = torch.rot90(ir,  k, dims=[-2, -1])

        target = dict(target)
        boxes = target.get('boxes', None)
        if boxes is not None and len(boxes) > 0:
            raw = boxes.data if hasattr(boxes, 'data') else boxes.as_subclass(torch.Tensor)
            H, W = boxes.canvas_size          # (H, W) before rotation
            x1, y1, x2, y2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

            if k == 1:                        # 90° CCW: (x,y)→(y, W-x)
                nx1, ny1, nx2, ny2 = y1, W - x2, y2, W - x1
                new_canvas = (W, H)
            elif k == 2:                      # 180°: (x,y)→(W-x, H-y)
                nx1, ny1, nx2, ny2 = W - x2, H - y2, W - x1, H - y1
                new_canvas = (H, W)
            else:                             # 270° CCW = 90° CW: (x,y)→(H-y, x)
                nx1, ny1, nx2, ny2 = H - y2, x1, H - y1, x2
                new_canvas = (W, H)

            new_raw = torch.stack([nx1, ny1, nx2, ny2], dim=1).clamp(min=0)
            nH, nW = new_canvas
            new_raw[:, 0::2] = new_raw[:, 0::2].clamp(max=nW)
            new_raw[:, 1::2] = new_raw[:, 1::2].clamp(max=nH)

            target['boxes'] = convert_to_tv_tensor(new_raw, key='boxes',
                                                    box_format='XYXY',
                                                    spatial_size=new_canvas)
        return rgb, ir, target

    def __call__(self, *inputs):
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class CrossModalCutOut:
    """Mask the same spatial region in BOTH RGB and IR streams simultaneously.

    This is different from per-modality CutOut: here, the SAME (x,y,w,h)
    patch is zeroed in both streams. This forces the model to use the
    complementary stream's context for that region, strengthening cross-modal
    reasoning rather than per-modality robustness.

    Based on:
        DMDTM threshold masking: 10% area masking per modality independently.
        Here we mask the same region in both (stricter cross-modal signal).

    Args:
        min_cover: minimum fraction of image area to mask.
        max_cover: maximum fraction of image area to mask.
        p: probability of applying per image.
        fill: fill value (0. = black).
    """

    def __init__(self, min_cover: float = 0.05, max_cover: float = 0.15,
                 p: float = 0.4, fill: float = 0.) -> None:
        self.min_cover = min_cover
        self.max_cover = max_cover
        self.p = p
        self.fill = fill

    def forward_dual(self, rgb: torch.Tensor, ir: torch.Tensor, target):
        if torch.rand(1).item() >= self.p:
            return rgb, ir, target

        _, H, W = rgb.shape
        cover = torch.rand(1).item() * (self.max_cover - self.min_cover) + self.min_cover
        area  = H * W * cover

        # Sample aspect ratio uniformly in log-space
        log_ratio = torch.zeros(1).uniform_(torch.log(torch.tensor(0.3)),
                                            torch.log(torch.tensor(3.3))).item()
        ratio = torch.exp(torch.tensor(log_ratio)).item()
        bh = min(H, int((area * ratio) ** 0.5))
        bw = min(W, int((area / ratio) ** 0.5))

        if bh < 1 or bw < 1:
            return rgb, ir, target

        y0 = random.randint(0, H - bh)
        x0 = random.randint(0, W - bw)

        rgb = rgb.clone()
        ir  = ir.clone()
        rgb[:, y0:y0+bh, x0:x0+bw] = self.fill
        ir[:, y0:y0+bh, x0:x0+bw]  = self.fill
        return rgb, ir, target

    def __call__(self, *inputs):
        return inputs if len(inputs) > 1 else inputs[0]


# ---------------------------------------------------------------------------
# Small-object augmentation — dual-stream, in _DUAL_STREAM_OPS
# ---------------------------------------------------------------------------

@register()
class SmallVehicleCopyPaste:
    """Within-image copy-paste for small vehicle instances.

    Data analysis: AP@small = 0.30 vs AP@medium = 0.60 is the largest gap.
    Bus@small is essentially unusable (AP = 0.05). Copy-pasting small crops
    to new locations increases the effective density of small instances the
    model sees per image, without requiring cross-batch coordination.

    Procedure:
        1. Find all boxes with area < small_threshold in the current image.
        2. Sample up to max_paste of them.
        3. For each, paste the SAME crop at a random location in both RGB
           and IR (preserving alignment), then append the new box + label.
        4. Run a lightweight IoU-based deduplication to avoid pasting on top
           of an existing vehicle (suppress paste if IoU > iou_suppress).

    Placement in the YAML pipeline:
        After Resize + ConvertPILImage (images are float tensors, boxes are
        BoundingBoxes in XYXY pixel space), before ConvertBoxes.

    Args:
        small_threshold: max box area (px²) to qualify as "small". Default
            matches COCO's small category cutoff of 32²=1024.
        max_paste:   maximum pastes per image.
        p_per_box:   probability of pasting each selected small box.
        iou_suppress: skip paste if the new location overlaps any existing
            box with IoU > this threshold.
        jitter: randomly scale the crop by ±jitter fraction before pasting,
            keeping it "small" (scale stays within [1-jitter, 1+jitter]).
        p: probability of running this augmentation at all.
    """

    def __init__(self, small_threshold: int = 1024, max_paste: int = 6,
                 p_per_box: float = 0.8, iou_suppress: float = 0.3,
                 jitter: float = 0.15, p: float = 0.5) -> None:
        self.small_threshold = small_threshold
        self.max_paste       = max_paste
        self.p_per_box       = p_per_box
        self.iou_suppress    = iou_suppress
        self.jitter          = jitter
        self.p               = p

    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """IoU of a single box [4] against N boxes [N,4], all XYXY."""
        ix1 = torch.maximum(box[0], boxes[:, 0])
        iy1 = torch.maximum(box[1], boxes[:, 1])
        ix2 = torch.minimum(box[2], boxes[:, 2])
        iy2 = torch.minimum(box[3], boxes[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_box   = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        return inter / union.clamp(min=1e-6)

    def forward_dual(self, rgb: torch.Tensor, ir: torch.Tensor, target: dict):
        if torch.rand(1).item() >= self.p:
            return rgb, ir, target

        boxes  = target.get('boxes', None)
        labels = target.get('labels', None)
        if boxes is None or len(boxes) == 0:
            return rgb, ir, target

        # boxes are BoundingBoxes (XYXY, pixel space) at this stage
        raw = boxes.data if hasattr(boxes, 'data') else boxes.as_subclass(torch.Tensor)
        canvas_size = boxes.canvas_size  # (H, W)
        H, W = canvas_size

        areas = (raw[:, 2] - raw[:, 0]) * (raw[:, 3] - raw[:, 1])
        small_idx = torch.where(areas < self.small_threshold)[0]
        if len(small_idx) == 0:
            return rgb, ir, target

        # shuffle and cap
        perm = torch.randperm(len(small_idx))
        small_idx = small_idx[perm[:self.max_paste]]

        new_raw_boxes: List[torch.Tensor] = []
        new_labels:    List[torch.Tensor] = []

        current_boxes = raw.clone()  # grows as we add pastes (for IoU check)

        rgb = rgb.clone()
        ir  = ir.clone()

        for idx in small_idx:
            if torch.rand(1).item() >= self.p_per_box:
                continue

            x1, y1, x2, y2 = raw[idx].int().tolist()
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2:
                continue

            # Optional scale jitter (stays small: jitter at most ±jitter)
            if self.jitter > 0:
                scale = 1. + (torch.rand(1).item() * 2 - 1) * self.jitter
                new_bw = max(2, int(bw * scale))
                new_bh = max(2, int(bh * scale))
            else:
                new_bw, new_bh = bw, bh

            # Random paste location
            if W - new_bw <= 0 or H - new_bh <= 0:
                continue
            nx1 = random.randint(0, W - new_bw)
            ny1 = random.randint(0, H - new_bh)
            nx2 = nx1 + new_bw
            ny2 = ny1 + new_bh

            candidate = torch.tensor([nx1, ny1, nx2, ny2], dtype=raw.dtype)

            # Suppress if heavily overlapping an existing box
            if len(current_boxes) > 0:
                iou = self._iou(candidate, current_boxes)
                if iou.max().item() > self.iou_suppress:
                    continue

            # Extract crop (may need resize if jitter changed size)
            rgb_crop = rgb[:, y1:y2, x1:x2]
            ir_crop  = ir[:,  y1:y2, x1:x2]
            if new_bw != bw or new_bh != bh:
                rgb_crop = F.interpolate(rgb_crop.unsqueeze(0), size=(new_bh, new_bw),
                                         mode='bilinear', align_corners=False).squeeze(0)
                ir_crop  = F.interpolate(ir_crop.unsqueeze(0),  size=(new_bh, new_bw),
                                         mode='bilinear', align_corners=False).squeeze(0)

            rgb[:, ny1:ny2, nx1:nx2] = rgb_crop
            ir[:,  ny1:ny2, nx1:nx2] = ir_crop

            new_raw_boxes.append(candidate)
            new_labels.append(labels[idx])
            current_boxes = torch.cat([current_boxes, candidate.unsqueeze(0)], dim=0)

        if not new_raw_boxes:
            return rgb, ir, target

        all_raw  = torch.cat([raw, torch.stack(new_raw_boxes)], dim=0)
        all_lbls = torch.cat([labels, torch.stack(new_labels)], dim=0)

        target = dict(target)
        target['boxes']  = convert_to_tv_tensor(all_raw, key='boxes',
                                                 box_format='XYXY',
                                                 spatial_size=canvas_size)
        target['labels'] = all_lbls
        return rgb, ir, target

    def __call__(self, *inputs):
        return inputs if len(inputs) > 1 else inputs[0]
