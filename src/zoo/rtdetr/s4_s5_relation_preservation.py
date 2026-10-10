# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.ops import roi_align

def _target_boxes_xyxy_pixels(target: Dict[str, Any], image_h: int, image_w: int) -> torch.Tensor:
    boxes = target.get("boxes")
    if boxes is None:
        return torch.zeros((0, 4), device=target["labels"].device, dtype=torch.float32)
    boxes = boxes.to(dtype=torch.float32)
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    if float(boxes.detach().max()) <= 1.5:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([
            (cx - 0.5 * w) * image_w,
            (cy - 0.5 * h) * image_h,
            (cx + 0.5 * w) * image_w,
            (cy + 0.5 * h) * image_h,
        ], dim=-1)
    return boxes


class S4S5RelationPreservationLoss(nn.Module):
    """Training-only object-relation preservation from fused S4 to fused S5.

    This module has no parameters. It consumes fused features after RGB/IR fusion
    and before the HybridEncoder, pools GT object ROIs, and matches the object
    relation matrix at S5 to the detached S4 relation matrix.
    """

    def __init__(self, roi_size: int = 3, max_objects: int = 128, sample_seed: int = 3407):
        super().__init__()
        self.roi_size = int(roi_size)
        self.max_objects = int(max_objects)
        self.sample_seed = int(sample_seed)
        self.last_stats: Dict[str, float] = {}
        self.last_indices: List[int] = []

    def forward(
        self,
        fused_s4: torch.Tensor,
        fused_s5: torch.Tensor,
        targets: List[Dict[str, Any]],
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        loss, stats, indices = self.compute(fused_s4, fused_s5, targets, image_hw)
        self.last_stats = stats
        self._gate_stats = dict(stats)
        self.last_indices = indices.detach().cpu().tolist() if indices.numel() else []
        return loss

    def compute(
        self,
        fused_s4: torch.Tensor,
        fused_s5: torch.Tensor,
        targets: List[Dict[str, Any]],
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        rois4, rois5, labels = self._build_rois(fused_s4, fused_s5, targets, image_hw)
        if rois4.shape[0] > self.max_objects:
            idx = self._deterministic_indices(rois4.shape[0], fused_s4.device)
            rois4 = rois4[idx]
            rois5 = rois5[idx]
            labels = labels[idx]
        else:
            idx = torch.arange(rois4.shape[0], device=fused_s4.device, dtype=torch.long)
        if rois4.shape[0] < 2:
            z = fused_s5.sum() * 0.0
            return z, {
                "s4s5_rel_raw": 0.0,
                "s4s5_rel_pairs": 0.0,
                "s4s5_rel_error": 0.0,
            }, idx

        z4 = self._pool(fused_s4, rois4)
        z5 = self._pool(fused_s5, rois5)
        with torch.autocast(device_type=fused_s5.device.type, enabled=False):
            z4f = F.normalize(z4.float(), dim=1)
            z5f = F.normalize(z5.float(), dim=1)
            r4 = z4f @ z4f.t()
            r5 = z5f @ z5f.t()
            mask = ~torch.eye(r4.shape[0], dtype=torch.bool, device=r4.device)
            loss = F.smooth_l1_loss(r5[mask], r4.detach()[mask], reduction="mean")
            err = (r5[mask] - r4.detach()[mask]).abs()
            same = labels[:, None].eq(labels[None, :]) & mask
            diff = labels[:, None].ne(labels[None, :]) & mask
            stats = {
                "s4s5_rel_raw": float(loss.detach().cpu()),
                "s4s5_rel_pairs": float(mask.sum().detach().cpu()),
                "s4s5_rel_error": float(err.mean().detach().cpu()),
                "s4_relation_mean": float(r4[mask].detach().mean().cpu()),
                "s4_relation_std": float(r4[mask].detach().std().cpu()),
                "s5_relation_mean": float(r5[mask].detach().mean().cpu()),
                "s5_relation_std": float(r5[mask].detach().std().cpu()),
                "same_class_relation_error": float((r5[same] - r4.detach()[same]).abs().mean().detach().cpu()) if same.any() else 0.0,
                "different_class_relation_error": float((r5[diff] - r4.detach()[diff]).abs().mean().detach().cpu()) if diff.any() else 0.0,
            }
        return loss.to(dtype=fused_s5.dtype), stats, idx

    def _pool(self, feat: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        pooled = roi_align(feat, rois, output_size=(self.roi_size, self.roi_size), spatial_scale=1.0, aligned=True)
        return pooled.flatten(2).mean(dim=-1)

    def _build_rois(
        self,
        f4: torch.Tensor,
        f5: torch.Tensor,
        targets: List[Dict[str, Any]],
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_h, image_w = image_hw
        h4, w4 = f4.shape[-2:]
        h5, w5 = f5.shape[-2:]
        rois4, rois5, labels = [], [], []
        for b, target in enumerate(targets):
            boxes = _target_boxes_xyxy_pixels(target, image_h, image_w)
            labs = target["labels"].to(dtype=torch.long)
            for box, lab in zip(boxes, labs):
                x1, y1, x2, y2 = box
                if (x2 <= x1) or (y2 <= y1):
                    continue
                rois4.append([b, x1 * w4 / image_w, y1 * h4 / image_h, x2 * w4 / image_w, y2 * h4 / image_h])
                rois5.append([b, x1 * w5 / image_w, y1 * h5 / image_h, x2 * w5 / image_w, y2 * h5 / image_h])
                labels.append(lab)
        if not rois4:
            dev = f4.device
            return (
                torch.zeros((0, 5), device=dev, dtype=torch.float32),
                torch.zeros((0, 5), device=dev, dtype=torch.float32),
                torch.zeros((0,), device=dev, dtype=torch.long),
            )
        return (
            torch.tensor(rois4, device=f4.device, dtype=torch.float32),
            torch.tensor(rois5, device=f5.device, dtype=torch.float32),
            torch.stack(labels).to(device=f4.device),
        )

    def _deterministic_indices(self, n: int, device: torch.device) -> torch.Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.sample_seed + int(n))
        idx = torch.randperm(n, generator=gen)[: self.max_objects]
        return idx.sort().values.to(device=device)


