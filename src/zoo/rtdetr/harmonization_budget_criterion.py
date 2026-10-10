# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Criterion adapter for the annotation-budget state-driven autonomous PFHM/ACRHM coordinator."""


from __future__ import annotations

import json

import os

from pathlib import Path

import torch

import torch.nn.functional as F

from ...core import register

from .acrhm_criterion import ACRHMCriterion

from .box_ops import box_cxcywh_to_xyxy, box_iou

from .rtdetrv2_criterion import RTDETRCriterionv2

from .pfhm_annotation_statistics import PFHMAnnotationAnalyzer

from .pfhm import PFHMLoss

from .dynamic_harmonization_coordinator import DynamicHarmonizationCoordinator

@register()
class BudgetHarmonizationCriterion(ACRHMCriterion):
    """One shared, annotation/annotation-budget state-driven PFHM-ACRHM loss path."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher", "coordinator"]

    def __init__(
        self,
        *args,
        annotation_file,
        category_ids,
        total_updates,
        x1_num_classes=None,
        coordinator=None,
        diagnostics_path: str = "",
        log_interval: int = 500,
        class_axis_normalization_power: float = 0.0,
        quality_target_floor: float = 0.0,
        final_loss_scale: float = 1.0,
        quality_loss_weight: float = 0.0,
        quality_matched_only: bool = False,
        quality_loss_type: str = "bce",
        quality_residual_center: float = 0.8,
        quality_residual_tau: float = 0.2,
        quality_pairwise_loss_weight: float = 0.0,
        quality_pairwise_iou_gap: float = 0.05,
        quality_pairwise_max_pairs: int = 2048,
        score_rank_loss_weight: float = 0.0,
        score_rank_low_iou: float = 0.50,
        score_rank_high_iou: float = 0.75,
        score_rank_tau: float = 0.15,
        score_rank_max_pairs: int = 2048,
        private_x1_loss_weight: float = 0.0,
        private_x1_consistency_weight: float = 0.0,
        private_x1_min_iou: float = 0.50,
        selective_preserve_weight: float = 0.0,
        selective_preserve_iou_threshold: float = 0.75,
        self_anchor_preserve_weight: float = 0.0,
        self_anchor_preserve_iou_threshold: float = 0.75,
        self_anchor_preserve_ramp_epochs: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not isinstance(coordinator, DynamicHarmonizationCoordinator):
            raise TypeError("BudgetHarmonizationCriterion requires DynamicHarmonizationCoordinator")
        stats = PFHMAnnotationAnalyzer.stats_from_coco(annotation_file)
        decision = PFHMAnnotationAnalyzer.analyze(stats)
        supplied_ids = [int(value) for value in category_ids]
        x1_num_classes = len(supplied_ids) if x1_num_classes is None else int(x1_num_classes)
        if x1_num_classes != stats.num_classes or supplied_ids != stats.category_ids:
            raise ValueError("state-auto criterion category mapping disagrees with annotations")
        self.x1_loss = PFHMLoss(
            stats,
            decision,
            total_updates=int(total_updates),
            quality_alpha=self.alpha,
            quality_gamma=self.gamma,
            class_axis_normalization_power=class_axis_normalization_power,
            quality_target_floor=quality_target_floor,
            final_loss_scale=final_loss_scale,
        )
        self.coordinator = coordinator
        self.quality_loss_weight = float(quality_loss_weight)
        self.quality_matched_only = bool(quality_matched_only)
        self.quality_loss_type = str(quality_loss_type)
        self.quality_residual_center = float(quality_residual_center)
        self.quality_residual_tau = float(quality_residual_tau)
        self.quality_pairwise_loss_weight = float(quality_pairwise_loss_weight)
        self.quality_pairwise_iou_gap = float(quality_pairwise_iou_gap)
        self.quality_pairwise_max_pairs = int(quality_pairwise_max_pairs)
        self.score_rank_loss_weight = float(score_rank_loss_weight)
        self.score_rank_low_iou = float(score_rank_low_iou)
        self.score_rank_high_iou = float(score_rank_high_iou)
        self.score_rank_tau = float(score_rank_tau)
        self.score_rank_max_pairs = int(score_rank_max_pairs)
        self.private_x1_loss_weight = float(private_x1_loss_weight)
        self.private_x1_consistency_weight = float(private_x1_consistency_weight)
        self.private_x1_min_iou = float(private_x1_min_iou)
        self.selective_preserve_weight = float(selective_preserve_weight)
        self.selective_preserve_iou_threshold = float(selective_preserve_iou_threshold)
        self.self_anchor_preserve_weight = float(self_anchor_preserve_weight)
        self.self_anchor_preserve_iou_threshold = float(self_anchor_preserve_iou_threshold)
        self.self_anchor_preserve_ramp_epochs = max(int(self_anchor_preserve_ramp_epochs), 1)
        self.teacher_model = None
        self.diagnostics_path = str(diagnostics_path)
        self.log_interval = int(log_interval)
        self._controls = None
        self._x3_class_gate = None
        self._x3_loc_gate = None
        self.last_x1_diagnostics = {}
        print(
            "[BudgetHarmonizationCriterion] dataset-name-free annotation-budget coordinator; "
            f"total_updates={total_updates} diagnostics_path={self.diagnostics_path}",
            flush=True,
        )

    def _quality_loss(self, outputs, targets, indices):
        logits = outputs.get("pred_quality_logits")
        if logits is None or (self.quality_loss_weight <= 0 and self.quality_pairwise_loss_weight <= 0):
            return {}, {}
        pred_boxes = outputs["pred_boxes"]
        target = torch.zeros_like(logits, dtype=logits.dtype, device=logits.device)
        idx = self._get_src_permutation_idx(indices)
        matched_ious = None
        matched_labels = None
        if idx[0].numel() > 0:
            src_boxes = pred_boxes[idx].detach()
            tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0).to(src_boxes.device).detach()
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            matched_ious = torch.diag(ious).to(dtype=target.dtype).detach()
            target[idx] = matched_ious
            matched_labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)], dim=0).to(logits.device)
        losses = {}
        if self.quality_loss_weight > 0:
            if self.quality_loss_type == "residual_mse":
                if idx[0].numel() == 0:
                    loss = logits.sum() * 0.0
                else:
                    tau = max(self.quality_residual_tau, 1.0e-6)
                    residual_target = ((matched_ious - self.quality_residual_center) / tau).clamp(-0.95, 0.95)
                    raw_target = torch.atanh(residual_target)
                    loss = F.mse_loss(logits[idx], raw_target.to(dtype=logits.dtype), reduction="mean")
            elif self.quality_matched_only:
                if idx[0].numel() == 0:
                    loss = logits.sum() * 0.0
                else:
                    loss = F.binary_cross_entropy_with_logits(logits[idx], matched_ious, reduction="mean")
            else:
                loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
            losses["loss_quality"] = loss * self.quality_loss_weight
        pair_count = logits.new_tensor(0.0)
        pair_gap = logits.new_tensor(0.0)
        if self.quality_pairwise_loss_weight > 0:
            if idx[0].numel() == 0:
                pair_loss = logits.sum() * 0.0
            else:
                pair_loss, pair_count, pair_gap = self._quality_pairwise_loss(
                    logits[idx], matched_ious, matched_labels, idx[0]
                )
            losses["loss_quality_pairwise"] = pair_loss * self.quality_pairwise_loss_weight
        with torch.no_grad():
            pred_q = logits.detach().sigmoid()
            stats = {
                "quality_pred_mean": pred_q.mean(),
                "quality_target_mean": target.detach().mean(),
                "quality_matched_target_mean": target[idx].detach().mean() if idx[0].numel() > 0 else target.new_tensor(0.0),
                "quality_loss_raw": losses.get("loss_quality", logits.sum() * 0.0).detach(),
                "quality_pairwise_loss_raw": losses.get("loss_quality_pairwise", logits.sum() * 0.0).detach(),
                "quality_pairwise_pairs": pair_count.detach(),
                "quality_pairwise_mean_iou_gap": pair_gap.detach(),
            }
        return losses, stats

    def _quality_pairwise_loss(self, matched_logits, matched_ious, matched_labels, batch_indices):
        device = matched_logits.device
        matched_ious = matched_ious.to(device=device)
        matched_labels = matched_labels.to(device=device)
        batch_indices = batch_indices.to(device=device)
        pieces = []
        pair_counts = []
        pair_gaps = []
        max_pairs = max(self.quality_pairwise_max_pairs, 1)
        for image_id in batch_indices.unique(sorted=True):
            image_mask = batch_indices == image_id
            image_labels = matched_labels[image_mask]
            for label in image_labels.unique(sorted=True):
                mask = image_mask & (matched_labels == label)
                if int(mask.sum().item()) < 2:
                    continue
                q = matched_logits[mask]
                iou = matched_ious[mask]
                pair_mask = (iou[:, None] - iou[None, :]) > self.quality_pairwise_iou_gap
                high, low = pair_mask.nonzero(as_tuple=True)
                if high.numel() == 0:
                    continue
                if high.numel() > max_pairs:
                    keep = torch.randperm(high.numel(), device=high.device)[:max_pairs]
                    high = high[keep]
                    low = low[keep]
                diff = q[high] - q[low]
                pieces.append(F.softplus(-diff).mean())
                pair_counts.append(high.new_tensor(float(high.numel()), dtype=torch.float32))
                pair_gaps.append((iou[high] - iou[low]).detach().mean())
        if not pieces:
            zero = matched_logits.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        loss = torch.stack(pieces).mean()
        count = torch.stack(pair_counts).sum().to(device=matched_logits.device, dtype=matched_logits.dtype)
        gap = torch.stack(pair_gaps).mean().to(device=matched_logits.device, dtype=matched_logits.dtype)
        return loss, count, gap

    def _score_rank_loss(self, outputs, targets, indices):
        residuals = outputs.get("pred_score_calibration_logits")
        if residuals is None or self.score_rank_loss_weight <= 0:
            return {}, {}
        base_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        idx = self._get_src_permutation_idx(indices)
        zero = residuals.sum() * 0.0
        if idx[0].numel() == 0:
            return {"loss_score_rank": zero}, {
                "score_rank_loss_raw": zero.detach(),
                "score_rank_pairs": zero.detach(),
                "score_rank_high_score_mean": zero.detach(),
                "score_rank_mid_score_mean": zero.detach(),
            }

        src_boxes = pred_boxes[idx].detach()
        tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0).to(src_boxes.device).detach()
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        matched_ious = torch.diag(ious).to(dtype=base_logits.dtype).detach()
        matched_labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        matched_labels = matched_labels.to(device=base_logits.device, dtype=torch.long)
        matched_labels = matched_labels.clamp(min=0, max=base_logits.shape[-1] - 1)
        batch_indices = idx[0].to(device=base_logits.device)
        query_indices = idx[1].to(device=base_logits.device)

        gather_idx = matched_labels[:, None]
        base_score = base_logits[batch_indices, query_indices].sigmoid().gather(1, gather_idx).squeeze(1).detach()
        residual = residuals[batch_indices, query_indices].gather(1, gather_idx).squeeze(1)
        calibrated_score = base_score * torch.exp(self.score_rank_tau * torch.tanh(residual))

        pieces = []
        pair_counts = []
        high_scores = []
        mid_scores = []
        max_pairs = max(self.score_rank_max_pairs, 1)
        for image_id in batch_indices.unique(sorted=True):
            image_mask = batch_indices == image_id
            image_labels = matched_labels[image_mask]
            for label in image_labels.unique(sorted=True):
                mask = image_mask & (matched_labels == label)
                if int(mask.sum().item()) < 2:
                    continue
                local_scores = calibrated_score[mask]
                local_ious = matched_ious[mask]
                high_mask = local_ious >= self.score_rank_high_iou
                mid_mask = (local_ious >= self.score_rank_low_iou) & (local_ious < self.score_rank_high_iou)
                if int(high_mask.sum().item()) == 0 or int(mid_mask.sum().item()) == 0:
                    continue
                high = high_mask.nonzero(as_tuple=True)[0]
                mid = mid_mask.nonzero(as_tuple=True)[0]
                high_grid, mid_grid = torch.meshgrid(high, mid, indexing="ij")
                high_flat = high_grid.flatten()
                mid_flat = mid_grid.flatten()
                if high_flat.numel() > max_pairs:
                    keep = torch.randperm(high_flat.numel(), device=high_flat.device)[:max_pairs]
                    high_flat = high_flat[keep]
                    mid_flat = mid_flat[keep]
                diff = local_scores[high_flat] - local_scores[mid_flat]
                pieces.append(F.softplus(-diff).mean())
                pair_counts.append(high_flat.new_tensor(float(high_flat.numel()), dtype=torch.float32))
                high_scores.append(local_scores[high].detach().mean())
                mid_scores.append(local_scores[mid].detach().mean())

        if not pieces:
            loss = zero
            pair_count = zero.detach()
            high_mean = zero.detach()
            mid_mean = zero.detach()
        else:
            loss = torch.stack(pieces).mean()
            pair_count = torch.stack(pair_counts).sum().to(device=base_logits.device, dtype=base_logits.dtype)
            high_mean = torch.stack(high_scores).mean().to(device=base_logits.device, dtype=base_logits.dtype)
            mid_mean = torch.stack(mid_scores).mean().to(device=base_logits.device, dtype=base_logits.dtype)

        return {"loss_score_rank": loss * self.score_rank_loss_weight}, {
            "score_rank_loss_raw": loss.detach(),
            "score_rank_pairs": pair_count.detach(),
            "score_rank_high_score_mean": high_mean.detach(),
            "score_rank_mid_score_mean": mid_mean.detach(),
        }

    def _private_x1_loss(self, outputs, targets, indices):
        private_logits = outputs.get("x1_private_fine_logits")
        if private_logits is None or (self.private_x1_loss_weight <= 0 and self.private_x1_consistency_weight <= 0):
            return {}, {}
        main_logits = outputs["x1_fine_logits"]
        boxes = outputs["pred_boxes"]
        all_private, all_main, all_classes, all_iou = [], [], [], []
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if query_indices.numel() == 0:
                continue
            query_indices = query_indices.to(private_logits.device)
            target_indices = target_indices.to(private_logits.device)
            labels = targets[batch_index]["labels"][target_indices].to(private_logits.device)
            target_boxes = targets[batch_index]["boxes"][target_indices].to(private_logits.device)
            all_private.append(private_logits[batch_index, query_indices])
            all_main.append(main_logits[batch_index, query_indices])
            all_classes.append(self.x1_loss._map_labels(labels))
            all_iou.append(self.x1_loss._paired_iou(boxes[batch_index, query_indices].detach(), target_boxes).detach())
        zero = private_logits.sum() * 0.0
        if not all_private:
            return {"loss_private_x1": zero, "loss_private_x1_consistency": zero}, {
                "private_x1_loss_raw": zero.detach(),
                "private_x1_consistency_raw": zero.detach(),
                "private_x1_samples": zero.detach(),
                "private_x1_mean_iou": zero.detach(),
            }
        private = torch.cat(all_private)
        main = torch.cat(all_main)
        classes = torch.cat(all_classes)
        iou = torch.cat(all_iou).to(private.dtype).clamp(0.0, 1.0)
        keep = iou >= self.private_x1_min_iou
        if not bool(keep.any()):
            return {"loss_private_x1": zero, "loss_private_x1_consistency": zero}, {
                "private_x1_loss_raw": zero.detach(),
                "private_x1_consistency_raw": zero.detach(),
                "private_x1_samples": zero.detach(),
                "private_x1_mean_iou": zero.detach(),
            }
        private = private[keep]
        main = main[keep]
        classes = classes[keep]
        iou = iou[keep]
        quality_target = torch.zeros_like(private)
        quality_target.scatter_(1, classes[:, None], iou[:, None])
        pred_score = private.detach().sigmoid()
        quality_weight = (
            self.alpha * pred_score.pow(self.gamma) * (1.0 - quality_target)
            + quality_target
        )
        private_loss = F.binary_cross_entropy_with_logits(
            private,
            quality_target,
            weight=quality_weight,
            reduction="none",
        ).sum(dim=-1).mean()
        teacher = private.detach().softmax(dim=-1)
        consistency = F.kl_div(
            F.log_softmax(main, dim=-1),
            teacher,
            reduction="batchmean",
        )
        return {
            "loss_private_x1": private_loss * self.private_x1_loss_weight,
            "loss_private_x1_consistency": consistency * self.private_x1_consistency_weight,
        }, {
            "private_x1_loss_raw": private_loss.detach(),
            "private_x1_consistency_raw": consistency.detach(),
            "private_x1_samples": private.new_tensor(float(private.shape[0])).detach(),
            "private_x1_mean_iou": iou.detach().mean(),
        }

    def _selective_preserve_loss(self, outputs, targets, indices):
        if self.selective_preserve_weight <= 0:
            return {}, {}
        teacher_logits = outputs.get("teacher_pred_logits")
        if teacher_logits is None:
            raise KeyError("selective_preserve_weight > 0 requires teacher_pred_logits")
        student_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                f"teacher/student logits shape mismatch: {teacher_logits.shape} vs {student_logits.shape}"
            )
        preserve_mask = torch.ones(
            student_logits.shape[:2],
            device=student_logits.device,
            dtype=torch.bool,
        )
        matched_ious = []
        high_quality = student_logits.new_tensor(0.0)
        matched_count = student_logits.new_tensor(0.0)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            idx = (idx[0].to(student_logits.device), idx[1].to(student_logits.device))
            src_boxes = pred_boxes[idx].detach()
            tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
            tgt_boxes = tgt_boxes.to(src_boxes.device).detach()
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            diag = torch.diag(ious).detach()
            matched_ious.append(diag)
            high = diag >= self.selective_preserve_iou_threshold
            preserve_mask[idx[0][high], idx[1][high]] = False
            high_quality = high.to(dtype=student_logits.dtype).sum()
            matched_count = high.new_tensor(float(high.numel()), dtype=student_logits.dtype)

        per_query = F.binary_cross_entropy_with_logits(
            student_logits,
            teacher_logits.detach().sigmoid().to(dtype=student_logits.dtype),
            reduction="none",
        ).mean(dim=-1)
        selected = preserve_mask.to(dtype=per_query.dtype)
        denom = selected.sum().clamp_min(1.0)
        raw = (per_query * selected).sum() / denom
        loss = raw * self.selective_preserve_weight
        if matched_ious:
            iou_tensor = torch.cat(matched_ious).to(dtype=student_logits.dtype)
            matched_iou_mean = iou_tensor.mean()
        else:
            matched_iou_mean = student_logits.new_tensor(0.0)
        stats = {
            "preserve_loss_raw": raw.detach(),
            "preserve_selected_queries": selected.sum().detach(),
            "preserve_selected_ratio": selected.mean().detach(),
            "preserve_high_iou_exempt": high_quality.detach(),
            "preserve_matched_queries": matched_count.detach(),
            "preserve_matched_iou_mean": matched_iou_mean.detach(),
        }
        return {"loss_preserve": loss}, stats

    def _self_anchor_preserve_lambda(self, epoch) -> float:
        if self.self_anchor_preserve_weight <= 0:
            return 0.0
        try:
            epoch_value = int(epoch)
        except Exception:
            epoch_value = 0
        if epoch_value <= 0:
            return 0.0
        scale = min(float(epoch_value) / float(self.self_anchor_preserve_ramp_epochs), 1.0)
        return self.self_anchor_preserve_weight * scale

    def _self_anchor_preserve_loss(self, outputs, targets, indices, epoch):
        weight = self._self_anchor_preserve_lambda(epoch)
        if self.self_anchor_preserve_weight <= 0 and "base_pred_logits" not in outputs:
            return {}, {}
        student_logits = outputs["pred_logits"]
        base_logits = outputs.get("base_pred_logits")
        if base_logits is None:
            if self.self_anchor_preserve_weight > 0:
                raise KeyError("self_anchor_preserve_weight > 0 requires base_pred_logits")
            return {}, {}
        if base_logits.shape != student_logits.shape:
            raise ValueError(
                f"base/student logits shape mismatch: {base_logits.shape} vs {student_logits.shape}"
            )
        pred_boxes = outputs["pred_boxes"]
        preserve_mask = torch.ones(
            student_logits.shape[:2],
            device=student_logits.device,
            dtype=torch.bool,
        )
        matched_ious = []
        high_quality = student_logits.new_tensor(0.0)
        matched_count = student_logits.new_tensor(0.0)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            idx = (idx[0].to(student_logits.device), idx[1].to(student_logits.device))
            src_boxes = pred_boxes[idx].detach()
            tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
            tgt_boxes = tgt_boxes.to(src_boxes.device).detach()
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            diag = torch.diag(ious).detach()
            matched_ious.append(diag)
            high = diag >= self.self_anchor_preserve_iou_threshold
            preserve_mask[idx[0][high], idx[1][high]] = False
            high_quality = high.to(dtype=student_logits.dtype).sum()
            matched_count = high.new_tensor(float(high.numel()), dtype=student_logits.dtype)

        per_query = F.binary_cross_entropy_with_logits(
            student_logits,
            base_logits.detach().sigmoid().to(dtype=student_logits.dtype),
            reduction="none",
        ).mean(dim=-1)
        selected = preserve_mask.to(dtype=per_query.dtype)
        denom = selected.sum().clamp_min(1.0)
        raw = (per_query * selected).sum() / denom
        loss = raw * float(weight)
        if matched_ious:
            iou_tensor = torch.cat(matched_ious).to(dtype=student_logits.dtype)
            matched_iou_mean = iou_tensor.mean()
        else:
            matched_iou_mean = student_logits.new_tensor(0.0)
        stats = {
            "self_anchor_loss_raw": raw.detach(),
            "self_anchor_weight": student_logits.new_tensor(float(weight)).detach(),
            "self_anchor_selected_queries": selected.sum().detach(),
            "self_anchor_selected_ratio": selected.mean().detach(),
            "self_anchor_high_iou_exempt": high_quality.detach(),
            "self_anchor_matched_queries": matched_count.detach(),
            "self_anchor_matched_iou_mean": matched_iou_mean.detach(),
        }
        return {"loss_self_anchor_preserve": loss}, stats

    def _matched_role_weights(self, targets, indices, device, localization=False):
        labels = torch.cat([
            target["labels"][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]).to(device)
        if labels.numel() == 0:
            return labels, None
        classes = self.x3.map_labels(labels)
        if localization:
            weights = self.x3.localization_weights(device)[classes].detach()
            gate = self._x3_loc_gate if self._x3_loc_gate is not None else weights.new_tensor(0.0)
            return labels, 1.0 + gate.to(device=device, dtype=weights.dtype) * (weights - 1.0)
        if self._controls is None:
            return labels, torch.ones_like(labels, dtype=torch.float32, device=device)
        weights = self._controls["auto_x3_class_weight"].to(device=device, dtype=torch.float32)[classes]
        gate = self._x3_class_gate if self._x3_class_gate is not None else weights.new_tensor(1.0)
        return labels, 1.0 + gate.to(device=device, dtype=weights.dtype) * (weights - 1.0)

    @staticmethod
    def _scalar(value, device):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=torch.float32).mean()
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    def _write_diagnostics(self, update: int, record: dict) -> None:
        if not self.diagnostics_path or os.environ.get("RANK", "0") not in {"0", "-1"}:
            return
        path = Path(self.diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def forward(self, outputs, targets, **kwargs):
        if "x1_fine_logits" not in outputs:
            raise KeyError("BudgetHarmonizationCriterion requires PFHM decoder outputs")
        matching_inputs = {k: v for k, v in outputs.items() if "aux" not in k}
        matching_inputs = RTDETRCriterionv2._outputs_for_matcher(matching_inputs)
        if hasattr(self.matcher, "begin_step"):
            self.matcher.begin_step()
        controls = self.coordinator(self.x3, self.x1_loss, self.matcher, outputs["pred_logits"].device)
        self._controls = controls
        self._x3_class_gate = controls["auto_x3_class_gate"].to(outputs["pred_logits"].device)
        self._x3_loc_gate = controls["auto_x3_loc_gate"].to(outputs["pred_logits"].device)
        previous_external = getattr(self.matcher, "external_reliability_weights", None)
        if hasattr(self.matcher, "external_reliability_weights"):
            self.matcher.external_reliability_weights = controls["auto_matcher_weight"].detach()
        try:
            losses = RTDETRCriterionv2.forward(self, outputs, targets, **kwargs)
            indices = self.matcher(matching_inputs, targets)["indices"]
        finally:
            if hasattr(self.matcher, "external_reliability_weights"):
                self.matcher.external_reliability_weights = previous_external

        q_losses, q_stats = self._quality_loss(outputs, targets, indices)
        losses.update(q_losses)
        rank_losses, rank_stats = self._score_rank_loss(outputs, targets, indices)
        losses.update(rank_losses)
        private_x1_losses, private_x1_stats = self._private_x1_loss(outputs, targets, indices)
        losses.update(private_x1_losses)
        preserve_losses, preserve_stats = self._selective_preserve_loss(outputs, targets, indices)
        losses.update(preserve_losses)
        self_anchor_losses, self_anchor_stats = self._self_anchor_preserve_loss(
            outputs, targets, indices, kwargs.get("epoch", 0)
        )
        losses.update(self_anchor_losses)

        x1_outputs = dict(outputs)
        grad_gate = controls["auto_x1_grad_gate"].to(outputs["x1_fine_logits"].device, outputs["x1_fine_logits"].dtype)
        logits = outputs["x1_fine_logits"]
        x1_outputs["x1_fine_logits"] = logits.detach() + grad_gate * (logits - logits.detach())
        x1_result = self.x1_loss(
            x1_outputs,
            targets,
            indices,
            external_class_weights=controls["auto_x1_class_weight"].to(logits.device, logits.dtype),
        )
        raw_x1 = x1_result["loss_x1_fine"]
        x1_gate = controls["auto_x1_loss_gate"].to(raw_x1.device, raw_x1.dtype)
        losses["loss_x1_fine"] = raw_x1 * x1_gate

        device = outputs["pred_logits"].device
        self.last_x1_diagnostics = {
            **{key: value.detach() for key, value in x1_result.items() if key != "loss_x1_fine"},
            "auto_x1_raw_loss": raw_x1.detach(),
            "auto_x1_effective_loss": losses["loss_x1_fine"].detach(),
            "auto_assignment_agreement": torch.tensor(float(getattr(self.matcher, "last_assignment_agreement", 1.0)), device=device),
            "auto_assignment_iou_delta": torch.tensor(float(getattr(self.matcher, "last_iou_delta", 0.0)), device=device),
            "auto_assignment_conflict": torch.tensor(
                float(getattr(self.matcher, "last_assignment_switch", 0.0))
                * max(-float(getattr(self.matcher, "last_iou_delta", 0.0)), 0.0),
                device=device,
            ),
            **{key: value.detach() for key, value in controls.items()},
            **q_stats,
            **rank_stats,
            **private_x1_stats,
            **preserve_stats,
            **self_anchor_stats,
        }
        if self.training and self.log_interval > 0:
            update = int(self.coordinator.updates.item())
            if update % self.log_interval == 0:
                def s(key):
                    return float(self.last_x1_diagnostics[key].detach().float().mean().item())

                print(
                    "[BudgetHarmonizationCriterion] "
                    f"update={update} raw_loss_x1_fine={s('auto_x1_raw_loss'):.6f} "
                    f"effective_loss_x1_fine={s('auto_x1_effective_loss'):.6f} "
                    f"x1_gate={s('auto_x1_loss_gate'):.5f} grad_gate={s('auto_x1_grad_gate'):.4f} "
                    f"x3_class_gate={s('auto_x3_class_gate'):.4f} x3_loc_gate={s('auto_x3_loc_gate'):.4f} "
                    f"couple={s('auto_couple_gate'):.4f} mu={s('auto_matcher_mu'):.4f} "
                    f"H={s('auto_H'):.4f} R={s('auto_R'):.4f} Rproto={s('auto_R_proto'):.4f} "
                    f"B={s('auto_budget'):.4f} beta={s('auto_beta'):.4f} "
                    f"x1_budget={s('auto_budget_x1'):.4f} x3_budget={s('auto_budget_x3'):.4f} "
                    f"joint=[{float(self.last_x1_diagnostics['auto_joint_weight'].min()):.3f},"
                    f"{s('auto_joint_weight'):.3f},"
                    f"{float(self.last_x1_diagnostics['auto_joint_weight'].max()):.3f}] "
                    f"agreement={s('auto_assignment_agreement'):.3f} "
                    f"iou_delta={s('auto_assignment_iou_delta'):.4f} "
                    f"conflict={s('auto_assignment_conflict'):.4f}",
                    flush=True,
                )
                self._write_diagnostics(update, {
                    "update": update,
                    "raw_loss_x1_fine": s("auto_x1_raw_loss"),
                    "effective_loss_x1_fine": s("auto_x1_effective_loss"),
                    "x1_gate": s("auto_x1_loss_gate"),
                    "grad_gate": s("auto_x1_grad_gate"),
                    "x3_class_gate": s("auto_x3_class_gate"),
                    "x3_loc_gate": s("auto_x3_loc_gate"),
                    "couple_gate": s("auto_couple_gate"),
                    "matcher_mu": s("auto_matcher_mu"),
                    "H": self.last_x1_diagnostics["auto_H"].detach().cpu().tolist(),
                    "R": self.last_x1_diagnostics["auto_R"].detach().cpu().tolist(),
                    "R_proto": self.last_x1_diagnostics["auto_R_proto"].detach().cpu().tolist(),
                    "budget": self.last_x1_diagnostics["auto_budget"].detach().cpu().tolist(),
                    "beta": self.last_x1_diagnostics["auto_beta"].detach().cpu().tolist(),
                    "x1_weight": self.last_x1_diagnostics["auto_x1_class_weight"].detach().cpu().tolist(),
                    "x3_weight": self.last_x1_diagnostics["auto_x3_class_weight"].detach().cpu().tolist(),
                    "match_weight": self.last_x1_diagnostics["auto_matcher_weight"].detach().cpu().tolist(),
                    "joint_weight": self.last_x1_diagnostics["auto_joint_weight"].detach().cpu().tolist(),
                    "agreement": s("auto_assignment_agreement"),
                    "iou_delta": s("auto_assignment_iou_delta"),
                    "conflict": s("auto_assignment_conflict"),
                })
        return losses


