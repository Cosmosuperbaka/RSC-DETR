# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Clean, dataset-agnostic PFHM fine-head module.

PFHM performs a mandatory PFHM-FH annotation diagnosis before constructing its
all-class prototype head. PFHMLoss consumes Hungarian matches and refines the
static frequency prior with training-only difficulty, margin, and IoU evidence.
Neither component reads dataset/class names, validation metrics, or the legacy
DatasetShiftGuard.
"""


from __future__ import annotations

import math

from dataclasses import asdict

from typing import Dict, Mapping, Optional, Sequence

import torch

import torch.nn as nn

import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_iou

from .pfhm_annotation_statistics import PFHMAnnotationDecision, PFHMAnnotationStats, PFHMAnnotationAnalyzer

def _bounded_mean_one(values: torch.Tensor, floor: float, cap: float) -> torch.Tensor:
    """Bound a positive vector while preserving an arithmetic mean of one."""
    if values.numel() == 0:
        return values
    if not (0.0 < floor <= 1.0 <= cap):
        raise ValueError(f"infeasible class-weight bounds: floor={floor}, cap={cap}")
    values = values.detach().float().clamp(min=1.0e-12)
    low = torch.zeros((), device=values.device)
    high = torch.tensor(max(cap / float(values.min().item()), 1.0), device=values.device)
    target = float(values.numel())
    for _ in range(64):
        scale = (low + high) * 0.5
        total = (values * scale).clamp(min=floor, max=cap).sum()
        if float(total.item()) < target:
            low = scale
        else:
            high = scale
    return (values * ((low + high) * 0.5)).clamp(min=floor, max=cap)


class PFHM(nn.Module):
    """All-class cosine prototype head initialized by mandatory PFHM-FH diagnosis."""

    def __init__(
        self,
        annotation_file: str,
        num_classes: int,
        hidden_dim: int = 256,
        category_ids: Optional[Sequence[int]] = None,
        temperature: float = 8.0,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        if not annotation_file:
            raise ValueError("PFHM requires a training annotation_file for PFHM-FH diagnosis")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        stats = PFHMAnnotationAnalyzer.stats_from_coco(annotation_file)
        decision = PFHMAnnotationAnalyzer.analyze(stats)
        expected_ids = list(stats.category_ids)
        supplied_ids = expected_ids if category_ids is None else [int(value) for value in category_ids]
        if int(num_classes) != stats.num_classes:
            raise ValueError(
                f"PFHM num_classes={num_classes} disagrees with annotation classes={stats.num_classes}"
            )
        if supplied_ids != expected_ids:
            raise ValueError(
                f"PFHM category mapping mismatch: supplied={supplied_ids}, annotations={expected_ids}"
            )

        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.annotation_file = str(annotation_file)
        self.static_stats = stats
        self.static_decision = decision

        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)
        initial_log_temperature = math.log(float(temperature))
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(initial_log_temperature))
        else:
            self.register_buffer("log_temperature", torch.tensor(initial_log_temperature))

        self.register_buffer("category_ids", torch.tensor(supplied_ids, dtype=torch.long))
        self.register_buffer(
            "frequency_prior_weights",
            torch.tensor(decision.frequency_prior_weights, dtype=torch.float32),
        )
        self.register_buffer("fine_grad_scale", torch.tensor(decision.fine_grad_scale))
        print(self.explain())

    def _scale_feature_gradient(self, features: torch.Tensor) -> torch.Tensor:
        scale = self.fine_grad_scale.to(device=features.device, dtype=features.dtype)
        return features.detach() + scale * (features - features.detach())

    def forward(self, matching_query_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if matching_query_features.ndim != 3 or matching_query_features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected query features [B,Q,{self.hidden_dim}], got {tuple(matching_query_features.shape)}"
            )
        features = self._scale_feature_gradient(matching_query_features)
        queries = F.normalize(features, p=2, dim=-1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        temperature = self.log_temperature.exp().clamp(max=100.0)
        return {
            "x1_fine_logits": torch.matmul(queries, prototypes.t()) * temperature,
            "x1_fine_prototypes": self.prototypes,
            "x1_category_ids": self.category_ids,
            "x1_frequency_prior": self.frequency_prior_weights,
        }

    def make_loss(self, total_updates: int, **kwargs) -> "PFHMLoss":
        return PFHMLoss(
            stats=self.static_stats,
            decision=self.static_decision,
            total_updates=total_updates,
            **kwargs,
        )

    def profile(self) -> Mapping[str, object]:
        return {"stats": asdict(self.static_stats), "decision": asdict(self.static_decision)}

    def explain(self) -> str:
        d = self.static_decision
        return (
            f"[PFHM][PFHM-FH] classes={self.num_classes} instances={self.static_stats.num_instances} "
            f"imbalance={d.imbalance_risk:.3f} small={d.small_object_risk:.3f} "
            f"density={d.density_risk:.3f} evidence={d.evidence_reliability:.3f} "
            f"loss={d.fine_loss_scale:.3f} grad={d.fine_grad_scale:.3f} "
            f"iou_q={d.fine_iou_quantile:.3f} coverage={d.min_fine_sample_coverage:.3f} "
            f"ap50_release={d.ap50_coverage_release:.3f}"
        )


class PFHMLoss(nn.Module):
    """Training-only PFHM controller and matched-query fine loss."""

    def __init__(
        self,
        stats: PFHMAnnotationStats,
        decision: PFHMAnnotationDecision,
        total_updates: int,
        ema_momentum: float = 0.985,
        ce_cap: float = 3.0,
        small_object_area: float = 1.0e-3,
        quality_alpha: float = 0.75,
        quality_gamma: float = 2.0,
        class_axis_normalization_power: float = 0.0,
        quality_target_floor: float = 0.0,
        final_loss_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1)")
        if not (0.0 <= quality_alpha <= 1.0) or quality_gamma < 0.0:
            raise ValueError("quality_alpha must be in [0, 1] and quality_gamma non-negative")
        if not (0.0 <= class_axis_normalization_power <= 1.0):
            raise ValueError("class_axis_normalization_power must be in [0, 1]")
        if not (0.0 <= quality_target_floor <= 1.0):
            raise ValueError("quality_target_floor must be in [0, 1]")
        if final_loss_scale < 0.0:
            raise ValueError("final_loss_scale must be non-negative")
        self.num_classes = stats.num_classes
        self.total_updates = int(total_updates)
        self.ema_momentum = float(ema_momentum)
        self.ce_cap = float(ce_cap)
        self.small_object_area = float(small_object_area)
        self.quality_alpha = float(quality_alpha)
        self.quality_gamma = float(quality_gamma)
        self.class_axis_normalization_power = float(class_axis_normalization_power)
        self.quality_target_floor = float(quality_target_floor)
        self.final_loss_scale = float(final_loss_scale)
        self.static = decision
        self.category_id_to_index = {int(cid): index for index, cid in enumerate(stats.category_ids)}

        self.register_buffer("frequency_prior", torch.tensor(decision.frequency_prior_weights))
        self.register_buffer("difficulty_ema", torch.ones(self.num_classes))
        self.register_buffer("margin_difficulty_ema", torch.ones(self.num_classes))
        self.register_buffer("iou_ema", torch.full((self.num_classes,), 0.5))
        self.register_buffer("seen", torch.zeros(self.num_classes))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

        self.start_updates = max(1, int(round(0.02 * self.total_updates * decision.fine_start_factor)))
        self.ramp_updates = max(1, int(round(0.08 * self.total_updates * decision.fine_ramp_factor)))

    def _map_labels(self, labels: torch.Tensor) -> torch.Tensor:
        mapped = torch.full_like(labels, -1)
        for category_id, index in self.category_id_to_index.items():
            mapped = torch.where(labels == category_id, torch.as_tensor(index, device=labels.device), mapped)
        if bool((mapped < 0).any()):
            unknown = labels[mapped < 0].detach().unique().cpu().tolist()
            raise ValueError(f"PFHM received labels absent from diagnosed category mapping: {unknown}")
        return mapped

    @staticmethod
    def _paired_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predicted.numel() == 0:
            return predicted.new_zeros((0,))
        matrix, _ = box_iou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(target))
        return matrix.diag()

    @torch.no_grad()
    def _update_state(
        self,
        classes: torch.Tensor,
        ce: torch.Tensor,
        margin_difficulty: torch.Tensor,
        iou: torch.Tensor,
    ) -> None:
        momentum = self.ema_momentum
        ce = ce.to(device=self.difficulty_ema.device, dtype=self.difficulty_ema.dtype)
        margin_difficulty = margin_difficulty.to(
            device=self.margin_difficulty_ema.device,
            dtype=self.margin_difficulty_ema.dtype,
        )
        iou = iou.to(device=self.iou_ema.device, dtype=self.iou_ema.dtype)
        classes = classes.to(self.seen.device)
        for index in classes.unique().tolist():
            mask = classes == int(index)
            if not bool(mask.any()):
                continue
            self.difficulty_ema[index].lerp_(ce[mask].mean(), 1.0 - momentum)
            self.margin_difficulty_ema[index].lerp_(
                margin_difficulty[mask].mean(), 1.0 - momentum
            )
            self.iou_ema[index].lerp_(iou[mask].mean(), 1.0 - momentum)
            self.seen[index].add_(mask.sum())
        self.updates.add_(1)

    def _class_weights(self, device: torch.device) -> torch.Tensor:
        evidence = (self.seen.to(device) / 128.0).clamp(0.0, 1.0)
        difficulty = self.difficulty_ema.to(device).clamp(min=1.0e-3)
        margin = self.margin_difficulty_ema.to(device).clamp(min=1.0e-3)
        reliability = self.iou_ema.to(device).clamp(0.05, 1.0)
        difficulty = difficulty / difficulty.mean().clamp(min=1.0e-6)
        margin = margin / margin.mean().clamp(min=1.0e-6)
        dynamic = difficulty.pow(0.45) * margin.pow(0.35) * reliability.pow(0.20)
        raw = self.frequency_prior.to(device) * dynamic
        raw = 1.0 + evidence * (raw - 1.0)
        floor = max(0.25, 1.0 / (2.0 * self.static.class_weight_cap))
        return _bounded_mean_one(raw, floor, self.static.class_weight_cap)

    def _ramp(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        update = self.updates.to(device=device, dtype=dtype)
        return ((update - self.start_updates) / float(self.ramp_updates)).clamp(0.0, 1.0)

    def _ap50_release(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        update = self.updates.to(device=device, dtype=dtype)
        progress = ((update - self.start_updates) / float(self.ramp_updates)).clamp(0.0, 1.0)
        warm = torch.full_like(progress, 0.30)
        return torch.where(progress > 0.0, warm + 0.70 * progress, torch.zeros_like(progress))

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]],
        matched_indices: Sequence[Sequence[torch.Tensor]],
        external_class_weights: Optional[torch.Tensor] = None,
        positive_loss_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["x1_fine_logits"]
        boxes = outputs["pred_boxes"]
        all_logits, all_classes, all_iou, all_area = [], [], [], []
        for batch_index, (query_indices, target_indices) in enumerate(matched_indices):
            if query_indices.numel() == 0:
                continue
            query_indices = query_indices.to(logits.device)
            target_indices = target_indices.to(logits.device)
            target_boxes = targets[batch_index]["boxes"][target_indices].to(logits.device)
            labels = targets[batch_index]["labels"][target_indices].to(logits.device)
            all_logits.append(logits[batch_index, query_indices])
            all_classes.append(self._map_labels(labels))
            all_iou.append(self._paired_iou(boxes[batch_index, query_indices], target_boxes).detach())
            all_area.append((target_boxes[:, 2] * target_boxes[:, 3]).detach())

        if not all_logits:
            zero = logits.sum() * 0.0
            return {"loss_x1_fine": zero, "x1_active_coverage": zero.detach()}

        logits = torch.cat(all_logits)
        classes = torch.cat(all_classes)
        iou = torch.cat(all_iou).to(logits.dtype).clamp(0.0, 1.0)
        area = torch.cat(all_area)
        probabilities = logits.detach().softmax(-1)
        true_probability = probabilities.gather(1, classes[:, None]).squeeze(1)
        other = probabilities.clone()
        other.scatter_(1, classes[:, None], -1.0)
        margin = true_probability - other.max(dim=1).values
        margin_difficulty = (1.0 - margin).clamp(0.0, 2.0)

        # Quality-aware prototype supervision.  A matched query is not treated
        # as a unit-confidence class target: its target confidence equals the
        # current matched IoU.  This is the same classification/localization
        # alignment principle used by the baseline VFL, now applied to PFHM's
        # fine prototype head.  It relies only on the Hungarian match and box
        # geometry, never on dataset identity or validation measurements.
        positive_quality_target = (
            self.quality_target_floor
            + (1.0 - self.quality_target_floor) * iou
        ).to(logits.dtype)
        quality_target = torch.zeros_like(logits)
        quality_target.scatter_(1, classes[:, None], positive_quality_target[:, None])
        pred_score = logits.detach().sigmoid()
        quality_weight = (
            self.quality_alpha * pred_score.pow(self.quality_gamma) * (1.0 - quality_target)
            + quality_target
        )
        quality_loss = F.binary_cross_entropy_with_logits(
            logits,
            quality_target,
            weight=quality_weight,
            reduction="none",
        ).sum(dim=-1)
        # Normalize only by the *observed class axis*.  Power 0 preserves
        # baseline-style summation; power 1 is a class mean.  Intermediate
        # values retain a scale-stable multi-class signal without assigning
        # any dataset-specific or class-specific training weight.
        quality_loss = quality_loss / float(self.num_classes ** self.class_axis_normalization_power)
        # Keep the auxiliary head on the same class-axis convention as the
        # baseline VFL: aggregate the independent class terms per query
        # rather than averaging them away.  The previous mean made the PFHM
        # gradient shrink in proportion to the number of classes, even though
        # the baseline detector and the prototype head describe the same
        # multi-label quality objective.  This is dataset agnostic: the
        # normalization follows the actual configured class axis, not a
        # dataset name or a hand-set per-dataset weight.
        if self.ce_cap > 0:
            quality_loss = quality_loss.clamp(max=self.ce_cap)
        self._update_state(classes, quality_loss.detach(), margin_difficulty, iou)

        release_strength = float(self.static.ap50_coverage_release)
        release = self._ap50_release(logits.device, logits.dtype)

        # Before altering the matched-IoU selection, estimate whether the
        # current batch can safely trade a little localization selectivity for
        # more AP50 recall.  These are training-only, label-aligned signals;
        # they never inspect validation AP or a dataset name.
        pre_ap_proxy = (
            0.65 * true_probability.mean()
            + 0.35 * (margin > 0).float().mean()
        ).clamp(0.0, 1.0)
        pre_ap50_proxy = (
            0.50 * (margin > 0).float().mean()
            + 0.50 * (iou >= 0.50).float().mean()
        ).clamp(0.0, 1.0)
        pre_ap75_proxy = (iou >= 0.75).float().mean().clamp(0.0, 1.0)
        ap_release_guard = ((pre_ap_proxy - 0.45) / 0.25).clamp(0.0, 1.0)
        ap50_release_guard = ((pre_ap50_proxy - 0.55) / 0.25).clamp(0.0, 1.0)
        ap75_release_guard = ((pre_ap75_proxy - 0.20) / 0.25).clamp(0.0, 1.0)
        release_safety = torch.minimum(
            ap_release_guard,
            torch.minimum(ap50_release_guard, ap75_release_guard),
        )

        # AP50 release is therefore reversible: if AP/AP50/AP75 reliability
        # weakens, the effective selection returns to the conservative static
        # policy instead of merely reducing the auxiliary loss afterwards.
        guarded_release = release_strength * release * release_safety
        base_iou_quantile = min(0.65, self.static.fine_iou_quantile + 0.045 * release_strength)
        base_coverage = max(0.58, self.static.min_fine_sample_coverage - 0.055 * release_strength)
        effective_iou_quantile = base_iou_quantile - 0.045 * float(guarded_release.item())
        effective_coverage = base_coverage + 0.055 * float(guarded_release.item())
        requested_q = min(effective_iou_quantile, 1.0 - effective_coverage)
        threshold = torch.quantile(iou.detach().float(), requested_q)
        keep = iou >= threshold.to(iou.dtype)
        coverage = keep.float().mean()

        class_weights = self._class_weights(logits.device).to(logits.dtype)
        sample_weights = class_weights[classes]
        base_sample_weights = sample_weights
        if external_class_weights is not None:
            external_class_weights = external_class_weights.to(
                device=sample_weights.device, dtype=sample_weights.dtype
            )
            sample_weights = sample_weights * external_class_weights[classes]
        quality = 0.35 + 0.65 * iou
        sample_weights = sample_weights * quality
        small = area <= self.small_object_area
        sample_weights = torch.where(
            small,
            sample_weights * self.static.small_object_scale,
            sample_weights,
        )
        base_sample_weights = torch.where(
            small,
            base_sample_weights * self.static.small_object_scale,
            base_sample_weights,
        )
        sample_weights = sample_weights * keep.to(sample_weights.dtype)
        base_sample_weights = base_sample_weights * keep.to(base_sample_weights.dtype)
        effective_sample_weights = sample_weights
        if positive_loss_scale is not None:
            positive_loss_scale = positive_loss_scale.detach().to(
                device=sample_weights.device, dtype=sample_weights.dtype
            )
            if positive_loss_scale.numel() == sample_weights.numel() and torch.isfinite(positive_loss_scale).all():
                effective_sample_weights = sample_weights * positive_loss_scale.clamp(min=0.0)
        raw_weighted = (quality_loss * base_sample_weights).sum() / base_sample_weights.sum().clamp(min=1.0e-6)
        weighted = (quality_loss * effective_sample_weights).sum() / sample_weights.sum().clamp(min=1.0e-6)

        # Training-only safety proxies. They cannot read validation metrics.
        ap_proxy = (0.5 * true_probability.mean() + 0.5 * coverage).clamp(0.0, 1.0)
        ap50_proxy = (0.65 * coverage + 0.35 * (margin > 0).float().mean()).clamp(0.0, 1.0)
        ap75_proxy = (iou >= 0.75).float().mean().clamp(0.0, 1.0)
        online_safety = torch.minimum(ap_proxy, torch.minimum(ap50_proxy, ap75_proxy))
        safety_floor = min(
            self.static.static_ap_safety_prior,
            self.static.static_ap50_safety_prior,
            self.static.static_ap75_safety_prior,
        )
        safety = (0.55 * safety_floor + 0.45 * online_safety).clamp(0.20, 1.0)
        coverage_release_scale = 1.0 - 0.18 * release_strength * release * (1.0 - release_safety)
        scale = (
            self.static.fine_loss_scale
            * self._ramp(logits.device, logits.dtype)
            * safety
            * coverage_release_scale.clamp(0.70, 1.0)
            * self.final_loss_scale
        )
        return {
            "loss_x1_fine": weighted * scale,
            "x1_raw_loss_before_external_class_weight": (raw_weighted * scale).detach(),
            "x1_active_coverage": coverage.detach(),
            "x1_ap_proxy": ap_proxy.detach(),
            "x1_ap50_proxy": ap50_proxy.detach(),
            "x1_ap75_proxy": ap75_proxy.detach(),
            "x1_matched_quality": iou.mean().detach(),
            "x1_positive_quality_target_mean": positive_quality_target.mean().detach(),
            "x1_positive_quality_target_lt_050": (positive_quality_target < 0.5).float().mean().detach(),
            "x1_effective_scale": scale.detach(),
            "x1_internal_class_weight_min": class_weights.min().detach(),
            "x1_internal_class_weight_mean": class_weights.mean().detach(),
            "x1_internal_class_weight_max": class_weights.max().detach(),
        }


