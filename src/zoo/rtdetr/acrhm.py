# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Clean, plug-in ACRHM class-matching reliability controller."""


from __future__ import annotations

from dataclasses import asdict

from typing import Dict, Mapping, Optional, Sequence

import torch

import torch.nn as nn

import torch.distributed as dist

from .acrhm_annotation_statistics import ACRHMOnlineAnalyzer, ACRHMAnnotationAnalyzer

def _bounded_mean_one(values: torch.Tensor, floor: float, cap: float) -> torch.Tensor:
    if values.numel() == 0:
        return values
    values = values.detach().float().clamp(min=1.0e-12)
    low = torch.zeros((), device=values.device)
    high = torch.tensor(max(cap / float(values.min().item()), 1.0), device=values.device)
    for _ in range(64):
        scale = (low + high) * 0.5
        total = (values * scale).clamp(min=floor, max=cap).sum()
        if float(total.item()) < values.numel():
            low = scale
        else:
            high = scale
    return (values * ((low + high) * 0.5)).clamp(min=floor, max=cap)


class ACRHM(nn.Module):
    """ACRHM-FH diagnosis plus online class-reliability state.

    The module exposes multipliers for Hungarian class cost, matched positive
    classification loss, and an optional very weak localization role. It never
    changes bbox/GIoU matching costs directly.
    """

    def __init__(
        self,
        annotation_file: str,
        category_ids: Optional[Sequence[int]] = None,
        total_updates: Optional[int] = None,
        ema_momentum: float = 0.93,
        role_strength_scale: float = 1.0,
        delayed_start_ratio: Optional[float] = None,
        delayed_ramp_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not annotation_file:
            raise ValueError("ACRHM requires annotation_file for ACRHM-FH diagnosis")
        if total_updates is not None and total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1)")
        if role_strength_scale < 0.0:
            raise ValueError("role_strength_scale must be non-negative")
        stats = ACRHMAnnotationAnalyzer.stats_from_coco(annotation_file)
        decision = ACRHMAnnotationAnalyzer.analyze(stats)
        supplied = stats.category_ids if category_ids is None else [int(v) for v in category_ids]
        if supplied != stats.category_ids:
            raise ValueError(
                f"ACRHM category mapping mismatch: supplied={supplied}, "
                f"annotations={stats.category_ids}"
            )

        self.stats = stats
        self.decision = decision
        self.total_updates = int(total_updates) if total_updates is not None else 1
        self.total_updates_is_auto = total_updates is None
        self.ema_momentum = float(ema_momentum)
        self.role_strength_scale = float(role_strength_scale)
        self.delayed_start_ratio = None if delayed_start_ratio is None else float(delayed_start_ratio)
        self.delayed_ramp_ratio = None if delayed_ramp_ratio is None else float(delayed_ramp_ratio)
        self.category_id_to_index = {category_id: i for i, category_id in enumerate(supplied)}
        start_ratio = decision.warmup_ratio if self.delayed_start_ratio is None else self.delayed_start_ratio
        ramp_ratio = decision.ramp_ratio if self.delayed_ramp_ratio is None else self.delayed_ramp_ratio
        self.warmup_updates = max(1, round(self.total_updates * start_ratio))
        self.ramp_updates = max(1, round(self.total_updates * ramp_ratio))

        classes = stats.num_classes
        self.register_buffer("category_ids", torch.tensor(supplied, dtype=torch.long))
        self.register_buffer(
            "frequency_prior", torch.tensor(decision.frequency_prior_weights, dtype=torch.float32)
        )
        self.register_buffer("frequency_ema", torch.ones(classes))
        self.register_buffer("difficulty_ema", torch.ones(classes))
        self.register_buffer("iou_ema", torch.full((classes,), 0.5))
        self.register_buffer("seen", torch.zeros(classes))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_dynamic_weights", torch.ones(classes))
        self.fh_online = ACRHMOnlineAnalyzer(classes)
        print(self.explain())

    def set_total_updates(self, total_updates: int) -> None:
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        self.total_updates = int(total_updates)
        start_ratio = self.decision.warmup_ratio if self.delayed_start_ratio is None else self.delayed_start_ratio
        ramp_ratio = self.decision.ramp_ratio if self.delayed_ramp_ratio is None else self.delayed_ramp_ratio
        self.warmup_updates = max(1, round(self.total_updates * start_ratio))
        self.ramp_updates = max(1, round(self.total_updates * ramp_ratio))
        self.total_updates_is_auto = False
        print(
            f"[ACRHM] total_updates={self.total_updates} "
            f"warmup_updates={self.warmup_updates} ramp_updates={self.ramp_updates}"
        )

    def map_labels(self, labels: torch.Tensor) -> torch.Tensor:
        mapped = torch.full_like(labels, -1)
        for category_id, index in self.category_id_to_index.items():
            mapped = torch.where(
                labels == category_id,
                torch.as_tensor(index, device=labels.device),
                mapped,
            )
        if bool((mapped < 0).any()):
            unknown = labels[mapped < 0].detach().unique().cpu().tolist()
            raise ValueError(f"ACRHM received unknown category ids: {unknown}")
        return mapped

    @torch.no_grad()
    def update(
        self,
        labels: torch.Tensor,
        classification_loss: torch.Tensor,
        matched_iou: torch.Tensor,
        wrong_classes: Optional[torch.Tensor] = None,
        probability_margin: Optional[torch.Tensor] = None,
        class_cost_gap: Optional[torch.Tensor] = None,
        full_cost_gap: Optional[torch.Tensor] = None,
        assignment_switch: Optional[torch.Tensor] = None,
    ) -> None:
        if labels.numel() == 0:
            return
        classes = self.map_labels(labels).to(self.seen.device)
        loss = classification_loss.detach().to(self.difficulty_ema)
        iou = matched_iou.detach().to(self.iou_ema).clamp(0.0, 1.0)
        counts = torch.bincount(classes, minlength=self.stats.num_classes).float()
        loss_sum = torch.zeros_like(counts).scatter_add_(0, classes, loss)
        iou_sum = torch.zeros_like(counts).scatter_add_(0, classes, iou)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(loss_sum)
            dist.all_reduce(iou_sum)
        normalized_counts = counts / counts.sum().clamp(min=1.0) * self.stats.num_classes
        momentum = self.ema_momentum
        for index in torch.nonzero(counts > 0, as_tuple=False).flatten().tolist():
            self.frequency_ema[index].lerp_(normalized_counts[index], 1.0 - momentum)
            self.difficulty_ema[index].lerp_(loss_sum[index] / counts[index], 1.0 - momentum)
            self.iou_ema[index].lerp_(iou_sum[index] / counts[index], 1.0 - momentum)
            self.seen[index].add_(counts[index])
        if wrong_classes is not None:
            self.fh_online.update(
                classes,
                wrong_classes,
                probability_margin,
                class_cost_gap,
                full_cost_gap,
                assignment_switch,
            )
        self.updates.add_(1)
        target = self._compute_dynamic_target(self.last_dynamic_weights.device)
        previous = self.last_dynamic_weights
        limit = self.decision.prior_deviation_limit
        updated = previous + (target - previous).clamp(min=-limit, max=limit)
        updated = _bounded_mean_one(
            updated,
            1.0 / (2.0 * self.decision.max_class_weight),
            self.decision.max_class_weight,
        )
        self.last_dynamic_weights.copy_(updated)

    def _schedule(self, device: torch.device) -> torch.Tensor:
        update = self.updates.to(device=device, dtype=torch.float32)
        return ((update - self.warmup_updates) / float(self.ramp_updates)).clamp(0.0, 1.0)

    def _compute_dynamic_target(self, device: torch.device) -> torch.Tensor:
        evidence = (self.seen.to(device) / 128.0).clamp(0.0, 1.0)
        frequency = self.frequency_ema.to(device).clamp(min=1.0e-3)
        difficulty = self.difficulty_ema.to(device).clamp(min=1.0e-3)
        iou = self.iou_ema.to(device).clamp(0.05, 1.0)
        frequency_factor = (frequency.mean() / frequency).pow(self.decision.frequency_power)
        difficulty_factor = (difficulty / difficulty.mean().clamp(min=1.0e-6)).pow(
            self.decision.difficulty_power
        )
        structural = self.fh_online.decision_factors()["structural_need"].to(device)
        structural_factor = structural / structural.mean().clamp(min=1.0e-6)
        # Low-IoU classes are not allowed to receive unrestricted class pressure.
        reliability = (iou / iou.mean().clamp(min=1.0e-6)).clamp(0.5, 1.5).pow(
            self.decision.iou_reliability_power
        )
        # ``frequency_power`` is already inferred by ACRHM-FH.  Applying another
        # fixed exponent here would silently weaken the diagnosed response
        # (for example 0.60 -> 0.33), so consume the inferred factor directly.
        online = frequency_factor * difficulty_factor * structural_factor * reliability
        prior = self.frequency_prior.to(device)
        # Frequency prior is only a cold-start hint. Once a class has enough
        # online evidence, confusion and assignment stability fully replace it.
        prior_blend = self.decision.prior_anchor_strength * (1.0 - evidence)
        anchored = prior_blend * prior + (1.0 - prior_blend) * online
        target = 1.0 + evidence * (anchored - 1.0)
        target = _bounded_mean_one(
            target,
            1.0 / (2.0 * self.decision.max_class_weight),
            self.decision.max_class_weight,
        )
        return target

    def dynamic_weights(self, device: Optional[torch.device] = None) -> torch.Tensor:
        device = self.last_dynamic_weights.device if device is None else device
        return self.last_dynamic_weights.to(device)

    def _role_weights(self, strength: float, device: Optional[torch.device]) -> torch.Tensor:
        weights = self.dynamic_weights(device)
        # preserve_easy: ACRHM only adds pressure to reliable hard classes.
        boost = (weights - 1.0).clamp(min=0.0)
        return 1.0 + self._schedule(weights.device) * float(strength) * self.role_strength_scale * boost

    def matcher_weights(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self._role_weights(self.decision.matcher_strength, device)

    def classification_weights(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self._role_weights(self.decision.classification_strength, device)

    def localization_weights(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return self._role_weights(self.decision.localization_strength, device)

    def profile(self) -> Mapping[str, object]:
        return {"stats": asdict(self.stats), "decision": asdict(self.decision)}

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        device = self.frequency_prior.device
        return {
            "updates": self.updates.detach().clone(),
            "frequency": self.frequency_ema.detach().clone(),
            "difficulty": self.difficulty_ema.detach().clone(),
            "iou": self.iou_ema.detach().clone(),
            "dynamic_weights": self.dynamic_weights(device).detach().clone(),
            "matcher_weights": self.matcher_weights(device).detach().clone(),
            "classification_weights": self.classification_weights(device).detach().clone(),
            "confusion_matrix": self.fh_online.confusion_matrix.detach().clone(),
            **{
                f"fh_{key}": value.detach().clone()
                for key, value in self.fh_online.decision_factors().items()
            },
        }

    def explain(self) -> str:
        d = self.decision
        return (
            f"[ACRHM][ACRHM-FH] classes={self.stats.num_classes} instances={self.stats.num_instances} "
            f"imbalance={d.imbalance_risk:.3f} density={d.density_risk:.3f} "
            f"evidence={d.evidence_reliability:.3f} matcher={d.matcher_strength:.3f} "
            f"classification={d.classification_strength:.3f} loc={d.localization_strength:.3f} "
            f"max_weight={d.max_class_weight:.3f}"
        )


