# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Small balance module connecting PFHM fine supervision with ACRHM reliability."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import torch
import torch.nn as nn
from ...core import register
from .pfhm_annotation_statistics import PFHMAnnotationAnalyzer
from .acrhm_annotation_statistics import ACRHMAnnotationAnalyzer

def _clip(value: float, low: float=0.0, high: float=1.0) -> float:
    return max(low, min(high, float(value)))

@dataclass
class HarmonizationBalanceState:
    weight: torch.Tensor
    progress: torch.Tensor
    x3_ready: torch.Tensor
    x1_proto_ready: torch.Tensor
    x1_proto_mean: torch.Tensor
    x1_participation_gate: torch.Tensor
    safety: torch.Tensor
    x3_gate: torch.Tensor
    x3_class_gate: torch.Tensor
    x3_loc_gate: torch.Tensor
    x1_loc_gate: torch.Tensor
    x1_aux_x3_gate: torch.Tensor

@register()
class HarmonizationBalance(nn.Module):
    """Schedule PFHM as a weak auxiliary signal while ACRHM remains the main path.

    The module is intentionally small: it does not change ACRHM internals and does
    not inspect dataset names or validation metrics.  It consumes training-only
    diagnostics exposed by PFHM/ACRHM and returns the current multiplier for the PFHM
    fine loss.
    """

    def __init__(self, total_updates: int, annotation_file: str='', category_ids: Optional[Sequence[int]]=None, max_x1_weight: Optional[float]=None, start_ratio: Optional[float]=None, ramp_ratio: Optional[float]=None, min_x3_progress: Optional[float]=None, min_ap50_proxy: Optional[float]=None, min_ap75_proxy: Optional[float]=None, safety_floor: Optional[float]=None, x3_primary_threshold: Optional[float]=None, x3_gate: Optional[float]=None, x3_class_gate: Optional[float]=None, x3_loc_gate: Optional[float]=None, x1_loc_gate: Optional[float]=None, x1_aux_x3_gate: Optional[float]=None, x1_participation_mode: str='x3_ready', x1_proto_min: float=0.05, x1_proto_ready: float=0.35, x1_proto_seen_norm: float=128.0, balance_variant: str='final_harmonization', print_interval: int=500) -> None:
        if balance_variant != 'final_harmonization':
            raise ValueError('This package contains only the final RSC-DETR harmonization strategy')
        super().__init__()
        if total_updates <= 0:
            raise ValueError('total_updates must be positive')
        self.balance_variant = str(balance_variant or 'final_harmonization')
        inferred = self._infer_controls(annotation_file, category_ids, self.balance_variant)
        max_x1_weight = inferred['max_x1_weight'] if max_x1_weight is None else float(max_x1_weight)
        start_ratio = inferred['start_ratio'] if start_ratio is None else float(start_ratio)
        ramp_ratio = inferred['ramp_ratio'] if ramp_ratio is None else float(ramp_ratio)
        min_x3_progress = inferred['min_x3_progress'] if min_x3_progress is None else float(min_x3_progress)
        min_ap50_proxy = inferred['min_ap50_proxy'] if min_ap50_proxy is None else float(min_ap50_proxy)
        min_ap75_proxy = inferred['min_ap75_proxy'] if min_ap75_proxy is None else float(min_ap75_proxy)
        safety_floor = inferred['safety_floor'] if safety_floor is None else float(safety_floor)
        x3_primary_threshold = inferred['x3_primary_threshold'] if x3_primary_threshold is None else float(x3_primary_threshold)
        if max_x1_weight < 0:
            raise ValueError('max_x1_weight must be non-negative')
        if not 0.0 <= start_ratio < 1.0:
            raise ValueError('start_ratio must be in [0, 1)')
        if not 0.0 < ramp_ratio <= 1.0:
            raise ValueError('ramp_ratio must be in (0, 1]')
        self.total_updates = int(total_updates)
        self.annotation_file = str(annotation_file or '')
        self.auto_profile = inferred
        self.max_x1_weight = float(max_x1_weight)
        self.start_updates = max(1, round(self.total_updates * float(start_ratio)))
        self.ramp_updates = max(1, round(self.total_updates * float(ramp_ratio)))
        self.min_x3_progress = float(min_x3_progress)
        self.min_ap50_proxy = float(min_ap50_proxy)
        self.min_ap75_proxy = float(min_ap75_proxy)
        self.safety_floor = float(safety_floor)
        self.x3_primary_threshold = float(x3_primary_threshold)
        self.x3_primary_score = float(inferred.get('x3_primary_score', 0.0))
        self.x3_static_gate = _clip(inferred.get('x3_static_gate', 0.0) if x3_gate is None else x3_gate)
        self.x3_class_static_gate = _clip(inferred.get('x3_class_gate', self.x3_static_gate) if x3_class_gate is None else x3_class_gate)
        self.x3_loc_static_gate = _clip(inferred.get('x3_loc_gate', self.x3_static_gate) if x3_loc_gate is None else x3_loc_gate)
        self.x1_loc_static_gate = _clip(inferred.get('x1_loc_gate', 1.0) if x1_loc_gate is None else x1_loc_gate)
        self.x1_aux_x3_static_gate = _clip(inferred.get('x1_aux_x3_gate', self.x3_static_gate) if x1_aux_x3_gate is None else x1_aux_x3_gate)
        self.x1_participation_mode = str(x1_participation_mode or 'x3_ready').lower().replace('-', '_')
        if self.x1_participation_mode not in {'x3_ready', 'annotation_prototype'}:
            raise ValueError(f"x1_participation_mode must be 'x3_ready' or 'annotation_prototype', got {x1_participation_mode}")
        self.x1_proto_min = float(x1_proto_min)
        self.x1_proto_ready = float(x1_proto_ready)
        self.x1_proto_seen_norm = float(x1_proto_seen_norm)
        if self.x1_proto_ready <= self.x1_proto_min:
            raise ValueError('x1_proto_ready must be larger than x1_proto_min')
        if self.x1_proto_seen_norm <= 0:
            raise ValueError('x1_proto_seen_norm must be positive')
        self.print_interval = int(print_interval)
        self.register_buffer('updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('runtime_safety_ema', torch.full((), -1.0, dtype=torch.float32))
        self.last_state: HarmonizationBalanceState | None = None
        print(f"[HarmonizationBalance] mode={inferred['mode']} variant={self.balance_variant} max_x1_weight={self.max_x1_weight:.3f} start_updates={self.start_updates} ramp_updates={self.ramp_updates} min_x3_progress={self.min_x3_progress:.3f} x3_primary_score={self.x3_primary_score:.3f} x3_gate={self.x3_static_gate:.3f} x3_class_gate={self.x3_class_static_gate:.3f} x3_loc_gate={self.x3_loc_static_gate:.3f} x1_loc_gate={self.x1_loc_static_gate:.3f} x1_aux_x3_gate={self.x1_aux_x3_static_gate:.3f} x1_participation_mode={self.x1_participation_mode}")

    @staticmethod
    def _infer_controls(annotation_file: str, category_ids: Optional[Sequence[int]], balance_variant: str='final_harmonization') -> Mapping[str, float]:
        if balance_variant != 'final_harmonization':
            raise ValueError('This package contains only the final RSC-DETR harmonization strategy')
        balance_variant = str(balance_variant or 'final_harmonization').lower().replace('-', '_')
        if not annotation_file:
            return {'mode': 'fallback', 'max_x1_weight': 0.22, 'start_ratio': 0.3, 'ramp_ratio': 0.46, 'min_x3_progress': 0.25, 'min_ap50_proxy': 0.43, 'min_ap75_proxy': 0.18, 'safety_floor': 0.25, 'x3_primary_threshold': 0.58, 'x3_primary_score': 0.5, 'x3_static_gate': 0.42, 'x3_class_gate': 0.64, 'x3_loc_gate': 0.14, 'x1_loc_gate': 0.94, 'x1_aux_x3_gate': 0.46}
        x1_stats = PFHMAnnotationAnalyzer.stats_from_coco(annotation_file)
        x3_stats = ACRHMAnnotationAnalyzer.stats_from_coco(annotation_file)
        supplied = None if category_ids is None else [int(value) for value in category_ids]
        if supplied is not None and supplied != x1_stats.category_ids:
            raise ValueError(f'HarmonizationBalance category mapping mismatch: supplied={supplied}, annotations={x1_stats.category_ids}')
        if x1_stats.category_ids != x3_stats.category_ids:
            raise ValueError('PFHM/ACRHM static analyses disagree on category mapping')
        x1 = PFHMAnnotationAnalyzer.analyze(x1_stats)
        x3 = ACRHMAnnotationAnalyzer.analyze(x3_stats)
        x1_safety = min(x1.static_ap_safety_prior, x1.static_ap50_safety_prior, x1.static_ap75_safety_prior)
        x1_aux_need = _clip(0.42 * x1.classification_need + 0.3 * x1.small_object_risk + 0.18 * x1_safety + 0.1 * (1.0 - x3.imbalance_risk))
        x3_pressure = _clip(0.45 * x3.matching_risk + 0.25 * x3.classification_need + 0.2 * x3.imbalance_risk + 0.1 * x3.density_risk)
        overlap_risk = _clip(0.45 * x3_pressure + 0.3 * x3.imbalance_risk + 0.25 * x3.density_risk)
        complementarity = _clip(x1_aux_need * (1.0 - 0.75 * overlap_risk) * (0.55 + 0.45 * x1_safety))
        x3_primary_score = _clip(0.34 * complementarity + 0.28 * (1.0 - x3_pressure) + 0.24 * (1.0 - x3.imbalance_risk) + 0.14 * (1.0 - x3.density_risk))
        x3_useful_floor = _clip(0.12 + 0.18 * x3.evidence_reliability + 0.12 * x1.small_object_risk + 0.1 * x3.classification_need - 0.16 * x3.imbalance_risk - 0.12 * x3.density_risk, 0.16, 0.38)
        x3_static_gate = max(_clip((x3_primary_score - 0.5) / 0.22), x3_useful_floor)
        x3_class_gate = _clip(max(x3_static_gate, 0.1 + 0.26 * x3.evidence_reliability + 0.22 * x3.classification_need + 0.14 * x1.small_object_risk + 0.1 * x3.matching_risk - 0.14 * x3.imbalance_risk - 0.08 * x3.density_risk), 0.14, 0.72)
        x3_loc_gate = _clip(0.07 + 0.36 * x3_primary_score + 0.18 * (1.0 - x3.localization_risk) + 0.12 * x1_safety + 0.08 * x3.evidence_reliability - 0.22 * x3.imbalance_risk - 0.18 * x3.density_risk - 0.1 * overlap_risk, 0.04, 0.56)
        x3_primary_threshold = 0.55
        x1_loc_gate = 1.0
        x1_aux_x3_gate = x3_static_gate
        recall_need = _clip(0.42 * x1.small_object_risk + 0.3 * x3.classification_need + 0.18 * x3.evidence_reliability + 0.1 * (1.0 - overlap_risk))
        loc_risk = _clip(0.42 * x3.localization_risk + 0.28 * overlap_risk + 0.18 * x3.imbalance_risk + 0.12 * x3.density_risk)
        loc_confidence = _clip(0.52 * x3_primary_score + 0.28 * (1.0 - overlap_risk) + 0.2 * (1.0 - x3.imbalance_risk))
        x3_class_gate = _clip(max(x3_class_gate, x3_static_gate + 0.08 * recall_need, 0.16 + 0.3 * x3.evidence_reliability + 0.26 * x3.classification_need + 0.18 * x1.small_object_risk + 0.1 * x3.matching_risk - 0.1 * x3.imbalance_risk - 0.05 * x3.density_risk), 0.18, 0.78)
        x3_loc_gate = _clip(x3_loc_gate + 0.16 * loc_confidence - 0.3 * loc_risk * overlap_risk - 0.1 * x3.density_risk, 0.03, 0.5)
        x3_static_gate = _clip(max(x3_static_gate, x3_useful_floor + 0.05 * recall_need) - 0.08 * loc_risk, 0.14, 0.7)
        dense_conflict = _clip(0.42 * overlap_risk + 0.24 * x3.density_risk + 0.22 * x3.imbalance_risk + 0.12 * loc_risk)
        budget_prior = _clip(0.3 * x3.classification_need + 0.24 * x1.small_object_risk + 0.2 * x3.evidence_reliability + 0.16 * x3.density_risk + 0.1 * x3.imbalance_risk - 0.16 * loc_risk * overlap_risk)
        native_prior = 1.0 - budget_prior
        x3_primary_threshold = _clip(0.54 + 0.08 * dense_conflict + 0.05 * loc_risk - 0.05 * budget_prior, 0.52, 0.64)
        x3_static_gate = _clip(0.34 + 0.2 * budget_prior + 0.08 * x3.evidence_reliability - 0.1 * dense_conflict, 0.28, 0.56)
        x3_class_gate = _clip(0.5 + 0.22 * budget_prior + 0.12 * x3.classification_need - 0.08 * dense_conflict, 0.5, 0.72)
        x3_loc_gate = _clip(0.11 + 0.12 * budget_prior + 0.08 * (1.0 - loc_risk) - 0.16 * dense_conflict - 0.12 * loc_risk * overlap_risk, 0.06, 0.24)
        x1_loc_gate = _clip(0.86 + 0.08 * x1_safety + 0.06 * loc_risk - 0.06 * budget_prior, 0.82, 1.0)
        x1_aux_x3_gate = _clip(0.4 + 0.18 * budget_prior + 0.08 * x3_primary_score - 0.12 * dense_conflict, 0.38, 0.58)
        max_x1_weight = _clip(0.08 + 0.42 * x1_aux_need * (1.0 - 1.05 * x3_pressure) + 0.08 * x1.evidence_reliability * x1_safety - 0.1 * x3.imbalance_risk, 0.12, 0.34)
        max_x1_weight = _clip(max_x1_weight + 0.05 * x1_aux_need - 0.08 * overlap_risk - 0.04 * x3.imbalance_risk, 0.1, 0.32)
        dense_conflict = _clip(0.42 * overlap_risk + 0.24 * x3.density_risk + 0.22 * x3.imbalance_risk + 0.12 * loc_risk)
        budget_prior = _clip(0.3 * x3.classification_need + 0.24 * x1.small_object_risk + 0.2 * x3.evidence_reliability + 0.16 * x3.density_risk + 0.1 * x3.imbalance_risk - 0.16 * loc_risk * overlap_risk)
        max_x1_weight = _clip(0.2 + 0.07 * budget_prior + 0.04 * x1_safety - 0.07 * dense_conflict, 0.18, 0.27)
        start_ratio = _clip(0.16 + 0.16 * x3_pressure + 0.1 * (1.0 - x1_safety) + 0.06 * x3.imbalance_risk, 0.16, 0.34)
        start_ratio = _clip(start_ratio + 0.04 * overlap_risk + 0.03 * x3.density_risk + 0.02 * x3.imbalance_risk, 0.26, 0.38)
        ramp_ratio = _clip(0.24 + 0.2 * x3_pressure + 0.1 * (1.0 - x1_safety) + 0.06 * x1.small_object_risk, 0.26, 0.46)
        ramp_ratio = _clip(ramp_ratio + 0.04 * overlap_risk + 0.03 * x3.density_risk, 0.38, 0.54)
        min_x3_progress = _clip(0.18 + 0.22 * x3_pressure, 0.18, 0.38)
        min_ap50_proxy = _clip(0.4 + 0.12 * (1.0 - x1_safety) + 0.05 * x3_pressure, 0.4, 0.58)
        min_ap75_proxy = _clip(0.14 + 0.12 * x1.small_object_risk + 0.05 * x3.localization_risk, 0.14, 0.32)
        safety_floor = _clip(0.18 + 0.16 * x1_safety, 0.18, 0.32)
        min_ap50_proxy = _clip(min_ap50_proxy - 0.02 * x3_class_gate, 0.38, 0.54)
        min_ap75_proxy = _clip(min_ap75_proxy - 0.02 * x1_loc_gate, 0.18, 0.3)
        safety_floor = _clip(safety_floor + 0.03 * overlap_risk + 0.02 * x3.density_risk, 0.22, 0.34)
        return {'mode': 'auto', 'max_x1_weight': max_x1_weight, 'start_ratio': start_ratio, 'ramp_ratio': ramp_ratio, 'min_x3_progress': min_x3_progress, 'min_ap50_proxy': min_ap50_proxy, 'min_ap75_proxy': min_ap75_proxy, 'safety_floor': safety_floor, 'x3_primary_threshold': x3_primary_threshold, 'x3_primary_score': x3_primary_score, 'x3_static_gate': x3_static_gate, 'x3_class_gate': x3_class_gate, 'x3_loc_gate': x3_loc_gate, 'x1_loc_gate': x1_loc_gate, 'x1_aux_x3_gate': x1_aux_x3_gate, 'x3_useful_floor': x3_useful_floor, 'x1_aux_need': x1_aux_need, 'x3_pressure': x3_pressure, 'overlap_risk': overlap_risk, 'complementarity': complementarity, 'x1_safety': x1_safety}

    def use_x3_primary(self) -> bool:
        return self.x3_primary_score >= self.x3_primary_threshold

    def use_x3_aux(self) -> bool:
        return self.x1_aux_x3_static_gate >= 0.5

    def x3_loss_gate(self, device: torch.device, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.x3_static_gate, device=device, dtype=dtype)

    def x3_class_gate(self, device: torch.device, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.x3_class_static_gate, device=device, dtype=dtype)

    def x3_loc_gate(self, device: torch.device, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.x3_loc_static_gate, device=device, dtype=dtype)

    def x1_loc_gate(self, device: torch.device, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.x1_loc_static_gate, device=device, dtype=dtype)

    def x1_aux_x3_gate(self, device: torch.device, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(self.x1_aux_x3_static_gate, device=device, dtype=dtype)

    def _as_tensor(self, value, device: torch.device, default: float) -> torch.Tensor:
        if value is None:
            return torch.tensor(default, device=device)
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=torch.float32)
        return torch.tensor(float(value), device=device)

    def _progress(self, device: torch.device) -> torch.Tensor:
        step = self.updates.to(device=device, dtype=torch.float32)
        return ((step - self.start_updates) / float(self.ramp_updates)).clamp(0.0, 1.0)

    def _annotation_prototype_readiness(self, x1_loss_module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if x1_loss_module is None or not all((hasattr(x1_loss_module, name) for name in ('seen', 'iou_ema', 'margin_difficulty_ema'))):
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return (zero, zero)
        seen = (x1_loss_module.seen.detach().to(device=device, dtype=torch.float32) / self.x1_proto_seen_norm).clamp(0.0, 1.0)
        iou = x1_loss_module.iou_ema.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        margin = (1.0 - x1_loss_module.margin_difficulty_ema.detach().to(device=device, dtype=torch.float32) / 2.0).clamp(0.0, 1.0)
        proto = (seen * iou * margin).clamp(0.0, 1.0)
        proto_mean = proto.mean().clamp(0.0, 1.0) if proto.numel() else torch.zeros((), device=device, dtype=torch.float32)
        ready = ((proto_mean - self.x1_proto_min) / max(self.x1_proto_ready - self.x1_proto_min, 1e-06)).clamp(0.0, 1.0)
        return (ready, proto_mean)

    def forward(self, x1_diagnostics: Mapping[str, torch.Tensor], x3_module, device: torch.device, x1_loss_module=None) -> torch.Tensor:
        self.updates.add_(1)
        progress = self._progress(device)
        if x3_module is not None and hasattr(x3_module, '_schedule'):
            x3_ready = x3_module._schedule(device).detach().float().clamp(0.0, 1.0)
        else:
            x3_ready = torch.ones((), device=device)
        x3_gate = (x3_ready / max(self.min_x3_progress, 1e-06)).clamp(0.0, 1.0)
        x1_proto_ready, x1_proto_mean = self._annotation_prototype_readiness(x1_loss_module, device)
        if self.x1_participation_mode == 'annotation_prototype':
            participation_gate = torch.maximum(x3_gate, x1_proto_ready)
        else:
            participation_gate = x3_gate
        ap50 = self._as_tensor(x1_diagnostics.get('x1_ap50_proxy'), device, 1.0).clamp(0.0, 1.0)
        ap75 = self._as_tensor(x1_diagnostics.get('x1_ap75_proxy'), device, 1.0).clamp(0.0, 1.0)
        quality = self._as_tensor(x1_diagnostics.get('x1_matched_quality'), device, 0.5).clamp(0.0, 1.0)
        ap50_gate = ((ap50 - self.min_ap50_proxy) / max(1.0 - self.min_ap50_proxy, 1e-06)).clamp(0.0, 1.0)
        ap75_gate = ((ap75 - self.min_ap75_proxy) / max(1.0 - self.min_ap75_proxy, 1e-06)).clamp(0.0, 1.0)
        quality_gate = ((quality - 0.35) / 0.45).clamp(0.0, 1.0)
        safety = torch.maximum(torch.tensor(self.safety_floor, device=device), torch.minimum(ap50_gate, torch.minimum(ap75_gate, quality_gate)))
        previous = self.runtime_safety_ema.to(device=device, dtype=torch.float32)
        initialized = previous.ge(0.0)
        ema = torch.where(initialized, previous * 0.97 + safety.detach() * 0.03, safety.detach())
        self.runtime_safety_ema.copy_(ema.detach().to(self.runtime_safety_ema))
        safety = torch.maximum(safety, ema * 0.82)
        x3_loss_gate = self.x3_loss_gate(device, dtype=torch.float32)
        x3_class_gate = self.x3_class_gate(device, dtype=torch.float32)
        x3_loc_gate = self.x3_loc_gate(device, dtype=torch.float32)
        x1_loc_gate = self.x1_loc_gate(device, dtype=torch.float32)
        x1_aux_x3_gate = self.x1_aux_x3_gate(device, dtype=torch.float32)
        weight = self.max_x1_weight * progress * participation_gate * safety
        self.last_state = HarmonizationBalanceState(weight=weight.detach(), progress=progress.detach(), x3_ready=x3_ready.detach(), x1_proto_ready=x1_proto_ready.detach(), x1_proto_mean=x1_proto_mean.detach(), x1_participation_gate=participation_gate.detach(), safety=safety.detach(), x3_gate=x3_loss_gate.detach(), x3_class_gate=x3_class_gate.detach(), x3_loc_gate=x3_loc_gate.detach(), x1_loc_gate=x1_loc_gate.detach(), x1_aux_x3_gate=x1_aux_x3_gate.detach())
        if self.training and self.print_interval > 0 and (int(self.updates.item()) % self.print_interval == 0):
            print(f'[HarmonizationBalance] update={int(self.updates.item())} weight={float(weight.item()):.4f} progress={float(progress.item()):.3f} x3={float(x3_ready.item()):.3f} x1_proto={float(x1_proto_ready.item()):.3f} participation={float(participation_gate.item()):.3f} safety={float(safety.item()):.3f} x3_gate={float(x3_loss_gate.item()):.3f} x3_class_gate={float(x3_class_gate.item()):.3f} x3_loc_gate={float(x3_loc_gate.item()):.3f} x1_loc_gate={float(x1_loc_gate.item()):.3f} x1_aux_x3_gate={float(x1_aux_x3_gate.item()):.3f}', flush=True)
        return weight
