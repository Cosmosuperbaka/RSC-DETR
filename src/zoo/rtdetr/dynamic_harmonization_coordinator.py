# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Dataset-agnostic annotation-budget state-driven PFHM/ACRHM coordinator.

This module intentionally does not load historical profiles, dataset names,
validation metrics, or test metrics.  It reads the current training annotation
once to initialize class priors, then uses detached online state from PFHM and
ACRHM to produce bounded gates and class weights.
"""


from __future__ import annotations

import json

import math

from dataclasses import dataclass

from pathlib import Path

from typing import Mapping, Optional, Sequence

import torch

import torch.nn as nn

from ...core import register

def _clip_float(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean_one(values: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if values.numel() == 0:
        return values
    values = values.detach().float().clamp(min=1.0e-8)
    low_scale = torch.zeros((), device=values.device)
    high_scale = torch.tensor(max(high / float(values.min().item()), 1.0), device=values.device)
    target = float(values.numel())
    for _ in range(64):
        scale = 0.5 * (low_scale + high_scale)
        total = (values * scale).clamp(low, high).sum()
        if float(total.item()) < target:
            low_scale = scale
        else:
            high_scale = scale
    return (values * (0.5 * (low_scale + high_scale))).clamp(low, high)


@dataclass(frozen=True)
class AnnotationState:
    num_classes: int
    num_images: int
    num_instances: int
    category_ids: list[int]
    class_counts: list[int]
    small_ratio: float
    density: float
    tail_risk: float
    evidence_mean: float
    entropy: float
    frequency_need: list[float]
    evidence: list[float]


def annotation_state(
    annotation_file: str,
    category_ids: Optional[Sequence[int]] = None,
    *,
    density_norm: float = 8.0,
    evidence_tau: float = 64.0,
    small_area: float = 32.0 * 32.0,
) -> AnnotationState:
    path = Path(annotation_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = [int(cat["id"]) for cat in data.get("categories", [])]
    if category_ids is not None:
        supplied = [int(value) for value in category_ids]
        if supplied != categories:
            raise ValueError(f"category_ids mismatch: supplied={supplied}, annotation={categories}")
    else:
        supplied = categories
    index = {cid: i for i, cid in enumerate(supplied)}
    counts = [0 for _ in supplied]
    small = 0
    valid = 0
    for ann in data.get("annotations", []):
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        cid = int(ann["category_id"])
        if cid not in index:
            continue
        bbox = ann.get("bbox", [0, 0, 0, 0])
        w, h = float(bbox[2]), float(bbox[3])
        if w <= 0 or h <= 0:
            continue
        area = float(ann.get("area", w * h))
        counts[index[cid]] += 1
        valid += 1
        if area < small_area:
            small += 1
    nonzero = [max(c, 1) for c in counts]
    n_max, n_min = max(nonzero), min(nonzero)
    tail_risk = _clip_float((math.log(n_max + 1.0) - math.log(n_min + 1.0)) / max(math.log(n_max + 1.0), 1.0e-6))
    probs = [c / max(valid, 1) for c in counts]
    entropy = -sum(p * math.log(max(p, 1.0e-12)) for p in probs) / max(math.log(max(len(supplied), 2)), 1.0e-6)
    frequency_need = [
        _clip_float((math.log(n_max + 1.0) - math.log(c + 1.0)) / max(math.log(n_max + 1.0), 1.0e-6))
        for c in nonzero
    ]
    evidence = [1.0 - math.exp(-float(c) / max(evidence_tau, 1.0e-6)) for c in counts]
    return AnnotationState(
        num_classes=len(supplied),
        num_images=len(data.get("images", [])),
        num_instances=valid,
        category_ids=supplied,
        class_counts=counts,
        small_ratio=float(small) / max(valid, 1),
        density=_clip_float(float(valid) / max(len(data.get("images", [])), 1) / density_norm),
        tail_risk=tail_risk,
        evidence_mean=sum(evidence) / max(len(evidence), 1),
        entropy=entropy,
        frequency_need=frequency_need,
        evidence=evidence,
    )


@register()
class DynamicHarmonizationCoordinator(nn.Module):
    """Compute PFHM/ACRHM controls from train annotations and detached online state."""

    def __init__(
        self,
        annotation_file: str,
        category_ids: Optional[Sequence[int]] = None,
        total_updates: int = 1,
        density_norm: float = 8.0,
        evidence_tau: float = 64.0,
        state_ema: float = 0.97,
        b_max: float = 0.60,
        beta_min: float = 0.25,
        beta_max: float = 0.75,
        x1_weight_max: float = 0.30,
        x1_weight_min: float = 0.0,
        x1_proto_min: float = 0.05,
        x1_proto_ready: float = 0.35,
        x1_grad_min: float = 0.10,
        x1_grad_max: float = 0.60,
        cls_gate_base: float = 0.22,
        cls_gate_min: float = 0.05,
        cls_gate_max: float = 0.85,
        cls_lambda_d: float = 0.42,
        cls_lambda_f: float = 0.22,
        cls_lambda_p: float = 0.18,
        loc_gate_min: float = 0.03,
        loc_gate_max: float = 0.34,
        couple_gate_max: float = 0.65,
        matcher_mu_max: float = 0.65,
        class_weight_min: float = 0.70,
        class_weight_max: float = 1.45,
        joint_weight_max: float = 1.20,
        print_interval: int = 500,
    ) -> None:
        super().__init__()
        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if not (0.0 <= state_ema < 1.0):
            raise ValueError("state_ema must be in [0, 1)")
        self.total_updates = int(total_updates)
        self.print_interval = int(print_interval)
        self.state = annotation_state(
            annotation_file,
            category_ids,
            density_norm=float(density_norm),
            evidence_tau=float(evidence_tau),
        )
        self.state_ema = float(state_ema)
        self.b_max = float(b_max)
        self.beta_min, self.beta_max = float(beta_min), float(beta_max)
        self.x1_weight_min, self.x1_weight_max = float(x1_weight_min), float(x1_weight_max)
        self.x1_proto_min, self.x1_proto_ready = float(x1_proto_min), float(x1_proto_ready)
        self.x1_grad_min, self.x1_grad_max = float(x1_grad_min), float(x1_grad_max)
        self.cls_gate_base = float(cls_gate_base)
        self.cls_gate_min, self.cls_gate_max = float(cls_gate_min), float(cls_gate_max)
        self.cls_lambda_d, self.cls_lambda_f, self.cls_lambda_p = float(cls_lambda_d), float(cls_lambda_f), float(cls_lambda_p)
        self.loc_gate_min, self.loc_gate_max = float(loc_gate_min), float(loc_gate_max)
        self.couple_gate_max, self.matcher_mu_max = float(couple_gate_max), float(matcher_mu_max)
        self.class_weight_min, self.class_weight_max = float(class_weight_min), float(class_weight_max)
        self.joint_weight_max = float(joint_weight_max)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("frequency_need", torch.tensor(self.state.frequency_need, dtype=torch.float32))
        self.register_buffer("evidence", torch.tensor(self.state.evidence, dtype=torch.float32))
        self.register_buffer("difficulty_ema", torch.full((self.state.num_classes,), 0.5))
        self.register_buffer("quality_ema", torch.full((self.state.num_classes,), 0.5))
        self.register_buffer("proto_ema", torch.full((self.state.num_classes,), 0.05))
        self.register_buffer("match_stability_ema", torch.full((self.state.num_classes,), 1.0))
        self.register_buffer("confusion_ema", torch.zeros(self.state.num_classes))
        self.last_controls: dict[str, torch.Tensor] = {}
        print(
            "[DynamicHarmonizationCoordinator] annotation-state "
            f"classes={self.state.num_classes} instances={self.state.num_instances} "
            f"T={self.state.tail_risk:.4f} S={self.state.small_ratio:.4f} "
            f"N={self.state.density:.4f} E={self.state.evidence_mean:.4f} "
            f"F={self.state.entropy:.4f}",
            flush=True,
        )

    @staticmethod
    def _norm01(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().float()
        low, high = value.min(), value.max()
        if float((high - low).abs().item()) < 1.0e-6:
            return torch.full_like(value, 0.5)
        return ((value - low) / (high - low).clamp(min=1.0e-6)).clamp(0.0, 1.0)

    def annotation_report(self) -> Mapping[str, object]:
        return {
            "category_ids": self.state.category_ids,
            "class_counts": self.state.class_counts,
            "T_tail_risk": self.state.tail_risk,
            "S_small_ratio": self.state.small_ratio,
            "N_density": self.state.density,
            "E_evidence_mean": self.state.evidence_mean,
            "F_frequency_entropy": self.state.entropy,
            "frequency_need": self.state.frequency_need,
            "evidence": self.state.evidence,
        }

    @torch.no_grad()
    def forward(self, x3_module, x1_loss=None, matcher=None, device: Optional[torch.device] = None) -> Mapping[str, torch.Tensor]:
        if device is None:
            device = self.frequency_need.device
        self.updates.add_(1)
        diag = x3_module.diagnostics() if x3_module is not None else {}
        difficulty = self._norm01(diag.get("difficulty", self.difficulty_ema).to(device=device).float())
        quality = diag.get("iou", self.quality_ema).to(device=device).float().clamp(0.0, 1.0)
        evidence = self.evidence.to(device=device).float().clamp(0.0, 1.0)
        switch = diag.get("fh_assignment_switch", torch.zeros_like(evidence)).to(device=device).float().clamp(0.0, 1.0)
        stability = (1.0 - switch).clamp(0.0, 1.0)
        confusion = diag.get("fh_pair_concentration", torch.zeros_like(evidence)).to(device=device).float().clamp(0.0, 1.0)

        if x1_loss is not None and hasattr(x1_loss, "seen"):
            x1_seen = (x1_loss.seen.to(device=device).float() / 128.0).clamp(0.0, 1.0)
            x1_iou = x1_loss.iou_ema.to(device=device).float().clamp(0.0, 1.0)
            x1_margin = (1.0 - x1_loss.margin_difficulty_ema.to(device=device).float() / 2.0).clamp(0.0, 1.0)
            proto = (x1_seen * x1_iou * x1_margin).clamp(0.0, 1.0)
        else:
            proto = torch.zeros_like(evidence)

        momentum = self.state_ema
        self.difficulty_ema.lerp_(difficulty.to(self.difficulty_ema), 1.0 - momentum)
        self.quality_ema.lerp_(quality.to(self.quality_ema), 1.0 - momentum)
        self.proto_ema.lerp_(proto.to(self.proto_ema), 1.0 - momentum)
        self.match_stability_ema.lerp_(stability.to(self.match_stability_ema), 1.0 - momentum)
        self.confusion_ema.lerp_(confusion.to(self.confusion_ema), 1.0 - momentum)

        F_c = self.frequency_need.to(device=device).float()
        D_c = self.difficulty_ema.to(device=device).float().clamp(0.0, 1.0)
        Q_c = self.quality_ema.to(device=device).float().clamp(0.0, 1.0)
        # Startup must respond to observed prototype evidence without waiting
        # thousands of EMA updates; the EMA still prevents noisy regressions.
        P_c = torch.maximum(self.proto_ema.to(device=device).float(), proto.to(device=device).float()).clamp(0.0, 1.0)
        M_c = self.match_stability_ema.to(device=device).float().clamp(0.0, 1.0)
        C_c = self.confusion_ema.to(device=device).float().clamp(0.0, 1.0)
        E_c = evidence

        density = torch.tensor(self.state.density, device=device, dtype=torch.float32).clamp(0.0, 1.0)
        entropy = torch.tensor(self.state.entropy, device=device, dtype=torch.float32).clamp(0.0, 1.0)
        small_ratio = torch.tensor(self.state.small_ratio, device=device, dtype=torch.float32).clamp(0.0, 1.0)
        evidence_mean = torch.tensor(self.state.evidence_mean, device=device, dtype=torch.float32).clamp(0.0, 1.0)
        density_gap = (1.0 - density).clamp(0.0, 1.0)

        H_c = (0.42 * F_c + 0.38 * D_c + 0.12 * C_c + 0.08 * density_gap).clamp(0.0, 1.0)
        R_c = (Q_c * M_c * E_c).clamp(0.0, 1.0)
        R_proto = (P_c * Q_c * E_c).clamp(0.0, 1.0)
        B_c = (self.b_max * H_c * R_c).clamp(0.0, self.b_max)
        beta = (R_proto / (R_proto + D_c + 1.0e-6)).clamp(self.beta_min, self.beta_max)
        B_x1 = beta * B_c
        B_x3 = (1.0 - beta) * B_c

        x1_class = _mean_one((1.0 + B_x1).clamp(self.class_weight_min, self.class_weight_max), self.class_weight_min, self.class_weight_max)
        x3_class = _mean_one((1.0 + B_x3).clamp(self.class_weight_min, self.class_weight_max), self.class_weight_min, self.class_weight_max)
        joint = (x1_class * x3_class).clamp(1.0 / self.joint_weight_max, self.joint_weight_max)

        p_bar = P_c.mean().clamp(0.0, 1.0)
        q_bar = Q_c.mean().clamp(0.0, 1.0)
        d_bar = D_c.mean().clamp(0.0, 1.0)
        f_bar = F_c.mean().clamp(0.0, 1.0)
        m_bar = M_c.mean().clamp(0.0, 1.0)
        r_bar = R_c.mean().clamp(0.0, 1.0)
        c_match = C_c.mean().clamp(0.0, 1.0)
        proto_ready = ((p_bar - self.x1_proto_min) / max(self.x1_proto_ready - self.x1_proto_min, 1.0e-6)).clamp(0.0, 1.0)
        x1_cap = (self.x1_weight_max * (0.75 - 0.13 * density_gap)).clamp(self.x1_weight_min, self.x1_weight_max)
        x1_gate = (x1_cap * proto_ready).clamp(self.x1_weight_min, self.x1_weight_max)
        grad_gate = (0.12 + 0.07 * density + 0.015 * p_bar * q_bar).clamp(self.x1_grad_min, self.x1_grad_max)
        cls_gate = (
            0.605 + 0.18 * density - 0.06 * entropy + 0.04 * d_bar + 0.03 * evidence_mean
        ).clamp(self.cls_gate_min, self.cls_gate_max)
        loc_static = 0.074 + 0.57 * density_gap * evidence_mean * (0.65 + 0.35 * entropy)
        loc_reliability = (0.70 + 0.30 * q_bar * m_bar * (1.0 - 0.5 * d_bar)).clamp(0.0, 1.0)
        loc_gate = (loc_static * loc_reliability).clamp(self.loc_gate_min, self.loc_gate_max)
        couple_gate = (p_bar * r_bar).clamp(0.0, self.couple_gate_max)
        matcher_static = 0.49 + 0.20 * density_gap + 0.04 * evidence_mean + 0.025 * (entropy - 0.70).clamp(min=0.0)
        matcher_reliability = (0.90 + 0.10 * m_bar * q_bar).clamp(0.0, 1.0)
        mu = (matcher_static * matcher_reliability).clamp(0.0, self.matcher_mu_max)
        match_weights = _mean_one((1.0 + mu * (x3_class - 1.0)).clamp(0.90, 1.10), 0.90, 1.10)

        controls = {
            "auto_H": H_c.detach(),
            "auto_R": R_c.detach(),
            "auto_R_proto": R_proto.detach(),
            "auto_budget": B_c.detach(),
            "auto_beta": beta.detach(),
            "auto_budget_x1": B_x1.detach(),
            "auto_budget_x3": B_x3.detach(),
            "auto_x1_class_weight": x1_class.detach(),
            "auto_x3_class_weight": x3_class.detach(),
            "auto_joint_weight": joint.detach(),
            "auto_x1_loss_gate": x1_gate.detach(),
            "auto_x1_grad_gate": grad_gate.detach(),
            "auto_x3_class_gate": cls_gate.detach(),
            "auto_x3_loc_gate": loc_gate.detach(),
            "auto_couple_gate": couple_gate.detach(),
            "auto_matcher_mu": mu.detach(),
            "auto_matcher_weight": match_weights.detach(),
            "auto_D": D_c.detach(),
            "auto_Q": Q_c.detach(),
            "auto_P": P_c.detach(),
            "auto_M": M_c.detach(),
            "auto_C": C_c.detach(),
        }
        self.last_controls = controls
        if self.training and self.print_interval > 0 and int(self.updates.item()) % self.print_interval == 0:
            print(
                "[DynamicHarmonizationCoordinator] "
                f"update={int(self.updates.item())} "
                f"x1_gate={float(x1_gate):.5f} grad_gate={float(grad_gate):.4f} "
                f"cls_gate={float(cls_gate):.4f} loc_gate={float(loc_gate):.4f} "
                f"couple={float(couple_gate):.4f} mu={float(mu):.4f} "
                f"H={float(H_c.mean()):.4f} R={float(R_c.mean()):.4f} "
                f"P={float(P_c.mean()):.4f} beta={float(beta.mean()):.4f} "
                f"joint=[{float(joint.min()):.3f},{float(joint.mean()):.3f},{float(joint.max()):.3f}]",
                flush=True,
            )
        return controls


