# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Route-only PFHM/ACRHM criterion for dynamic harmonization.

dynamic harmonization confines new behavior to annotation-conditioned route selection.  After the
route is known, the selected successful criterion is instantiated and called as
the implementation object:

* native route: ``NativeHarmonizationCriterion`` (M3FD native-reliability path)
* budget route: ``BudgetHarmonizationCriterion`` (VEDAI annotation-budget path)
"""


from __future__ import annotations

import torch.nn as nn

from ...core import register

from .harmonization_native_criterion import NativeHarmonizationCriterion

from .harmonization_budget_criterion import BudgetHarmonizationCriterion

from .dynamic_harmonization_coordinator import annotation_state

@register()
class DynamicHarmonizationCriterion(nn.Module):
    """Thin route wrapper that delegates to the chosen anchor criterion."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher", "baseline_matcher", "balance", "coordinator"]

    def __init__(
        self,
        weight_dict,
        losses,
        alpha=0.75,
        gamma=2.0,
        num_classes=80,
        annotation_file=None,
        category_ids=None,
        total_updates=1,
        x1_num_classes=None,
        matcher=None,
        baseline_matcher=None,
        balance=None,
        coordinator=None,
        diagnostics_path: str = "",
        class_axis_normalization_power: float = 0.0,
        quality_target_floor: float = 0.0,
        final_loss_scale: float = 1.0,
        log_interval: int = 500,
        density_norm: float = 8.0,
        evidence_tau: float = 64.0,
        route_density_center: float = 0.65,
        route_density_scale: float = 4.0,
        route_class_center: float = 7.0,
        route_class_scale: float = 1.2,
        route_entropy_center: float = 0.76,
        route_entropy_scale: float = 2.0,
        route_evidence_center: float = 0.96,
        route_evidence_scale: float = 2.0,
        native_hard_threshold: float = 0.90,
        budget_hard_threshold: float = 0.10,
    ):
        super().__init__()
        state = annotation_state(
            annotation_file,
            category_ids,
            density_norm=float(density_norm),
            evidence_tau=float(evidence_tau),
        )
        score = (
            float(route_density_scale) * (state.density - float(route_density_center))
            + float(route_class_scale) * (float(route_class_center) - float(state.num_classes))
            - float(route_entropy_scale) * (state.entropy - float(route_entropy_center))
            + float(route_evidence_scale) * (state.evidence_mean - float(route_evidence_center))
        )
        import math

        native_affinity = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-score))))
        low, high = float(budget_hard_threshold), float(native_hard_threshold)
        if high <= low:
            raise ValueError("native_hard_threshold must be larger than budget_hard_threshold")
        native_fraction = max(0.0, min(1.0, (native_affinity - low) / (high - low)))
        use_native = native_fraction >= 1.0
        use_budget = native_fraction <= 0.0
        self.route_report = {
            "native_affinity": native_affinity,
            "native_fraction": native_fraction,
            "route": "native_reliability" if use_native else ("annotation_budget" if use_budget else "continuous_mix"),
            "density": state.density,
            "num_classes": state.num_classes,
            "entropy": state.entropy,
            "evidence_mean": state.evidence_mean,
            "tail_risk": state.tail_risk,
            "small_ratio": state.small_ratio,
        }

        if use_native:
            if baseline_matcher is None or balance is None:
                raise ValueError("DynamicHarmonizationCriterion native path requires native-reliability baseline_matcher and balance")
            self.impl = NativeHarmonizationCriterion(
                weight_dict=weight_dict,
                losses=losses,
                alpha=alpha,
                gamma=gamma,
                num_classes=num_classes,
                matcher=matcher,
                annotation_file=annotation_file,
                category_ids=category_ids,
                total_updates=int(total_updates),
                x1_num_classes=x1_num_classes,
                class_axis_normalization_power=class_axis_normalization_power,
                baseline_matcher=baseline_matcher,
                balance=balance,
            )
            self.route_anchor = "native_reliability"
            print(
                "[DynamicHarmonizationCriterion] route=native_reliability "
                f"native_affinity={native_affinity:.6f} native_fraction=1.000000 "
                "impl=NativeHarmonizationCriterion",
                flush=True,
            )
            return

        if use_budget:
            if coordinator is None:
                raise ValueError("DynamicHarmonizationCriterion annotation-budget path requires the original DynamicHarmonizationCoordinator")
            self.impl = BudgetHarmonizationCriterion(
                weight_dict=weight_dict,
                losses=losses,
                alpha=alpha,
                gamma=gamma,
                num_classes=num_classes,
                matcher=matcher,
                annotation_file=annotation_file,
                category_ids=category_ids,
                total_updates=int(total_updates),
                x1_num_classes=x1_num_classes,
                coordinator=coordinator,
                diagnostics_path=diagnostics_path,
                log_interval=log_interval,
                class_axis_normalization_power=class_axis_normalization_power,
                quality_target_floor=quality_target_floor,
                final_loss_scale=final_loss_scale,
            )
            self.route_anchor = "annotation_budget"
            print(
                "[DynamicHarmonizationCriterion] route=annotation_budget "
                f"native_affinity={native_affinity:.6f} native_fraction=0.000000 "
                "impl=BudgetHarmonizationCriterion",
                flush=True,
            )
            return

        if baseline_matcher is None or balance is None or coordinator is None:
            raise ValueError("DynamicHarmonizationCriterion continuous mix requires native-reliability modules and the original DynamicHarmonizationCoordinator")
        self.native_fraction = float(native_fraction)
        self.native_impl = NativeHarmonizationCriterion(
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            matcher=matcher,
            annotation_file=annotation_file,
            category_ids=category_ids,
            total_updates=int(total_updates),
            x1_num_classes=x1_num_classes,
            class_axis_normalization_power=class_axis_normalization_power,
            baseline_matcher=baseline_matcher,
            balance=balance,
        )
        self.budget_impl = BudgetHarmonizationCriterion(
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            matcher=matcher,
            annotation_file=annotation_file,
            category_ids=category_ids,
            total_updates=int(total_updates),
            x1_num_classes=x1_num_classes,
            coordinator=coordinator,
            diagnostics_path=diagnostics_path,
            log_interval=log_interval,
            class_axis_normalization_power=class_axis_normalization_power,
            quality_target_floor=quality_target_floor,
            final_loss_scale=final_loss_scale,
        )
        self.route_anchor = "continuous_mix"
        print(
            "[DynamicHarmonizationCriterion] route=continuous_mix "
            f"native_affinity={native_affinity:.6f} native_fraction={native_fraction:.6f} "
            "impl=weighted(NativeHarmonizationCriterion,BudgetHarmonizationCriterion)",
            flush=True,
        )
        return

    @property
    def last_x1_diagnostics(self):
        if hasattr(self, "native_impl") and hasattr(self, "budget_impl"):
            return getattr(self.budget_impl, "last_x1_diagnostics", {}) or getattr(self.native_impl, "last_x1_diagnostics", {})
        return getattr(self.impl, "last_x1_diagnostics", {})

    @property
    def balance(self):
        if hasattr(self, "native_impl"):
            return getattr(self.native_impl, "balance", None)
        return getattr(self.impl, "balance", None)

    def forward(self, outputs, targets, **kwargs):
        if hasattr(self, "native_impl") and hasattr(self, "budget_impl"):
            native_losses = self.native_impl(outputs, targets, **kwargs)
            budget_losses = self.budget_impl(outputs, targets, **kwargs)
            alpha = self.native_fraction
            keys = set(native_losses) | set(budget_losses)
            return {
                key: native_losses.get(key, 0.0) * alpha + budget_losses.get(key, 0.0) * (1.0 - alpha)
                for key in keys
            }
        return self.impl(outputs, targets, **kwargs)


