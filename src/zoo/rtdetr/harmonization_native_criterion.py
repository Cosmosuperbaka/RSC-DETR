# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Direct composition of clean PFHM fine supervision and ACRHM matcher weighting."""


from .acrhm_criterion import ACRHMCriterion

from .rtdetrv2_criterion import RTDETRCriterionv2

from .pfhm_annotation_statistics import PFHMAnnotationAnalyzer

from .harmonization_balance import HarmonizationBalance

from .pfhm import PFHMLoss

from ...core import register

@register()
class NativeHarmonizationCriterion(ACRHMCriterion):
    """Autonomous PFHM/ACRHM wrapper without changing PFHM or ACRHM internals.

    The balance module decides whether the main matching path should be ACRHM or
    the baseline matcher.  ACRHM classification and localization weights use
    separate gates, so data that benefits from ACRHM recall can still suppress
    ACRHM localization when high-IoU risk is high.
    """

    __share__ = ["num_classes"]
    __inject__ = ["matcher", "baseline_matcher", "balance"]

    def __init__(
        self,
        *args,
        annotation_file,
        category_ids,
        total_updates,
        x1_num_classes=None,
        class_axis_normalization_power=0.0,
        baseline_matcher=None,
        balance=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if baseline_matcher is None:
            raise ValueError("NativeHarmonizationCriterion requires baseline_matcher")
        stats = PFHMAnnotationAnalyzer.stats_from_coco(annotation_file)
        decision = PFHMAnnotationAnalyzer.analyze(stats)
        supplied_ids = [int(value) for value in category_ids]
        x1_num_classes = len(supplied_ids) if x1_num_classes is None else int(x1_num_classes)
        if x1_num_classes != stats.num_classes:
            raise ValueError(
                f"NativeHarmonizationCriterion x1_num_classes={x1_num_classes} disagrees with "
                f"annotation classes={stats.num_classes}"
            )
        if supplied_ids != stats.category_ids:
            raise ValueError(
                f"NativeHarmonizationCriterion category mapping mismatch: supplied={supplied_ids}, "
                f"annotations={stats.category_ids}"
            )
        self.x1_loss = PFHMLoss(
            stats,
            decision,
            total_updates=int(total_updates),
            quality_alpha=self.alpha,
            quality_gamma=self.gamma,
            class_axis_normalization_power=class_axis_normalization_power,
        )
        self.x3_matcher = self.matcher
        self.baseline_matcher = baseline_matcher
        self.balance = balance or HarmonizationBalance(total_updates=int(total_updates))
        self.last_x1_diagnostics = {}
        self._x3_class_gate = None
        self._x3_loc_gate = None
        print(
            "[NativeHarmonizationCriterion] autonomous PFHM/ACRHM connection; "
            f"primary={'x3' if self.balance.use_x3_primary() else 'baseline'} "
            f"x1_aux_matcher={'x3' if self.balance.use_x3_aux() else 'primary'} "
            f"x3_gate={self.balance.x3_static_gate:.3f} "
            f"x3_class_gate={self.balance.x3_class_static_gate:.3f} "
            f"x3_loc_gate={self.balance.x3_loc_static_gate:.3f} "
            f"x1_loc_gate={self.balance.x1_loc_static_gate:.3f} "
            f"x1_aux_x3_gate={self.balance.x1_aux_x3_static_gate:.3f} "
            f"total_updates={total_updates}"
        )

    def _active_matcher(self):
        return self.x3_matcher if self.balance.use_x3_primary() else self.baseline_matcher

    def _x1_aux_matcher(self):
        return self.x3_matcher if self.balance.use_x3_aux() else self._active_matcher()

    def _matched_role_weights(self, targets, indices, device, localization=False):
        labels, weights = super()._matched_role_weights(
            targets, indices, device, localization=localization
        )
        if weights is None:
            return labels, None
        gate = self._x3_loc_gate if localization else self._x3_class_gate
        if gate is None:
            gate = (
                self.balance.x3_loc_gate(device, dtype=weights.dtype)
                if localization
                else self.balance.x3_class_gate(device, dtype=weights.dtype)
            )
        else:
            gate = gate.to(device=device, dtype=weights.dtype)
        return labels, 1.0 + gate * (weights - 1.0)

    def forward(self, outputs, targets, **kwargs):
        active = self._active_matcher()
        x1_aux = self._x1_aux_matcher()
        if hasattr(active, "begin_step"):
            active.begin_step()
        if x1_aux is not active and hasattr(x1_aux, "begin_step"):
            x1_aux.begin_step()
        previous = self.matcher
        self.matcher = active
        self._x3_class_gate = self.balance.x3_class_gate(
            outputs["pred_logits"].device,
            dtype=outputs["pred_logits"].dtype,
        )
        self._x3_loc_gate = self.balance.x3_loc_gate(
            outputs["pred_logits"].device,
            dtype=outputs["pred_logits"].dtype,
        )
        try:
            losses = RTDETRCriterionv2.forward(self, outputs, targets, **kwargs)
        finally:
            self.matcher = previous
        if "x1_fine_logits" not in outputs:
            raise KeyError("NativeHarmonizationCriterion requires PFHM decoder outputs")
        matching_inputs = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = x1_aux(matching_inputs, targets)["indices"]
        x1_result = self.x1_loss(outputs, targets, indices)
        diagnostics = {key: value for key, value in x1_result.items() if key != "loss_x1_fine"}
        balance_weight = self.balance(
            diagnostics,
            self.x3,
            outputs["pred_logits"].device,
            x1_loss_module=self.x1_loss,
        )
        losses["loss_x1_fine"] = x1_result["loss_x1_fine"] * balance_weight.to(
            x1_result["loss_x1_fine"].dtype
        )
        self.last_x1_diagnostics = {
            **diagnostics,
            "x1x3_balance_weight": balance_weight.detach(),
            "x1x3_x3_class_gate": self._x3_class_gate.detach(),
            "x1x3_x3_loc_gate": self._x3_loc_gate.detach(),
            "x1x3_x1_loc_gate": self.balance.x1_loc_gate(
                outputs["pred_logits"].device,
                dtype=outputs["pred_logits"].dtype,
            ).detach(),
            "x1x3_x1_aux_x3_gate": self.balance.x1_aux_x3_gate(
                outputs["pred_logits"].device,
                dtype=outputs["pred_logits"].dtype,
            ).detach(),
        }
        if self.balance.last_state is not None:
            self.last_x1_diagnostics.update(
                {
                    "x1x3_balance_progress": self.balance.last_state.progress,
                    "x1x3_balance_x3_ready": self.balance.last_state.x3_ready,
                    "x1x3_balance_x1_proto_ready": self.balance.last_state.x1_proto_ready,
                    "x1x3_balance_x1_proto_mean": self.balance.last_state.x1_proto_mean,
                    "x1x3_balance_x1_participation_gate": self.balance.last_state.x1_participation_gate,
                    "x1x3_balance_safety": self.balance.last_state.safety,
                    "x1x3_balance_x3_gate": self.balance.last_state.x3_gate,
                    "x1x3_balance_x3_class_gate": self.balance.last_state.x3_class_gate,
                    "x1x3_balance_x3_loc_gate": self.balance.last_state.x3_loc_gate,
                    "x1x3_balance_x1_loc_gate": self.balance.last_state.x1_loc_gate,
                    "x1x3_balance_x1_aux_x3_gate": self.balance.last_state.x1_aux_x3_gate,
                }
            )
        return losses


