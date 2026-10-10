# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Dataset-agnostic diagnosis for the clean ACRHM class-reliability module.

ACRHM-FH reads training annotations only. It never inspects dataset/class names,
validation metrics, experiment paths, or legacy dataset-specific guards.
"""


from __future__ import annotations

import argparse

import json

import math

from dataclasses import asdict, dataclass

from typing import Dict, List, Mapping, Sequence

import torch

import torch.nn as nn

import torch.distributed as dist

def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _project_mean_one(values: Sequence[float], floor: float, cap: float) -> List[float]:
    if not values:
        return []
    if not (0.0 < floor <= 1.0 <= cap):
        raise ValueError(f"infeasible bounds: floor={floor}, cap={cap}")
    positive = [max(float(value), 1.0e-12) for value in values]
    low, high = 0.0, max(cap / min(positive), 1.0)
    for _ in range(80):
        scale = (low + high) * 0.5
        total = sum(_clip(scale * value, floor, cap) for value in positive)
        if total < len(positive):
            low = scale
        else:
            high = scale
    scale = (low + high) * 0.5
    return [_clip(scale * value, floor, cap) for value in positive]


@dataclass(frozen=True)
class ACRHMAnnotationStats:
    num_images: int
    num_instances: int
    num_classes: int
    category_ids: List[int]
    class_counts: List[int]
    instances_per_image: float
    head_dominance: float
    normalized_imbalance: float
    small_object_ratio: float
    min_class_count: int


@dataclass(frozen=True)
class ACRHMAnnotationDecision:
    evidence_reliability: float
    imbalance_risk: float
    density_risk: float
    small_object_risk: float
    classification_need: float
    matching_risk: float
    localization_risk: float
    matcher_strength: float
    classification_strength: float
    localization_strength: float
    frequency_power: float
    difficulty_power: float
    iou_reliability_power: float
    max_class_weight: float
    warmup_ratio: float
    ramp_ratio: float
    prior_anchor_strength: float
    prior_deviation_limit: float
    frequency_prior_weights: List[float]


class ACRHMAnnotationAnalyzer:
    """Infer one continuous ACRHM profile for any COCO-format training set."""

    SMALL_RELATIVE_AREA = 1.0e-3

    @staticmethod
    def stats_from_coco(annotation_file: str) -> ACRHMAnnotationStats:
        with open(annotation_file, "r") as handle:
            payload = json.load(handle)
        images = {item["id"]: item for item in payload.get("images", [])}
        category_ids = [int(item["id"]) for item in payload.get("categories", [])]
        if not images:
            raise ValueError("ACRHM-FH requires at least one training image")
        if not category_ids or len(category_ids) != len(set(category_ids)):
            raise ValueError("ACRHM-FH requires unique declared categories")
        counts: Dict[int, int] = {category_id: 0 for category_id in category_ids}
        image_counts = {image_id: 0 for image_id in images}
        small = 0
        total_annotations = 0
        for annotation in payload.get("annotations", []):
            category_id = int(annotation["category_id"])
            if category_id not in counts:
                category_ids.append(category_id)
                counts[category_id] = 0
            image_id = annotation["image_id"]
            if image_id not in images:
                raise ValueError(f"annotation references missing image id {image_id}")
            image = images[image_id]
            width, height = float(image.get("width", 0)), float(image.get("height", 0))
            if width <= 0 or height <= 0:
                raise ValueError(f"image {image_id} has invalid dimensions")
            box = annotation.get("bbox", [0, 0, 0, 0])
            area = float(annotation.get("area", float(box[2]) * float(box[3])))
            relative_area = max(area, 0.0) / (width * height)
            small += relative_area < ACRHMAnnotationAnalyzer.SMALL_RELATIVE_AREA
            counts[category_id] += 1
            image_counts[image_id] += 1
            total_annotations += 1
        if total_annotations == 0:
            raise ValueError("ACRHM-FH requires at least one training annotation")

        class_counts = [counts[category_id] for category_id in category_ids]
        probabilities = [count / total_annotations for count in class_counts]
        uniform = 1.0 / len(class_counts)
        concentration = sum(probability * probability for probability in probabilities)
        imbalance = (concentration - uniform) / max(1.0 - uniform, 1.0e-6)
        return ACRHMAnnotationStats(
            num_images=len(images),
            num_instances=total_annotations,
            num_classes=len(category_ids),
            category_ids=category_ids,
            class_counts=class_counts,
            instances_per_image=_mean(list(image_counts.values())),
            head_dominance=max(probabilities),
            normalized_imbalance=_clip(imbalance),
            small_object_ratio=small / total_annotations,
            min_class_count=min(class_counts),
        )

    @staticmethod
    def analyze(stats: ACRHMAnnotationStats) -> ACRHMAnnotationDecision:
        evidence = _clip(math.log1p(stats.min_class_count) / math.log1p(1200.0))
        imbalance = _clip(stats.normalized_imbalance)
        density = _clip(math.log1p(stats.instances_per_image) / math.log1p(25.0))
        small = _clip(stats.small_object_ratio / 0.45)
        class_complexity = _clip(math.log2(max(stats.num_classes, 1)) / 4.0)
        classification_need = _clip(
            0.48 * imbalance + 0.22 * density + 0.18 * class_complexity
            + 0.12 * (1.0 - evidence)
        )
        matching_risk = _clip(0.42 * imbalance + 0.30 * density + 0.28 * (1.0 - evidence))
        localization_risk = _clip(0.58 * small + 0.24 * (1.0 - evidence) + 0.18 * density)

        # ACRHM native-reliability: use one continuous pressure curve, not dataset names.
        #
        # Medium pressure should stay in the successful low-disturbance
        # regime: low class/matcher disturbance, mild localization protection,
        # and a class-weight cap around 1.5.  When imbalance, density, and
        # evidence are all high, the curve moves toward the stronger long-tail
        # regime: stronger matcher/classification compensation and a cap near
        # 2.0.  The transition is intentionally sharp only after the data show
        # reliable long-tail density, so ordinary small-object datasets do not
        # become over-weighted just because objects are small.
        raw_pressure = _clip(
            0.55 * _clip(imbalance / 0.25)
            + 0.35 * _clip(density / 0.80)
            + 0.10 * evidence
        )
        long_tail_density = _clip((raw_pressure - 0.45) / 0.50)
        matcher_strength = 0.26 + 0.20 * long_tail_density
        classification_strength = 0.17 + 0.16 * long_tail_density
        max_weight = 1.44 + 0.58 * long_tail_density
        frequency_power = 0.50 + 0.10 * long_tail_density
        difficulty_power = 0.55 + 0.15 * long_tail_density
        iou_power = 0.18 + 0.06 * localization_risk
        localization_strength = 0.032 * small * (1.0 - 0.85 * long_tail_density)
        warmup_ratio = 0.04 + 0.06 * long_tail_density
        ramp_ratio = 0.16 - 0.01 * long_tail_density
        anchor = _clip(0.60 * evidence * (1.0 - imbalance) ** 2, 0.03, 0.45)
        deviation = _clip(0.09 + 0.20 * long_tail_density, 0.09, 0.29)

        observed = [index for index, count in enumerate(stats.class_counts) if count > 0]
        observed_total = sum(stats.class_counts[index] for index in observed)
        uniform = 1.0 / len(observed)
        raw = [
            (uniform / (stats.class_counts[index] / observed_total)) ** frequency_power
            for index in observed
        ]
        projected = _project_mean_one(raw, 1.0 / (2.0 * max_weight), max_weight)
        priors = [1.0] * stats.num_classes
        for index, value in zip(observed, projected):
            priors[index] = value

        return ACRHMAnnotationDecision(
            evidence_reliability=evidence,
            imbalance_risk=imbalance,
            density_risk=density,
            small_object_risk=small,
            classification_need=classification_need,
            matching_risk=matching_risk,
            localization_risk=localization_risk,
            matcher_strength=matcher_strength,
            classification_strength=classification_strength,
            localization_strength=localization_strength,
            frequency_power=frequency_power,
            difficulty_power=difficulty_power,
            iou_reliability_power=iou_power,
            max_class_weight=max_weight,
            warmup_ratio=warmup_ratio,
            ramp_ratio=ramp_ratio,
            prior_anchor_strength=anchor,
            prior_deviation_limit=deviation,
            frequency_prior_weights=priors,
        )

    @classmethod
    def analyze_coco(cls, annotation_file: str) -> Mapping[str, object]:
        stats = cls.stats_from_coco(annotation_file)
        return {"stats": asdict(stats), "decision": asdict(cls.analyze(stats))}


class ACRHMOnlineAnalyzer(nn.Module):
    """Online structural evidence used by ACRHM-FH to refine ACRHM decisions.

    All inputs are training-only observations. The confusion matrix records the
    strongest wrong-class destination. Assignment switch means that ACRHM changes
    the baseline Hungarian query selected for the same GT in the same step.
    """

    def __init__(self, num_classes: int, momentum: float = 0.97):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.register_buffer("confusion_matrix", torch.zeros(num_classes, num_classes))
        self.register_buffer("probability_margin_ema", torch.ones(num_classes))
        self.register_buffer("class_cost_gap_ema", torch.ones(num_classes))
        self.register_buffer("full_cost_gap_ema", torch.ones(num_classes))
        self.register_buffer("assignment_switch_ema", torch.zeros(num_classes))
        self.register_buffer("seen", torch.zeros(num_classes))

    @torch.no_grad()
    def update(
        self,
        true_classes: torch.Tensor,
        wrong_classes: torch.Tensor,
        probability_margin: torch.Tensor,
        class_cost_gap: torch.Tensor,
        full_cost_gap: torch.Tensor,
        assignment_switch: torch.Tensor,
    ) -> None:
        if true_classes.numel() == 0:
            return
        device = self.seen.device
        true_classes = true_classes.detach().to(device=device, dtype=torch.long)
        wrong_classes = wrong_classes.detach().to(device=device, dtype=torch.long)
        probability_margin = probability_margin.detach().to(self.probability_margin_ema)
        class_cost_gap = class_cost_gap.detach().to(self.class_cost_gap_ema)
        full_cost_gap = full_cost_gap.detach().to(self.full_cost_gap_ema)
        assignment_switch = assignment_switch.detach().to(self.assignment_switch_ema).clamp(0, 1)

        count = torch.bincount(true_classes, minlength=self.num_classes).float()
        margin_sum = torch.zeros_like(count).scatter_add_(0, true_classes, probability_margin)
        class_gap_sum = torch.zeros_like(count).scatter_add_(0, true_classes, class_cost_gap)
        full_gap_sum = torch.zeros_like(count).scatter_add_(0, true_classes, full_cost_gap)
        switch_sum = torch.zeros_like(count).scatter_add_(0, true_classes, assignment_switch)
        matrix = torch.zeros_like(self.confusion_matrix)
        valid_wrong = (wrong_classes >= 0) & (wrong_classes < self.num_classes)
        flat = true_classes[valid_wrong] * self.num_classes + wrong_classes[valid_wrong]
        matrix.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=matrix.dtype))
        if dist.is_available() and dist.is_initialized():
            for tensor in (count, margin_sum, class_gap_sum, full_gap_sum, switch_sum, matrix):
                dist.all_reduce(tensor)

        momentum = self.momentum
        for index in torch.nonzero(count > 0, as_tuple=False).flatten().tolist():
            denominator = count[index]
            self.probability_margin_ema[index].lerp_(margin_sum[index] / denominator, 1 - momentum)
            self.class_cost_gap_ema[index].lerp_(class_gap_sum[index] / denominator, 1 - momentum)
            self.full_cost_gap_ema[index].lerp_(full_gap_sum[index] / denominator, 1 - momentum)
            self.assignment_switch_ema[index].lerp_(switch_sum[index] / denominator, 1 - momentum)
            self.seen[index].add_(denominator)
        self.confusion_matrix.add_(matrix)

    def decision_factors(self) -> Mapping[str, torch.Tensor]:
        evidence = (self.seen / 128.0).clamp(0.0, 1.0)
        margin_risk = (1.0 - self.probability_margin_ema).clamp(0.0, 2.0)
        class_gap_risk = (1.0 - self.class_cost_gap_ema).clamp(0.0, 2.0)
        full_gap_risk = (1.0 - self.full_cost_gap_ema).clamp(0.0, 2.0)
        row_sum = self.confusion_matrix.sum(dim=1).clamp(min=1.0)
        pair_concentration = self.confusion_matrix.max(dim=1).values / row_sum
        structural_need = (
            0.28 * margin_risk
            + 0.22 * class_gap_risk
            + 0.18 * full_gap_risk
            + 0.18 * pair_concentration
            + 0.14 * self.assignment_switch_ema
        )
        structural_need = 1.0 + evidence * structural_need
        return {
            "evidence": evidence,
            "structural_need": structural_need,
            "margin_risk": margin_risk,
            "class_gap_risk": class_gap_risk,
            "full_gap_risk": full_gap_risk,
            "pair_concentration": pair_concentration,
            "assignment_switch": self.assignment_switch_ema,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-agnostic ACRHM-FH diagnosis")
    parser.add_argument("annotations", nargs="+")
    args = parser.parse_args()
    for annotation_file in args.annotations:
        print(json.dumps(
            {"annotations": annotation_file, **ACRHMAnnotationAnalyzer.analyze_coco(annotation_file)},
            indent=2,
        ))


if __name__ == "__main__":
    main()


