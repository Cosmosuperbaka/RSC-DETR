# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Dataset-agnostic risk analysis for the PFHM fine head (PFHM-FH).

This module deliberately does not inspect dataset names, class names, validation
metrics, or experiment paths.  Its static decision is derived only from training
annotations.  Online classification/localization evidence can later refine the
frequency prior without changing the dataset-agnostic formulas defined here.
"""


from __future__ import annotations

import argparse

import json

import math

from dataclasses import asdict, dataclass

from typing import Dict, List, Mapping, Sequence

def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = _clip(q) * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def _project_mean_one(values: Sequence[float], floor: float, cap: float) -> List[float]:
    """Project positive values to a bounded vector whose arithmetic mean is one."""
    if not values:
        return []
    if not (0.0 < floor <= 1.0 <= cap):
        raise ValueError(f"infeasible weight bounds: floor={floor}, cap={cap}")
    positive = [max(float(value), 1.0e-12) for value in values]
    target = float(len(positive))
    low, high = 0.0, max(cap / min(positive), 1.0)
    for _ in range(80):
        scale = (low + high) * 0.5
        total = sum(_clip(scale * value, floor, cap) for value in positive)
        if total < target:
            low = scale
        else:
            high = scale
    scale = (low + high) * 0.5
    return [_clip(scale * value, floor, cap) for value in positive]


@dataclass(frozen=True)
class PFHMAnnotationStats:
    num_images: int
    num_instances: int
    num_classes: int
    category_ids: List[int]
    class_counts: List[int]
    instances_per_image: float
    head_dominance: float
    normalized_imbalance: float
    small_object_ratio: float
    median_relative_area: float
    min_class_count: int


@dataclass(frozen=True)
class PFHMAnnotationDecision:
    # Interpretable risks, all in [0, 1].
    evidence_reliability: float
    imbalance_risk: float
    small_object_risk: float
    density_risk: float
    sample_scarcity_risk: float
    classification_need: float
    static_ap_safety_prior: float
    static_ap50_safety_prior: float
    static_ap75_safety_prior: float

    # Initial PFHM controls. Online evidence may refine these conservatively.
    fine_loss_scale: float
    fine_grad_scale: float
    fine_start_factor: float
    fine_ramp_factor: float
    # Delayed AP50 coverage release. A value near zero keeps the static matched
    # IoU/coverage policy unchanged; larger values release more coverage later.
    ap50_coverage_release: float
    # The criterion derives a threshold from this matched-IoU quantile while
    # retaining at least min_fine_sample_coverage of eligible matches.
    fine_iou_quantile: float
    min_fine_sample_coverage: float
    small_object_scale: float
    class_weight_power: float
    class_weight_cap: float
    frequency_prior_weights: List[float]


class PFHMAnnotationAnalyzer:
    """Compute an PFHM-FH static profile using one formula for every dataset."""

    SMALL_RELATIVE_AREA = 1.0e-3

    @staticmethod
    def stats_from_coco(annotation_file: str) -> PFHMAnnotationStats:
        with open(annotation_file, "r") as handle:
            payload = json.load(handle)

        images = {item["id"]: item for item in payload.get("images", [])}
        category_ids = [item["id"] for item in payload.get("categories", [])]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("COCO categories contain duplicate category ids")
        counts: Dict[int, int] = {cid: 0 for cid in category_ids}
        image_counts: Dict[int, int] = {iid: 0 for iid in images}
        relative_areas: List[float] = []

        for annotation in payload.get("annotations", []):
            cid = annotation.get("category_id")
            if cid not in counts:
                counts[cid] = 0
                category_ids.append(cid)
            counts[cid] += 1
            image_id = annotation.get("image_id")
            image_counts[image_id] = image_counts.get(image_id, 0) + 1
            image = images.get(image_id, {})
            image_area = max(float(image.get("width", 1)) * float(image.get("height", 1)), 1.0)
            box = annotation.get("bbox", [0, 0, 0, 0])
            area = float(annotation.get("area", float(box[2]) * float(box[3])))
            relative_areas.append(max(area, 0.0) / image_area)

        class_counts = [counts[cid] for cid in category_ids]
        total = sum(class_counts)
        num_classes = len(class_counts)
        if not images:
            raise ValueError("PFHM-FH requires at least one training image")
        if num_classes == 0:
            raise ValueError("PFHM-FH requires at least one declared or observed category")
        if total == 0:
            raise ValueError("PFHM-FH requires at least one training annotation")
        probabilities = [count / max(total, 1) for count in class_counts]
        uniform = 1.0 / max(num_classes, 1)
        concentration = sum(probability ** 2 for probability in probabilities)
        normalized_imbalance = (
            (concentration - uniform) / max(1.0 - uniform, 1.0e-6)
            if num_classes > 1 else 0.0
        )
        small = sum(area < PFHMAnnotationAnalyzer.SMALL_RELATIVE_AREA for area in relative_areas)

        return PFHMAnnotationStats(
            num_images=len(images),
            num_instances=total,
            num_classes=num_classes,
            category_ids=category_ids,
            class_counts=class_counts,
            instances_per_image=_mean(list(image_counts.values())),
            head_dominance=max(probabilities, default=0.0),
            normalized_imbalance=_clip(normalized_imbalance),
            small_object_ratio=small / max(len(relative_areas), 1),
            median_relative_area=_quantile(relative_areas, 0.5),
            min_class_count=min(class_counts),
        )

    @staticmethod
    def analyze(stats: PFHMAnnotationStats) -> PFHMAnnotationDecision:
        # Reliability grows with the least observed class, not total dataset size.
        # This prevents a dominant head class from hiding unsupported tail classes.
        evidence = _clip(math.log1p(stats.min_class_count) / math.log1p(1200.0))
        imbalance = _clip(stats.normalized_imbalance)
        small = _clip(stats.small_object_ratio / 0.45)
        density = _clip(math.log1p(stats.instances_per_image) / math.log1p(25.0))
        scarcity = 1.0 - evidence

        # More classes, imbalance, and density increase the need for an auxiliary
        # classifier. Small/scarce data increase need slightly but reduce safety.
        class_complexity = _clip(math.log2(max(stats.num_classes, 1)) / 4.0)
        need = _clip(
            0.34 * imbalance
            + 0.24 * density
            + 0.24 * class_complexity
            + 0.10 * small
            + 0.08 * scarcity
        )

        # Three separate safety axes protect overall AP, recall-like AP50, and
        # high-quality localization AP75. They are training-signal proxies only.
        static_ap_safety = _clip(1.0 - 0.38 * scarcity - 0.18 * imbalance - 0.10 * density)
        static_ap50_safety = _clip(1.0 - 0.30 * scarcity - 0.22 * imbalance - 0.13 * density)
        static_ap75_safety = _clip(1.0 - 0.42 * small - 0.28 * scarcity - 0.08 * density)
        joint_safety = min(static_ap_safety, static_ap50_safety, static_ap75_safety)

        # Frequency defines only the initial shape. Online difficulty and IoU
        # reliability must later move each weight toward/away from this prior.
        weight_power = 0.18 + 0.58 * imbalance * evidence
        cap = 1.25 + 0.95 * imbalance * evidence
        floor = max(0.25, 1.0 / max(cap * 2.0, 1.0))
        observed = [index for index, count in enumerate(stats.class_counts) if count > 0]
        observed_total = sum(stats.class_counts[index] for index in observed)
        observed_uniform = 1.0 / max(len(observed), 1)
        raw_observed = [
            (observed_uniform / (stats.class_counts[index] / observed_total)) ** weight_power
            for index in observed
        ]
        projected = _project_mean_one(raw_observed, floor, cap)
        weights = [1.0] * stats.num_classes
        for index, weight in zip(observed, projected):
            weights[index] = weight

        # Reliable dense data with low small-object risk usually needs more AP50
        # coverage rather than stricter matched-IoU filtering. Gate this by AP75
        # safety so recall support does not become a localization shortcut.
        ap50_coverage_need = (
            evidence
            * density
            * (1.0 - small)
            * (0.45 + 0.55 * imbalance)
            * static_ap75_safety
        )

        # Strong relative reweighting is allowed on reliable imbalanced data, but
        # global loss/gradient remain limited by the weakest AP safety axis.
        fine_loss = 0.14 + 0.34 * need * joint_safety * (0.55 + 0.45 * evidence)
        grad = 0.12 + 0.34 * need * static_ap_safety * static_ap75_safety * evidence

        # This is a requested matched-IoU quantile, not an absolute IoU cutoff.
        # High small-object risk asks for a more selective quantile, while the
        # coverage floor prevents AP/AP50 loss from discarding too many matches.
        iou_quantile = _clip(
            0.30 + 0.28 * small + 0.10 * scarcity - 0.045 * ap50_coverage_need,
            0.25,
            0.65,
        )
        min_coverage = _clip(
            0.82 - 0.18 * small - 0.08 * scarcity + 0.055 * ap50_coverage_need,
            0.58,
            0.88,
        )

        return PFHMAnnotationDecision(
            evidence_reliability=evidence,
            imbalance_risk=imbalance,
            small_object_risk=small,
            density_risk=density,
            sample_scarcity_risk=scarcity,
            classification_need=need,
            static_ap_safety_prior=static_ap_safety,
            static_ap50_safety_prior=static_ap50_safety,
            static_ap75_safety_prior=static_ap75_safety,
            fine_loss_scale=_clip(fine_loss, 0.14, 0.42),
            fine_grad_scale=_clip(grad, 0.12, 0.38),
            fine_start_factor=1.0 + 0.70 * scarcity + 0.45 * small,
            fine_ramp_factor=1.0 + 0.55 * scarcity + 0.35 * small + 0.12 * density,
            ap50_coverage_release=_clip(ap50_coverage_need, 0.0, 1.0),
            fine_iou_quantile=iou_quantile,
            min_fine_sample_coverage=min_coverage,
            small_object_scale=_clip(1.0 - 0.42 * small * (0.45 + 0.55 * scarcity), 0.45, 1.0),
            class_weight_power=weight_power,
            class_weight_cap=cap,
            frequency_prior_weights=weights,
        )

    @classmethod
    def analyze_coco(cls, annotation_file: str) -> Mapping[str, object]:
        stats = cls.stats_from_coco(annotation_file)
        decision = cls.analyze(stats)
        return {"stats": asdict(stats), "decision": asdict(decision)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset-agnostic PFHM-FH static analysis")
    parser.add_argument("annotations", nargs="+", help="COCO-format training annotation JSON files")
    args = parser.parse_args()
    for annotation_file in args.annotations:
        print(json.dumps({"annotations": annotation_file, **PFHMAnnotationAnalyzer.analyze_coco(annotation_file)}, indent=2))


if __name__ == "__main__":
    main()


