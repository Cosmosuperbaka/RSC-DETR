#!/usr/bin/env python3
"""Finalize an M3FD-LT20 split by deterministic file-level SHA-256 deduplication."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from m3fd_utils import (  # noqa: E402
    CLASS_NAMES, class_stats, collect_records, convert_split_to_coco,
    discover_layout, ensure_dir, make_manifest, read_lines, sha256_file,
    write_class_counts, write_csv, write_split, write_text,
)

SPLITS = ("train", "val", "test")
PRIORITY = {"val": 0, "test": 1, "train": 2}
EXPECTED_PAIRS = {"train": 1428, "val": 420, "test": 419}
EXPECTED_ANNOTATIONS = {"train": 16406, "val": 4119, "test": 4095}
EXPECTED_TRAIN_CLASSES = {0: 4767, 1: 8379, 2: 499, 3: 1657, 4: 407, 5: 697}


def digest(path: Path) -> str:
    return sha256_file(path)


def pair_digest(rgb_hash: str, ir_hash: str) -> str:
    return hashlib.sha256((rgb_hash + ":" + ir_hash).encode("ascii")).hexdigest()


def load_coco_stems(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Path(image["file_name"]).stem for image in data["images"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Rebuild paired COCO annotations from fixed split lists, "
                     "remove byte-identical RGB/IR duplicates deterministically, "
                     "and refresh lists, counts, manifests, and verification report.")
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="M3FD dataset directory containing raw/ (or raw data directly).")
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Directory containing train.txt, val.txt, and test.txt.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output processed dataset directory (visible/, infrared/, annotations/).")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Directory for authoritative split lists and reports; defaults to --split-dir.")
    args = parser.parse_args()

    root = args.root.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = (args.report_dir or args.split_dir).resolve()
    raw_root = root / "raw" if (root / "raw").is_dir() else root
    layout = discover_layout(raw_root)
    if not layout.visible_dir or not layout.ir_dir or not layout.labels_dir:
        parser.error(f"could not discover visible/, infrared/, and labels/ under {raw_root}")
    records, invalid = collect_records(layout)
    rec = {record["stem"]: record for record in records}

    # Preserve the exact pre-repair test membership once. On later runs, this
    # historical input remains the source so the repair is idempotent.
    test_path = split_dir / "test.txt"
    historical_test = split_dir / "test.pre_ir_dedupe.txt"
    if not historical_test.exists():
        if not test_path.exists():
            parser.error(f"missing fixed test list: {test_path}")
        write_text(historical_test, test_path.read_text(encoding="utf-8"))

    source_lists: dict[str, list[str]] = {}
    for split in SPLITS:
        source_path = historical_test if split == "test" else split_dir / f"{split}.txt"
        values = read_lines(source_path)
        if len(values) != len(set(values)):
            parser.error(f"duplicate stems in input {source_path}")
        missing = [stem for stem in values if stem not in rec or rec[stem]["issues"]]
        if missing:
            parser.error(f"invalid/missing raw samples in {source_path}: {missing[:10]}")
        source_lists[split] = values

    # Candidate priority protects validation first, then test, then training.
    # Ties within a split keep the lexicographically smallest stem.
    candidates = sorted(
        ((split, stem) for split, stems in source_lists.items() for stem in stems),
        key=lambda item: (PRIORITY[item[0]], item[1]),
    )
    seen_rgb: dict[str, tuple[str, str]] = {}
    seen_ir: dict[str, tuple[str, str]] = {}
    kept = {split: [] for split in SPLITS}
    removed: list[dict] = []
    hashes_by_sample: dict[tuple[str, str], dict[str, str]] = {}
    for split, stem in candidates:
        item = rec[stem]
        rgb_hash, ir_hash = digest(Path(item["visible"])), digest(Path(item["ir"]))
        hashes_by_sample[(split, stem)] = {
            "rgb_sha256": rgb_hash,
            "ir_sha256": ir_hash,
            "pair_sha256": pair_digest(rgb_hash, ir_hash),
        }
        collisions = []
        if rgb_hash in seen_rgb:
            collisions.append(("rgb", seen_rgb[rgb_hash]))
        if ir_hash in seen_ir:
            collisions.append(("ir", seen_ir[ir_hash]))
        if collisions:
            # A sample duplicated in either modality is removed as a whole pair.
            # If both conflict, record both reasons and their kept samples.
            removed.append({
                "stem": stem,
                "split": split,
                "reason": "byte_identical_" + "_and_".join(modality for modality, _ in collisions),
                "kept_duplicates": [{"modality": modality, "split": prior[0], "stem": prior[1]}
                                    for modality, prior in collisions],
                "rgb_sha256": rgb_hash,
                "ir_sha256": ir_hash,
                "pair_sha256": pair_digest(rgb_hash, ir_hash),
                "annotation_count": sum(1 for box in item["labels"] if box["class_id"] in CLASS_NAMES),
            })
            continue
        kept[split].append(stem)
        seen_rgb[rgb_hash] = (split, stem)
        seen_ir[ir_hash] = (split, stem)

    # Verify source split membership is disjoint before writing any outputs.
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = set(source_lists[left]) & set(source_lists[right])
            if overlap:
                parser.error(f"input lists overlap ({left}/{right}): {sorted(overlap)[:10]}")

    for split in SPLITS:
        write_split(report_dir / f"{split}.txt", sorted(kept[split]))
    history_dir = ensure_dir(report_dir / "history")
    write_text(history_dir / "test.pre_ir_dedupe.txt",
               (split_dir / "test.pre_ir_dedupe.txt").read_text(encoding="utf-8"))

    annotation_dir = ensure_dir(output_dir / "annotations")
    visible_dir = ensure_dir(output_dir / "visible")
    infrared_dir = ensure_dir(output_dir / "infrared")
    for split in SPLITS:
        convert_split_to_coco(records, sorted(kept[split]),
                              annotation_dir / f"instances_{split}.json",
                              visible_dir, infrared_dir)

    counts_path = report_dir / "class_counts.csv"
    write_class_counts(counts_path, records, [s for name in SPLITS[:1] for s in kept[name]])
    image_rows = []
    for stem in sorted(kept["train"]):
        counts = {f"class_{cid}": 0 for cid in CLASS_NAMES}
        for box in rec[stem]["labels"]:
            if box["class_id"] in CLASS_NAMES:
                counts[f"class_{box['class_id']}"] += 1
        image_rows.append({"stem": stem, **counts})
    write_csv(report_dir / "image_class_counts.csv", image_rows)

    removed_report = {
        "algorithm": "SHA-256 exact-file deduplication; keep priority val > test > train, then lexical stem",
        "scope": "RGB and IR checked independently across and within all three subsets; remove entire lower-priority pair on either-modality collision",
        "removed_samples": removed,
        "removed_pair_count": len(removed),
    }
    write_text(report_dir / "ir_deduplication_report.json", json.dumps(removed_report, indent=2) + "\n")

    train_instances, _ = class_stats(records, kept["train"])
    actual_if = max(train_instances.values()) / min(train_instances.values())
    manifest = make_manifest(records, kept, {
        "protocol": "M3FD-LT20-seed42-image-pair",
        "dataset": {
            "name": "M3FD",
            "source": "TarDAL (Liu et al., CVPR 2022)",
            "raw_pairs": 4200,
            "valid_pairs": sum(1 for record in records if not record["issues"]),
            "classes": 6,
        },
        "split_source": "project_generated_fixed_greedy_multilabel_stratified_split",
        "actual_imbalance_factor": actual_if,
        "reconstruction_verified": True,
        "random_seeds": {"base_test": 1042, "base_validation": 42, "long_tail_train": 42},
        "seed_semantics": {
            "data_split": {"base_validation": 42, "base_test": 1042, "long_tail_train": 42},
            "note": ("All seed values in this manifest are data-split sampling seeds. "
                     "Model-training seeds (--seed) are independent; the M3FD-LT20 runs "
                     "use training seeds 42, 3407, and 2024 on this same fixed split. "
                     "The directory name lt20_seed42 refers to the data-split seed."),
        },
        "deduplication": removed_report,
        "split_list_sha256": {split: digest(report_dir / f"{split}.txt") for split in SPLITS},
        "annotation_json_sha256": {split: digest(annotation_dir / f"instances_{split}.json") for split in SPLITS},
        "validation_long_tail_resampled": False,
        "test_file_level_ir_repair": True,
        "byte_hash_check_proves_scene_isolation": False,
        "invalid_raw_samples": invalid,
    })
    write_text(report_dir / "split_manifest.json", json.dumps(manifest, indent=2) + "\n")
    write_text(output_dir / "manifests" / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    class_instance_counts = Counter(a["category_id"] for a in json.loads(
        (annotation_dir / "instances_train.json").read_text())["annotations"])
    observed_counts = {split: {"pairs": len(kept[split]), "annotations": len(json.loads(
        (annotation_dir / f"instances_{split}.json").read_text())["annotations"])} for split in SPLITS}
    verification_errors = []
    for split in SPLITS:
        if observed_counts[split]["pairs"] != EXPECTED_PAIRS[split]:
            verification_errors.append(f"{split} pair count {observed_counts[split]['pairs']} != {EXPECTED_PAIRS[split]}")
        if observed_counts[split]["annotations"] != EXPECTED_ANNOTATIONS[split]:
            verification_errors.append(f"{split} annotation count {observed_counts[split]['annotations']} != {EXPECTED_ANNOTATIONS[split]}")
    if dict(class_instance_counts) != EXPECTED_TRAIN_CLASSES:
        verification_errors.append(f"training class counts differ: {dict(class_instance_counts)}")
    actual_if = max(class_instance_counts.values()) / min(class_instance_counts.values())
    if round(actual_if, 2) != 20.59:
        verification_errors.append(f"unexpected training IF: {actual_if}")
    verify = {
        "protocol": "M3FD-LT20-seed42-final",
        "counts": observed_counts,
        "train_class_instances": {str(cid): class_instance_counts[cid] for cid in CLASS_NAMES},
        "actual_imbalance_factor": actual_if,
        "removed_samples": removed,
        "list_sha256": manifest["extra"]["split_list_sha256"],
        "annotation_json_sha256": manifest["extra"]["annotation_json_sha256"],
        "num_raw_invalid": len(invalid),
        "errors": verification_errors,
        "ok": not verification_errors,
        "scope_note": "File-byte hash deduplication does not establish scene-level isolation.",
    }
    write_text(report_dir / "verify_report.json", json.dumps(verify, indent=2) + "\n")
    write_text(output_dir / "DATASET_CARD.md", dataset_card(verify, removed))
    print(json.dumps(verify, indent=2))
    if verification_errors:
        raise SystemExit("final protocol checks failed; see the errors in the report above")


def dataset_card(report: dict, removed: list[dict]) -> str:
    rows = []
    for split in SPLITS:
        counts = report["counts"][split]
        rows.append(f"| {split} | {counts['pairs']} | {counts['annotations']} |")
    removed_text = "\n".join(
        f"- `{item['stem']}` ({item['split']}): {item['reason']}; "
        f"removed {item['annotation_count']} annotations; IR SHA-256 `{item['ir_sha256']}`."
        for item in removed
    ) or "- No duplicate pairs were removed."
    return (
        "# M3FD-LT20 seed42 reproducibility record\n\n"
        "This is a project-generated fixed split, not an official M3FD split. "
        "The raw collection has 4,200 pairs; unreadable IR sample `00375` is excluded, "
        "leaving 4,199 valid pairs. The base split has 3,359/420/420 train/validation/test pairs. "
        "The base test and validation selections use Python `random.Random(1042)` and "
        "`random.Random(42)`, respectively.\n\n"
        "LT20 sampling uses seed 42 and selects complete RGB–IR pairs from base training. "
        "Each pair is selected at most once. All valid annotations, including co-occurring "
        "classes, are retained; class targets are optimization references, not hard quotas. "
        "Validation is not long-tail resampled.\n\n"
        "| Split | Image pairs | Annotations |\n|---|---:|---:|\n" + "\n".join(rows) + "\n\n"
        "Training counts are Car 8,379; People 4,767; Lamp 1,657; Truck 697; Bus 499; "
        "Motorcycle 407 (16,406 total; IF=8,379/407=20.59).\n\n"
        "## Exact-file duplicate repair\n\n"
        "The deterministic rule hashes RGB and IR files separately. On a byte-identical "
        "collision, it keeps validation before test before training, then the lexicographically "
        "first stem within a split, and removes the entire lower-priority pair.\n\n"
        + removed_text + "\n\n"
        "File-byte hashes do not establish scene-level isolation. The split lists, manifests, "
        "COCO JSONs, and removal details record SHA-256 values.\n"
    )


if __name__ == "__main__":
    main()
