#!/usr/bin/env python3
"""Strictly verify final M3FD-LT20 lists, COCO annotations, and file hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m3fd_utils import sha256_file  # noqa: E402

EXPECTED_PAIRS = {"train": 1428, "val": 420, "test": 419}
EXPECTED_ANN = {"train": 16406, "val": 4119, "test": 4095}
EXPECTED_CLASS = {1: 8379, 0: 4767, 3: 1657, 5: 697, 2: 499, 4: 407}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True,
                        help="Directory containing final train.txt, val.txt, test.txt and split_manifest.json.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Processed dataset directory containing annotations/, visible/, infrared/, manifests/.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON report output path; verification summary is always printed.")
    args = parser.parse_args()
    split_dir, dataset_dir = args.split_dir.resolve(), args.dataset_dir.resolve()
    lists, coco, hashes, counts = {}, {}, {}, {}
    all_rgb, all_ir, all_pairs = {}, {}, {}
    errors = []
    for split in ("train", "val", "test"):
        list_path = split_dir / f"{split}.txt"
        ann_path = dataset_dir / "annotations" / f"instances_{split}.json"
        stems = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        data = json.loads(ann_path.read_text())
        list_names = [Path(stem).stem for stem in stems]
        json_names = [Path(image["file_name"]).stem for image in data["images"]]
        if len(stems) != len(set(stems)):
            errors.append(f"{split}: duplicate list stems")
        if len(json_names) != len(set(json_names)):
            errors.append(f"{split}: duplicate JSON filenames")
        if sorted(list_names) != sorted(json_names):
            errors.append(f"{split}: list/COCO filename mismatch")
        if len(stems) != EXPECTED_PAIRS[split]:
            errors.append(f"{split}: expected {EXPECTED_PAIRS[split]} pairs, found {len(stems)}")
        if len(data["annotations"]) != EXPECTED_ANN[split]:
            errors.append(f"{split}: expected {EXPECTED_ANN[split]} annotations, found {len(data['annotations'])}")
        image_by_id = {image["id"]: image for image in data["images"]}
        classes = Counter(annotation["category_id"] for annotation in data["annotations"])
        counts[split] = {"pairs": len(data["images"]), "annotations": len(data["annotations"]),
                         "class_instances": {str(key): classes[key] for key in sorted(classes)}}
        rgb_paths, ir_paths = {}, {}
        for image in data["images"]:
            stem = Path(image["file_name"]).stem
            rgb = dataset_dir / "visible" / image["file_name"]
            ir_name = image.get("ir_file_name", image["file_name"])
            ir = dataset_dir / "infrared" / ir_name
            if not rgb.is_file() or not ir.is_file():
                errors.append(f"{split}/{stem}: missing RGB or IR file")
                continue
            rgb_hash, ir_hash = sha256_file(rgb), sha256_file(ir)
            pair_hash = hashlib.sha256((rgb_hash + ":" + ir_hash).encode("ascii")).hexdigest()
            rgb_paths[stem], ir_paths[stem] = rgb_hash, ir_hash
            for modality, value, index in (("rgb", rgb_hash, all_rgb), ("ir", ir_hash, all_ir),
                                           ("pair", pair_hash, all_pairs)):
                if value in index:
                    errors.append(f"duplicate {modality} bytes: {index[value]} and {split}/{stem}")
                else:
                    index[value] = f"{split}/{stem}"
        lists[split] = stems
        coco[split] = data
        hashes[split] = {"list_sha256": sha256_file(list_path), "annotation_json_sha256": sha256_file(ann_path)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(lists[a]) & set(lists[b]):
            errors.append(f"list overlap: {a}/{b}")
    observed_class = Counter(a["category_id"] for a in coco["train"]["annotations"])
    if dict(observed_class) != EXPECTED_CLASS:
        errors.append(f"training class counts differ: {dict(observed_class)}")
    factor = max(observed_class.values()) / min(observed_class.values())
    if round(factor, 2) != 20.59:
        errors.append(f"unexpected imbalance factor: {factor}")
    if "00539" in {Path(x).stem for x in lists["test"]}:
        errors.append("test list still contains removed sample 00539")
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest.get("extra", {}).get("split_list_sha256", {})
        recorded_json = manifest.get("extra", {}).get("annotation_json_sha256", {})
        for split in lists:
            if recorded.get(split) != hashes[split]["list_sha256"]:
                errors.append(f"manifest list hash mismatch: {split}")
            if recorded_json.get(split) != hashes[split]["annotation_json_sha256"]:
                errors.append(f"manifest annotation hash mismatch: {split}")
    summary = {"counts": counts, "actual_imbalance_factor": factor, "hashes": hashes,
               "duplicate_file_bytes": False, "errors": errors, "ok": not errors,
               "scope_note": "File-byte hash checks do not establish scene-level isolation."}
    rendered = json.dumps(summary, indent=2) + "\n"
    print(rendered, end="")
    if args.report:
        args.report.resolve().write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
