#!/usr/bin/env python3
"""Check that a dataset is intact before you trust a number that came out of it.

Silent dataset rot is the worst kind of reproducibility bug: nothing crashes,
the metrics just drift. This tool verifies the things that actually go wrong --
missing images, images that cannot be decoded, annotation/image mismatches,
duplicate images across splits, and box coordinates outside the frame.

    python3 tools/check_dataset.py --dataset vedai
    python3 tools/check_dataset.py --dataset m3fd_lt20 --check-duplicates
    python3 tools/check_dataset.py --root /path/to/dataset --report out.json

Exit code is 0 when clean, 1 when any ERROR-level finding is present, so it can
gate CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import sys as _sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        _sys.path.insert(0, str(_parent))
        break

from rscdetr_paths import DATASETS, ROOT  # noqa: E402

PRESETS = {
    "vedai": {
        "root": DATASETS / "VEDAI",
        "annotations": ["annotations/vedai_fold01_test_class8.json"],
        "images": ["Vehicules1024"],
        "paired_suffixes": [("_co", "_ir")],
    },
    "m3fd_lt20": {
        "root": DATASETS / "M3FD/processed/lt20_seed42",
        "annotations": ["annotations/instances_train.json",
                        "annotations/instances_test.json"],
        "images": ["images"],
        "paired_suffixes": [],
    },
    "dvtod": {
        "root": ROOT / "datasets/RTDOD_HBB_3class",
        "annotations": ["annotations/instances_val.json"],
        "images": ["images"],
        "paired_suffixes": [],
    },
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class Report:
    def __init__(self) -> None:
        self.findings: list[dict] = []

    def add(self, level: str, kind: str, detail: str) -> None:
        self.findings.append({"level": level, "kind": kind, "detail": detail})

    def error(self, kind: str, detail: str) -> None:
        self.add("ERROR", kind, detail)

    def warn(self, kind: str, detail: str) -> None:
        self.add("WARN", kind, detail)

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f["level"] == "ERROR")

    @property
    def warnings(self) -> int:
        return sum(1 for f in self.findings if f["level"] == "WARN")


def check_annotations(root: Path, names: list[str], report: Report) -> dict:
    """Load every annotation file and cross-check its internal consistency."""
    stats = {}
    for name in names:
        path = root / name
        if not path.is_file():
            report.error("annotation-missing", str(path))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.error("annotation-corrupt", f"{path}: {exc}")
            continue

        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = data.get("categories", [])

        if not images:
            report.error("annotation-empty", f"{path}: no images")
        if not anns:
            report.error("annotation-empty", f"{path}: no annotations")
        if not cats:
            report.error("annotation-empty", f"{path}: no categories")

        image_ids = {im["id"] for im in images}
        orphan = [a["id"] for a in anns if a["image_id"] not in image_ids]
        if orphan:
            report.error("annotation-orphan",
                         f"{path}: {len(orphan)} annotations reference unknown images")

        cat_ids = {c["id"] for c in cats}
        bad_cat = [a["id"] for a in anns if a["category_id"] not in cat_ids]
        if bad_cat:
            report.error("annotation-bad-category",
                         f"{path}: {len(bad_cat)} annotations use undefined categories")

        # degenerate or out-of-frame boxes
        degenerate = out_of_frame = 0
        sizes = {}
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                degenerate += 1
            sizes.setdefault(a["image_id"], None)
            if a["image_id"] in {im["id"]: im for im in images}:
                im = next(im for im in images if im["id"] == a["image_id"])
                if x < -1 or y < -1 or x + w > im["width"] + 1 or y + h > im["height"] + 1:
                    out_of_frame += 1
        if degenerate:
            report.warn("degenerate-boxes",
                        f"{path}: {degenerate} boxes with area <= 1px")
        if out_of_frame:
            report.error("box-out-of-frame",
                         f"{path}: {out_of_frame} boxes extend past the image")

        counts = Counter(a["category_id"] for a in anns)
        stats[name] = {
            "images": len(images),
            "annotations": len(anns),
            "categories": len(cats),
            "instances_per_class": {str(k): v for k, v in sorted(counts.items())},
        }
    return stats


def check_images(root: Path, dirs: list[str], report: Report) -> dict:
    """Count images, verify they decode, and check modality pairing."""
    found = {}
    for name in dirs:
        directory = root / name
        if not directory.is_dir():
            report.error("image-dir-missing", str(directory))
            continue
        files = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXT]
        if not files:
            report.error("image-dir-empty", str(directory))
            continue
        found[name] = files

    total = 0
    for name, files in found.items():
        total += len(files)
        unreadable = []
        for path in files:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:  # noqa: BLE001
                unreadable.append(path.name)
        if unreadable:
            report.error("image-unreadable",
                         f"{name}: {len(unreadable)} files fail to decode "
                         f"(e.g. {', '.join(unreadable[:3])})")
    return {"total_images": total,
            "per_directory": {k: len(v) for k, v in found.items()}}


def check_pairing(files: list[Path], suffixes: list[tuple[str, str]],
                  report: Report) -> None:
    """RGB/IR pairs must exist for every stem."""
    for a_suffix, b_suffix in suffixes:
        a = {p.stem[: -len(a_suffix)] for p in files
             if p.stem.endswith(a_suffix)}
        b = {p.stem[: -len(b_suffix)] for p in files
             if p.stem.endswith(b_suffix)}
        if not a and not b:
            continue
        only_a = a - b
        only_b = b - a
        if only_a:
            report.error("pairing-broken",
                         f"{len(only_a)} stems have {a_suffix} but no {b_suffix} "
                         f"(e.g. {sorted(only_a)[:3]})")
        if only_b:
            report.error("pairing-broken",
                         f"{len(only_b)} stems have {b_suffix} but no {a_suffix} "
                         f"(e.g. {sorted(only_b)[:3]})")
        if not only_a and not only_b:
            report.warn("pairing-ok", f"{len(a)} complete {a_suffix}/{b_suffix} pairs")


def check_duplicates(directories: dict[str, list[Path]], report: Report) -> None:
    """Flag identical images appearing more than once (across or within splits)."""
    seen: dict[str, list[str]] = {}
    for name, files in directories.items():
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            seen.setdefault(digest, []).append(f"{name}/{path.name}")

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        cross_split = sum(
            1 for v in dupes.values()
            if len({item.split("/")[0] for item in v}) > 1
        )
        report.warn("duplicate-images",
                    f"{len(dupes)} duplicate groups, "
                    f"{cross_split} of them span different directories")
        for digest, members in list(dupes.items())[:5]:
            report.warn("duplicate-example",
                        f"{digest[:12]}: {', '.join(members[:4])}")
    else:
        report.warn("duplicates-none", f"{len(seen)} unique images, no duplicates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(PRESETS))
    parser.add_argument("--root", type=Path, help="override the preset root")
    parser.add_argument("--check-duplicates", action="store_true")
    parser.add_argument("--report", type=Path, help="write a JSON report")
    args = parser.parse_args()

    if not args.dataset and not args.root:
        parser.error("either --dataset or --root is required")

    preset = PRESETS.get(args.dataset, {}) if args.dataset else {}
    root = args.root or preset.get("root")
    if not root or not Path(root).is_dir():
        print(f"dataset root not found: {root}")
        print("Set the RSCDETR_DATASETS environment variable or pass --root.")
        return 1
    root = Path(root)

    report = Report()
    print(f"checking {root}\n")

    if preset:
        ann_stats = check_annotations(root, preset["annotations"], report)
        img_stats = check_images(root, preset["images"], report)
        for directory, files in [
            (d, [p for p in (root / d).rglob("*") if p.suffix.lower() in IMAGE_EXT])
            for d in preset["images"] if (root / d).is_dir()
        ]:
            check_pairing(files, preset["paired_suffixes"], report)
        if args.check_duplicates:
            dirs = {d: [p for p in (root / d).rglob("*") if p.suffix.lower() in IMAGE_EXT]
                    for d in preset["images"] if (root / d).is_dir()}
            check_duplicates(dirs, report)
    else:
        img_stats = check_images(root, ["."], report)
        ann_stats = {}

    for finding in report.findings:
        marker = "ERROR" if finding["level"] == "ERROR" else " warn"
        print(f"[{marker}] {finding['kind']}: {finding['detail']}")

    print(f"\n{report.errors} error(s), {report.warnings} warning(s)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "dataset": args.dataset,
            "root": str(root),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "annotations": ann_stats,
            "images": img_stats,
            "findings": report.findings,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")

    return 1 if report.errors else 0


if __name__ == "__main__":
    from PIL import Image  # imported here so --help works without Pillow
    raise SystemExit(main())
