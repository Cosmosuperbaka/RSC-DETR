#!/usr/bin/env python3
"""Mean ± std across the VEDAI 10-fold runs for every method.

Closes the gap flagged in docs/REPRODUCIBILITY.md ("Remaining official folds
2-10"): a single table that reports mean ± std across the ten folds.

Each fold directory holds a ``log.txt`` with one JSON object per line; the
``test_coco_eval_bbox`` field carries the COCO [AP, AP50, AP75, ...] vector
for that epoch's validation pass. The per-fold headline numbers are taken at
the best epoch by AP — the same criterion that produced ``best.pth``.

Methods are discovered as ``<results-root>/*-10fold/<run-dir>/log.txt``;
fold indices are parsed from the run-directory name. Folds that were never
run to completion are reported as missing instead of being silently skipped.

Usage (on the experiment server, with the results tree in place):
    python evaluation/summarize_vedai_10fold.py \
        --results-root ~/rsc-detr/results/VEDAI \
        --out evaluation/reports/vedai_10fold_summary.json
"""

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

COCO_KEYS = ["AP", "AP50", "AP75", "APs", "APm", "APl",
             "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl"]


def best_epoch(log_path: Path):
    """Return (epoch, coco_vector) at the best AP, or None if the log is empty."""
    best = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vec = row.get("test_coco_eval_bbox")
            if not vec:
                continue
            if best is None or vec[0] > best[1][0]:
                best = (row.get("epoch"), vec)
    return best


def collect(results_root: Path):
    methods = {}
    for group_dir in sorted(results_root.glob("*-10fold")):
        method = group_dir.name.replace("-10fold", "")
        for run_dir in sorted(group_dir.iterdir()):
            log = run_dir / "log.txt"
            if not log.is_file():
                continue
            m = re.search(r"fold(\d+)", run_dir.name, re.I)
            fold = int(m.group(1)) if m else None
            best = best_epoch(log)
            if best is None:
                methods.setdefault(method, {})[fold] = None
                continue
            methods.setdefault(method, {})[fold] = {
                "run": str(run_dir),
                "epoch": best[0],
                **{k: round(100.0 * v, 3) for k, v in zip(COCO_KEYS, best[1])},
            }
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path,
                        help="e.g. ~/rsc-detr/results/VEDAI")
    parser.add_argument("--out", type=Path, default=None,
                        help="write a JSON report here")
    args = parser.parse_args()

    methods = collect(args.results_root)
    print(f"# VEDAI 10-fold summary — best-epoch COCO metrics per fold\n")

    for method, folds in methods.items():
        done = {f: v for f, v in folds.items() if v}
        print(f"## {method}  ({len(done)}/10 folds complete)\n")
        print("| fold | epoch | AP | AP50 | AP75 |")
        print("|---|---|---|---|---|")
        for fold in sorted(folds):
            v = folds[fold]
            if v is None:
                print(f"| {fold} | — | missing | missing | missing |")
            else:
                print(f"| {fold} | {v['epoch']} | {v['AP']} | {v['AP50']} | {v['AP75']} |")
        if len(done) >= 2:
            for key in ("AP", "AP50", "AP75"):
                vals = [done[f][key] for f in sorted(done)]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"\n{key}: mean {mean:.2f} ± {std:.2f} (n={len(vals)})")
        else:
            print("\nmean ± std: not computable yet (needs ≥2 completed folds)")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "date": datetime.now(timezone.utc).isoformat(),
            "results_root": str(args.results_root),
            "methods": methods,
        }
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"report written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
