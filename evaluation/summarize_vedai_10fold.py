#!/usr/bin/env python3
"""Mean ± std across the VEDAI 10-fold runs.

Each fold directory holds a ``log.txt`` with one JSON object per line. The
per-epoch held-out metrics are stored in ``val_coco_eval_bbox`` for the fold
configs (their ``val_dataloader`` points at the fold's test annotation); older
logs may use ``test_coco_eval_bbox``. Both keys are accepted.

The per-fold headline numbers are taken at the best epoch by AP — the same
criterion that produced ``best.pth``. Folds that were never run to completion
are reported as missing instead of being silently skipped.

The grouped RSC-DETR run (``v19c-spsf-s3407-10fold``) contains folds 2-10;
fold 1 is the separate headline run. Pass it with ``--fold1-run``.

Usage (on the experiment server, with the results tree in place):

    python evaluation/summarize_vedai_10fold.py \
        --results-root ~/rsc-detr/results/VEDAI \
        --fold1-run v19c-spsf-s3407=~/re-detr-last/result/VEDAI/test-x1x3-v19c-spsf_seed3407 \
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
METHOD_ALIASES = {
    "v19c-spsf-s3407": "rscdetr_v19c_spsf",
}
MAIN_METHODS = ("baseline", "rscdetr_v19c_spsf")


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
            vec = row.get("val_coco_eval_bbox") or row.get("test_coco_eval_bbox")
            if not vec:
                continue
            if best is None or vec[0] > best[1][0]:
                best = (row.get("epoch"), vec)
    return best


def fold_from_name(name: str):
    match = re.search(r"fold(\d+)", name, re.I)
    return int(match.group(1)) if match else None


def collect(results_root: Path, extra_runs: dict[str, Path]):
    methods = {}
    for group_dir in sorted(results_root.glob("*-10fold")):
        method = group_dir.name.replace("-10fold", "")
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            # Skip interrupted/backup copies; they share the same fold number.
            lowered = run_dir.name.lower()
            if any(marker in lowered for marker in ("partial", "backup", "smoke", "interrupted")):
                continue
            log = run_dir / "log.txt"
            if not log.is_file():
                continue
            fold = fold_from_name(run_dir.name)
            if fold is None:
                continue
            best = best_epoch(log)
            methods.setdefault(method, {})[fold] = (
                None if best is None else {
                    "run": str(run_dir),
                    "epoch": best[0],
                    **{k: round(100.0 * v, 3) for k, v in zip(COCO_KEYS, best[1])},
                }
            )
    for method, run_dir in extra_runs.items():
        log = run_dir / "log.txt"
        if not log.is_file():
            raise SystemExit(f"missing log for --fold1-run {method}: {log}")
        best = best_epoch(log)
        if best is None:
            raise SystemExit(f"no COCO metrics in {log}")
        methods.setdefault(method, {})[1] = {
            "run": str(run_dir),
            "epoch": best[0],
            **{k: round(100.0 * v, 3) for k, v in zip(COCO_KEYS, best[1])},
        }
    return methods


def summarize(folds: dict[int, dict | None]):
    done = {fold: value for fold, value in folds.items() if value}
    summary = {
        "per_fold": {str(fold): value for fold, value in sorted(done.items())},
        "mean": {},
        "std": {},
        "n": len(done),
        "missing_folds": sorted(set(range(1, 11)) - set(done)),
    }
    if len(done) < 2:
        return summary
    for key in ("AP", "AP50", "AP75"):
        values = [done[fold][key] for fold in sorted(done)]
        summary["mean"][key] = round(statistics.mean(values), 3)
        summary["std"][key] = round(statistics.stdev(values), 3)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path,
                        help="e.g. ~/rsc-detr/results/VEDAI")
    parser.add_argument("--fold1-run", action="append", default=[],
                        metavar="METHOD=PATH",
                        help="inject a separate fold-1 run for METHOD")
    parser.add_argument("--out", type=Path, default=None,
                        help="write a JSON report here")
    parser.add_argument("--all-methods", action="store_true",
                        help="include rtdetr-rgb / rtdetr-ir and other partial groups")
    args = parser.parse_args()

    extra = {}
    for spec in args.fold1_run:
        if "=" not in spec:
            parser.error(f"--fold1-run must be METHOD=PATH, got {spec!r}")
        method, path = spec.split("=", 1)
        extra[method] = Path(path).expanduser()

    raw = collect(args.results_root, extra)
    report = {
        "date": datetime.now(timezone.utc).isoformat(),
        "results_root": str(args.results_root),
        "protocol": ("Official VEDAI folds 1-10. Each fold's val_dataloader points "
                     "at its held-out test annotation; AP is the best-epoch value. "
                     "Fold 1 may come from a separate headline run."),
        "methods": {},
    }
    for method, folds in raw.items():
        alias = METHOD_ALIASES.get(method, method)
        if not args.all_methods and alias not in MAIN_METHODS:
            continue
        report["methods"][alias] = summarize(folds)

    if "baseline" in report["methods"] and "rscdetr_v19c_spsf" in report["methods"]:
        base = report["methods"]["baseline"]["per_fold"]
        ours = report["methods"]["rscdetr_v19c_spsf"]["per_fold"]
        common = sorted(set(base) & set(ours), key=int)
        if len(common) >= 2:
            diffs = [ours[fold]["AP"] - base[fold]["AP"] for fold in common]
            report["paired_difference"] = {
                "rscdetr_minus_baseline_AP": {
                    "per_fold": {fold: round(ours[fold]["AP"] - base[fold]["AP"], 3)
                                 for fold in common},
                    "mean": round(statistics.mean(diffs), 3),
                    "std": round(statistics.stdev(diffs), 3),
                    "rscdetr_wins": sum(1 for value in diffs if value > 0),
                }
            }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written -> {args.out}")
    print(json.dumps(report["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
