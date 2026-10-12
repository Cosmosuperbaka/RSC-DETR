# Modified for the RSC-DETR release; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Re-evaluate a released checkpoint and write a hash-stamped report.

This is the missing piece flagged in docs/REPRODUCIBILITY.md ("Re-evaluation
archive for released weights"): one command that loads a *released* weight
file with a *shipped* config, re-runs the COCO evaluation, and records the
result together with the SHA-256 of every input, so the numbers in the paper
can be traced to byte-exact artefacts.

The evaluation follows the training code path exactly: the solver is built
from the config, the checkpoint is loaded with ``load_resume_state`` (the
same loader ``--test-only -r`` uses), and the EMA module — which is what
``DetSolver.val`` evaluates — is scored on the config's validation split.

Usage:
    python tools/eval_released.py -c configs/RSC-DETR_VEDAI.yml \
        -w weights/vedai_rscdetr_fold01_s3407_1024_eval.pth \
        --out evaluation/reports/vedai_fold01.json
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from src.misc import dist_utils  # noqa: E402
from src.core import YAMLConfig, yaml_utils  # noqa: E402
from src.solver import TASKS  # noqa: E402
from src.solver.det_engine import evaluate  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("-w", "--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="report JSON path")
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("--ann-file", type=str, default=None,
                        help="override the config's validation annotation file")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update = {}
    if args.device:
        update["device"] = args.device
    cfg = YAMLConfig(str(args.config), **update)

    if args.ann_file:
        # optional split override; kept explicit in the report
        cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = args.ann_file

    # BaseSolver.__init__ only stores the config; eval() performs the lazy
    # setup (model, EMA, criterion, dataloaders) and prefers test_dataloader
    # over val_dataloader when the config defines one — this mirrors what
    # `tools/train.py --test-only` executes.
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    solver.load_resume_state(str(args.weights))

    module = solver.ema.module if getattr(solver, "ema", None) else solver.model
    test_stats, _coco_evaluator = evaluate(
        module, solver.criterion, solver.postprocessor,
        solver.val_dataloader, solver.evaluator, solver.device,
    )
    # COCO 12-vector: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100,
    #                  ARs, ARm, ARl]
    vector = test_stats["coco_eval_bbox"]
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl",
            "AR@1", "AR@10", "AR@100", "ARs", "ARm", "ARl"]
    metrics = {k: (None if v is None else round(100.0 * v, 3))
               for k, v in zip(keys, vector)}

    ann_file = solver.val_dataloader.dataset.ann_file
    report = {
        "date": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "config_sha256": sha256_of(args.config.resolve()),
        "weights": str(args.weights),
        "weights_sha256": sha256_of(args.weights.resolve()),
        "ann_file": ann_file,
        "ann_file_sha256": sha256_of(Path(ann_file).resolve())
        if Path(ann_file).resolve().is_file() else None,
        "img_folder": cfg.yaml_cfg["val_dataloader"]["dataset"].get("img_folder"),
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "protocol": "DetSolver.val() path; EMA module evaluated; see "
                    "docs/BENCHMARK_PROTOCOL.md for timing (not used here)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report written -> {args.out}")
    dist_utils.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
