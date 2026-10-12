#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Params / GFLOPs / Latency for the DVTOD comparison methods.

Each adapter returns a callable ``run()`` that performs one *inference-mode*
forward pass at the resolution the method is actually trained / deployed with,
plus the parameter count.

FLOPs are counted with ``torch.utils.flop_counter.FlopCounterMode`` (matmul +
conv only), latency is the median of repeated timed forwards after warmup.

Usage:
    python measure_complexity.py --method c2dff  --weights <best.pt> --imgsz 1920
    python measure_complexity.py --method cmfadet --weights <best.pt> --imgsz 1920
    python measure_complexity.py --method sffr   --weights <best.pt> --imgsz 1920
    python measure_complexity.py --method rtdetr --config <cfg.yml> --weights <best.pth>
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


import torch


for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        sys.path.insert(0, str(_parent))
        break
from rscdetr_paths import ROOT  # noqa: E402


def human(n):
    return f"{n:,}"


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def _measure(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _flops(fn, inputs):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


# ---------------------------------------------------------------- YOLO (ultralytics)
def build_ultralytics(weights, imgsz, ch=6, repo=None):
    if repo:
        sys.path.insert(0, str(repo))
        os.chdir(repo)
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS["wandb"] = False

    model = YOLO(weights)
    net = model.model
    net.eval().cuda()
    x = torch.randn(1, ch, imgsz, imgsz, device="cuda")

    def run():
        with torch.no_grad():
            net(x)
    return net, run, _count_params(net)


# ---------------------------------------------------------------- SFFR / YOLOv5
def build_yolov5_dual(weights, imgsz,
                      repo=f"{ROOT}/compare/SFFR_VEDAI"):
    """Dual-stream YOLOv5 forks (SFFR, DARFNet): the DetectionModel takes (rgb, ir).

    ``attempt_load`` may return either the DetectionModel or an ``Ensemble``
    wrapper; both are callable as ``model(rgb, ir)`` in these forks (that is how
    test.py runs inference), so call the loaded object directly instead of
    unwrapping ``.model`` (which is a plain ``nn.Sequential``).
    """
    sys.path.insert(0, repo)
    os.chdir(repo)
    from models.experimental import attempt_load
    model = attempt_load(weights, map_location="cpu")
    try:
        model.eval()
    except AttributeError:
        model = model[0]
        model.eval()
    model = model.cuda()
    x1 = torch.randn(1, 3, imgsz, imgsz, device="cuda")
    x2 = torch.randn(1, 3, imgsz, imgsz, device="cuda")

    def run():
        with torch.no_grad():
            model(x1, x2)
    return model, run, _count_params(model)


# ---------------------------------------------------------------- RT-DETR
def build_rtdetr(config, weights, repo=f"{ROOT}/rtdetrv2_pytorch",
                 device="cuda:0", fallback_size=None):
    sys.path.insert(0, repo)
    os.chdir(repo)
    from src.core import YAMLConfig
    cfg = YAMLConfig(config)
    model = cfg.model.eval()
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("ema"), dict) and "module" in state["ema"]:
            params = state["ema"]["module"]
        elif isinstance(state, dict) and "model" in state:
            params = state["model"]
        else:
            params = state
        model.load_state_dict(params, strict=False)
    model = model.to(device)
    # configs with a fixed eval size (VEDAI, DVTOD) use it; native-resolution
    # configs (M3FD, eval_spatial_size: null) fall back to the protocol size
    size = cfg.yaml_cfg.get("eval_spatial_size") or fallback_size or [640, 640]
    h, w = int(size[0]), int(size[1])
    samples = {
        "rgb": torch.randn(1, 3, h, w, device=device),
        "ir": torch.randn(1, 3, h, w, device=device),
    }

    def run():
        with torch.no_grad():
            model(samples)
    return model, run, _count_params(model), (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["c2dff", "cmfadet", "sffr", "darfnet",
                             "c2former", "rscdetr", "concat"])
    ap.add_argument("--weights", default="")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--config", default="")
    ap.add_argument("--repo", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    if args.method in ("c2dff", "cmfadet"):
        net, run, n_params = build_ultralytics(
            args.weights, args.imgsz, ch=6, repo=args.repo or None)
        res = [args.imgsz, args.imgsz]
    elif args.method in ("sffr", "darfnet"):
        default_repo = (f"{ROOT}/compare/SFFR_VEDAI"
                        if args.method == "sffr"
                        else f"{ROOT}/compare/DARFNet_VEDAI")
        net, run, n_params = build_yolov5_dual(
            args.weights, args.imgsz, repo=args.repo or default_repo)
        res = [args.imgsz, args.imgsz]
    else:
        net, run, n_params, res = build_rtdetr(
            args.config, args.weights, repo=args.repo or str(ROOT),
            fallback_size=[args.imgsz, args.imgsz])
        run()

    gflops = _flops(run, None) / 1e9
    lat = _measure(run)
    result = {
        "method": args.method,
        "weights": args.weights,
        "resolution": list(res),
        "params": int(n_params),
        "params_M": round(n_params / 1e6, 3),
        "gflops": round(gflops, 2),
        "latency_ms": round(lat, 2),
        "fps": round(1000.0 / lat, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
