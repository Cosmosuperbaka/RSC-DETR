#!/usr/bin/env python3
"""Measure inference latency under the controlled protocol.

See docs/BENCHMARK_PROTOCOL.md -- this script exists to implement it, not to
invent its own conventions. Run it on an idle GPU and record the JSON it emits
alongside any number that ends up in a paper.

    CUDA_VISIBLE_DEVICES=3 python3 benchmark/benchmark_latency.py \
        --config <config.yml> \
        --weights <best.pth> \
        --resolution 1024 1024 \
        --out results/latency/vedai_rscdetr.json

Two sanity gates are enforced before anything is timed:

  * the card must be idle (utilisation and allocated memory both low);
  * the input resolution and precision must be stated explicitly, so a number
    can never be reported without its context.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import sys as _sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        _sys.path.insert(0, str(_parent))
        break

from rscdetr_paths import ROOT  # noqa: E402

IDLE_UTIL_MAX = 10      # percent
IDLE_MEM_MAX = 2048     # MiB


def gpu_state(index: int) -> dict:
    """Query utilisation and used memory for one physical card."""
    import subprocess

    query = "index,utilization.gpu,memory.used,clocks.sm"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and int(parts[0]) == index:
            return {
                "index": index,
                "util_before_pct": int(parts[1]),
                "memory_used_mib": int(parts[2]),
                "sm_clock_mhz": int(parts[3]),
            }
    return {}


def build_model(config_path: Path, weights_path: Path, device: str):
    """Instantiate the detector exactly as training does."""
    import torch
    from rdetrv2_pytorch.src.core import YAMLConfig, yaml_utils  # noqa: F401

    update = yaml_utils.parse_cli([])
    update.update({"resume": str(weights_path), "device": device})
    cfg = YAMLConfig(str(config_path), **update)
    model = cfg.model
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = state.get("ema", state).get("module", state.get("model", state))
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), cfg


def make_input(rgb_shape, device: str):
    import torch

    b, _, h, w = rgb_shape
    return {
        "rgb": torch.randn(b, 3, h, w, device=device),
        "ir": torch.randn(b, 3, h, w, device=device),
    }


def measure(model, samples, warmup: int, iters: int, device: str) -> dict:
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            model(samples)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        times: list[float] = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(samples)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

        peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    times.sort()
    return {
        "latency_ms": {
            "median": statistics.median(times),
            "mean": statistics.fmean(times),
            "p10": times[int(0.10 * len(times))],
            "p90": times[int(0.90 * len(times))],
            "min": times[0],
            "max": times[-1],
        },
        "throughput_fps": 1000.0 / statistics.median(times),
        "peak_memory_mib": peak,
        "n_timed": len(times),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--resolution", nargs=2, type=int, required=True,
                        metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--amp", action="store_true",
                        help="run under autocast; results are NOT comparable to fp32")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-idle-check", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available -- nothing to time.")
        return 1

    device = "cuda:0"          # CUDA_VISIBLE_DEVICES already selected the card
    index = int(torch.cuda.current_device())

    state = gpu_state(index)
    print(f"GPU {index}: util={state.get('util_before_pct')}% "
          f"mem={state.get('memory_used_mib')} MiB clock={state.get('sm_clock_mhz')} MHz")

    if not args.skip_idle_check:
        if state.get("util_before_pct", 0) > IDLE_UTIL_MAX or \
           state.get("memory_used_mib", 0) > IDLE_MEM_MAX:
            print("\nREFUSING TO MEASURE: the card is not idle.")
            print("The host is shared -- pick a quiet GPU and try again,")
            print("or pass --skip-idle-check if you know the contention is acceptable.")
            return 2

    config, weights = args.config.resolve(), args.weights.resolve()
    if not config.is_file() or not weights.is_file():
        print(f"missing input: config={config.is_file()} weights={weights.is_file()}")
        return 1

    model, _cfg = build_model(config, weights, device)

    h, w = args.resolution
    samples = make_input((args.batch_size, 3, h, w), device)

    precision = "amp" if args.amp else "fp32"
    print(f"model ready -- resolution={h}x{w} batch={args.batch_size} "
          f"precision={precision} warmup={args.warmup} iters={args.iters}")

    if args.amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            result = measure(model, samples, args.warmup, args.iters, device)
    else:
        result = measure(model, samples, args.warmup, args.iters, device)

    payload = {
        "model": config.stem,
        "weights": str(weights),
        "gpu": {
            "name": torch.cuda.get_device_name(index),
            "driver": state.get("driver"),
            "sm_clock_mhz": state.get("sm_clock_mhz"),
            **state,
        },
        "precision": precision,
        "batch_size": args.batch_size,
        "resolution": [h, w],
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **result,
    }

    payload["gpu"].pop("index", None)

    print("\n" + "=" * 58)
    print(f"  median   {result['latency_ms']['median']:.2f} ms  "
          f"({result['throughput_fps']:.1f} FPS)")
    print(f"  p10/p90  {result['latency_ms']['p10']:.2f} / "
          f"{result['latency_ms']['p90']:.2f} ms")
    print(f"  peak mem {result['peak_memory_mib']:.0f} MiB")
    print("=" * 58)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print("\n(pass --out to record this; a number without its conditions is useless)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
