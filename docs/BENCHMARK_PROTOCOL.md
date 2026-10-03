# Timing protocol

A latency number is only meaningful alongside the exact conditions that produced
it. This file fixes those conditions so that numbers from different runs — and
different people — are comparable.

**Every latency figure in the manuscript must cite this document.**

---

## Why a protocol at all

The obvious mistakes, all of which produce numbers that look plausible:

| Mistake | Effect |
|---|---|
| Timing the first forward pass | Includes cuDNN autotuning; inflated 2–5x |
| No `torch.cuda.synchronize()` | Measures kernel *launch*, not execution; deflated by an order of magnitude |
| Comparing FP32 against AMP-halved runs | Mixed precision is roughly 1.5–2x faster; the comparison becomes meaningless |
| Measuring on a busy GPU | The host has **8 shared GPUs**; contention alone can double the time |
| Reporting a single pass | ±10% run-to-run noise is normal even on a quiet card |
| Forgetting the pre/post-processing | End-to-end latency includes decode, not just backbone + head |

---

## Fixed conditions

These are not suggestions. A number produced with a different setting is not
comparable and must not be put in the same table.

| Parameter | Value | Rationale |
|---|---|---|
| GPU | single **RTX 4090**, idle | all eight identical, but only one is timed |
| GPU selection | `CUDA_VISIBLE_DEVICES` pins one device | prevents accidental multi-GPU sharding |
| Precision | **FP32** for the headline table; AMP reported separately, clearly labelled | the two are not interchangeable |
| Batch size | 1 for per-image latency; also report the training batch size | readers care about both |
| Input resolution | 1024x1024 (VEDAI, M3FD) and 1088x1920 (DVTOD) | matches training |
| Warm-up | **50** iterations, discarded | covers cuDNN autotune and clock ramp |
| Timed iterations | **200** | enough for a stable median |
| Synchronisation | `torch.cuda.synchronize()` before and after each measured block | otherwise you time the queue, not the work |
| Clock state | default (no `nvidia-smi -lgc`); record the observed clock | pinning changes the answer and must be declared |
| Reported statistic | **median**, with p10/p90 | median resists the occasional slow pass |
| Memory | `torch.cuda.max_memory_allocated()` after reset, on its own run | peak alloc overlaps with the previous model otherwise |

## Procedure

```bash
# 1. confirm the card is actually idle
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 2. pin one quiet GPU and run the benchmark
CUDA_VISIBLE_DEVICES=3 python3 benchmark/benchmark_latency.py \
    --config <path/to/config.yml> \
    --weights <path/to/best.pth> \
    --resolution 1024 1024 \
    --warmup 50 --iters 200 \
    --out results/latency/<name>.json

# 3. repeat on a different idle card; the two medians should agree within 5%
CUDA_VISIBLE_DEVICES=4 python3 benchmark/benchmark_latency.py ... --out results/latency/<name>_rep.json
```

If the two repetitions disagree by more than 5%, the machine was not quiet.
Discard both and start over on a confirmed-idle card.

## What gets recorded

The script writes a JSON blob that carries the conditions *with* the number, so
a figure can never be separated from its context:

```json
{
  "model": "rscdetr",
  "gpu": {"name": "NVIDIA GeForce RTX 4090", "index": 3,
          "driver": "590.48.01", "util_before_pct": 0},
  "precision": "fp32",
  "batch_size": 1,
  "resolution": [1024, 1024],
  "warmup_iters": 50,
  "timed_iters": 200,
  "latency_ms": {"median": 0.0, "p10": 0.0, "p90": 0.0, "mean": 0.0},
  "throughput_fps": 0.0,
  "peak_memory_mib": 0.0,
  "torch": "2.1.2+cu121",
  "cudnn": 8902,
  "timestamp": "2026-10-03T00:00:00"
}
```

## Reporting in the manuscript

State, at minimum: **GPU model, precision, batch size, resolution**. A table
cell like `12.3 ms` with no conditions attached is not acceptable; put the
conditions in the caption.

Params and GFLOPs come from `evaluation/measure_complexity.py`, which counts
them without running the model. Latency comes from this protocol. Do not mix
the two sources in one column without saying so.
