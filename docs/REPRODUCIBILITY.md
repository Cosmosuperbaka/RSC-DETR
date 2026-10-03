# Reproducibility status

Tracking every gap that stands between this repository and a fully reproducible
release. Status values:

| Symbol | Meaning |
|---|---|
| ✅ | artefact exists and is reachable |
| 🟡 | partially there — see the note |
| ❌ | missing, must be produced |
| 🔒 | blocked on compute, cannot be faked |

Last audited: **2026-10-03**, against the live experiment server.

> This file is the contract. When you close a gap, update the row here in the
> same commit that produces the artefact.

---

## VEDAI

| Item | Status | Where it lives / what is missing |
|---|---|---|
| Fold-1 split definition | ✅ | `datasets/VEDAI/annotations/vedai_fold01_test_class8.json` |
| Training config | ✅ | `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_vedai_*.yml` |
| Released weights | ✅ | `results/VEDAI/<run>/` |
| AP / AP50 / AP75 | ✅ | manuscript Table; `evaluation/eval_coco_preds.py` recomputes |
| Params / GFLOPs | ✅ | `evaluation/measure_complexity.py` |
| Ablation results | ✅ | `results/VEDAI/` (no-spsf, no-x1-gate, no-x3-gate) |
| Qualitative figure | ✅ | `figures/render_vedai_qualitative.py` |
| **Remaining official folds (2–10)** | 🟡 | Baseline 10-fold **is** present (`results/VEDAI/baseline-10fold/baseline-fold1..10/`) and so are `rtdetr-rgb-10fold` / `rtdetr-ir-10fold` / `v19c-spsf-s3407-10fold`. What is still missing is a single table that reports **mean ± std across the ten folds** for every method. |
| **Multi-seed mean ± std** | ❌ | Only seed 3407 has been run to completion for the released config. Needs seeds 3407 / 42 / 2024 (see `docs/MULTISEED_PLAN.md`). 🔒 |
| **Inference latency** | ❌ | Never measured under a controlled protocol. Script is ready (`benchmark/benchmark_latency.py`); needs a quiet GPU. 🔒 |
| **Software lock** | ✅ | `environment/env-lock.yaml` + `environment/pip-freeze-rtdetrv2-py310.txt` |
| **Re-evaluation archive for released weights** | 🟡 | Per-method `predictions.json` exist under `outputs/vedai_*`. Missing: a single script that re-runs evaluation from the *released* weights and writes a hash-stamped report. |

---

## M3FD-LT20

| Item | Status | Where it lives / what is missing |
|---|---|---|
| train/val/test split | ✅ | `datasets/M3FD/processed/lt20_seed42/` |
| Training config | ✅ | `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m3fd_lt20_*.yml` |
| Released weights | ✅ | see `registry/experiments.yaml` |
| Weight provenance log | ✅ | run logs next to each checkpoint |
| Overall + per-class metrics | ✅ | manuscript Table |
| Scale and head/tail metrics | ✅ | manuscript Table |
| Qualitative figure | ✅ | `figures/render_m3fd_qualitative.py` |
| **Single authoritative AP (55.90 vs 55.95)** | ❌ | **Highest priority.** Two numbers circulate: `55.90` in the manuscript table, `55.95` in the weight-provenance log. Both trace to the same run directory. The 0.05 gap is almost certainly a re-evaluation on a de-duplicated test set that was never promoted to the manuscript. Must be resolved by re-evaluating the released checkpoint once and freezing the result. |
| **Params / GFLOPs column** | 🟡 | `evaluation/measure_complexity.py` supports M3FD but no measured row has been recorded. |
| **Multi-seed mean ± std** | ❌ | Same as VEDAI. 🔒 |
| **Latency and peak memory** | ❌ | Needs the benchmark protocol. 🔒 |
| **Software lock** | ✅ | see `environment/` |
| **De-duplicated test-set generator + manifest** | ❌ | A de-duplicated variant of the test set was used at some point; the script that produced it was never committed and no manifest was kept. Needs to be recovered or rewritten. |

---

## DVTOD

| Item | Status | Where it lives / what is missing |
|---|---|---|
| 1606 / 573 train/val image pairs | ✅ | `datasets/RTDOD_HBB_3class/` |
| Input size 1088×1920, 15 epochs, batch 2 | ✅ | manuscript §Experimental Setup |
| AMP disabled, 166-iteration warm-up, LR decay at epoch 13 | ✅ | manuscript §Experimental Setup |
| Overall / per-class / scale metrics | ✅ | manuscript Table |
| Params / GFLOPs | ✅ | manuscript Table |
| **Independent test split** | ❌ | Only train/val exist today. The reported numbers are validation numbers. A held-out test split has to be carved out and every method re-scored. 🔒 |
| **Runnable formal config** | 🟡 | Configs **do** exist — `configs/rtdetrv2/rtdetrv2_r50vd_rtdod_hbb_seed42_1920_{concat,shdetr}_{15e,20e}_scaled.yml`. What is missing is a statement of which one is the canonical paper config. |
| **Released weights** | 🟡 | Present under `result/RTDOD_HBB_3class/<run>/`, but not listed anywhere a reader can find them. |
| **Training and evaluation logs** | ✅ | `result/RTDOD_HBB_3class/*.log` (e.g. `concat_seed42_1920x1080_45e.log`, 800 KB) |
| **Hashes for config / checkpoint / predictions** | ❌ | Use `tools/hash_artifacts.py`; no manifest committed yet. |
| **Explicit random seed** | 🟡 | Seed 42 is in every config filename and log, but not stated in the manuscript's experimental-setup section. |
| **Multi-seed mean ± std** | ❌ | 🔒 |
| **Latency and peak memory** | ❌ | 🔒 |

---

## Cross-cutting

| Item | Status | Notes |
|---|---|---|
| **Full dependency lock** | ✅ | `environment/` — 122 packages, verbatim freeze |
| **Unified export script** | ✅ | `export/export_model.py` (ONNX + TorchScript) |
| **Automated dataset integrity check** | ✅ | `tools/check_dataset.py` |
| **Automated train/eval regression test** | ✅ | `tests/test_smoke.py` (pytest) |
| **Unified timing protocol** | ✅ | `docs/BENCHMARK_PROTOCOL.md` + `benchmark/benchmark_latency.py` |
| **Versioned experiment registry** | ✅ | `registry/experiments.yaml` — links data version, config, weights, logs, metrics |
| **Artifact hashing** | ✅ | `tools/hash_artifacts.py` |

---

## Suggested order of attack

Mirrors the priority list, with the current blocker called out for each step.

| # | Step | Blocker | Cost |
|---|---|---|---|
| 1 | Resolve the M3FD AP discrepancy (55.90 vs 55.95) and freeze one number | none — just re-evaluate the released checkpoint | ~1 GPU-hour |
| 2 | Archive DVTOD config + weights + logs + hashes into the registry | none | ~1 hour, no GPU |
| 3 | Run ≥3 seeds for all three datasets | GPU time | ~3x current training budget |
| 4 | Lock the software environment | **done** | — |
| 5 | Stand up benchmarking and regression testing | **done** (scripts); needs a quiet GPU to record the numbers | ~2 GPU-hours |

Steps 1 and 2 need no new code and no new GPU allocation — they are pure
bookkeeping on artefacts that already exist. Do them first.
