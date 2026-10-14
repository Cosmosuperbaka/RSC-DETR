# Reproducibility status

Tracking every gap that stands between this repository and a fully reproducible
release. Status values:

| Symbol | Meaning |
|---|---|
| ✅ | artefact exists and is reachable |
| 🟡 | partially there — see the note |
| ❌ | missing, must be produced |
| 🔒 | blocked on compute, cannot be faked |

Last audited: **2026-10-12**, against the live experiment server.

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
| **Remaining official folds (2–10)** | ✅ (2026-10-14) | **All ten RSC-DETR folds trained and scored.** Folds 3,5,6,7,8,9,10 trained 2026-10-12/13 (fold02/04 re-evaluated from their checkpoints — their logs predate COCO logging); **RSC-DETR 10-fold: AP 56.42 ± 2.30** (fold01 59.31, fold08 59.03, fold05 53.94 …) — `evaluation/reports/multiseed_and_10fold_summary.json`. Baseline 10/10: AP 56.26 ± 2.00. rgb/ir comparison folds re-evaluated from checkpoints (20/20 rc=0). |
| **Multi-seed mean ± std** | ✅ (2026-10-14) | Seeds 3407 / 42 / 2024 all trained and scored: **VEDAI (fold-1 test) 57.72 ± 1.77**, **M3FD-LT20 (test) 55.42 ± 0.51**, **DVTOD (val) 55.45 ± 1.05** — `evaluation/reports/multiseed_and_10fold_summary.json`. |
| **Inference latency** | ✅ (2026-10-12) | Measured under `docs/BENCHMARK_PROTOCOL.md` (idle RTX 4090, FP32, batch 1, 1024×1024, warmup 50 / 200 timed): **median 35.30 ms (28.3 fps), peak 794 MiB** — `evaluation/reports/latency_vedai.json`. |
| **Software lock** | ✅ | `environment/env-lock.yaml` + `environment/pip-freeze-rtdetrv2-py310.txt` |
| **Re-evaluation archive for released weights** | ✅ (2026-10-12) | `tools/eval_released.py` re-scores a released checkpoint with a shipped config (same `DetSolver.val` path, EMA module) and writes a JSON report stamped with the SHA-256 of weights / config / annotations plus the software environment. Executed for the fold-1 headline checkpoint: **AP 59.311 / AP50 88.536 / AP75 71.467** (`evaluation/reports/vedai_fold01_reeval.json`) — matches the manuscript 59.31. |

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
| **Single authoritative AP (55.90 vs 55.95)** | ✅ (2026-10-12) | **Resolved.** The re-evaluation of the released checkpoint on the standard test split exists at `outputs/m3fd_lt20_s42_b8_valbest_test_requested/v19c_spsf/` (`per_class.json`, 6-class means **AP 55.897 / AP50 84.088 / AP75 60.507**) and matches the manuscript value **55.90, which is now frozen**. The 55.95 figure traces to a re-evaluation on a de-duplicated test variant and is not the manuscript number. |
| **Params / GFLOPs column** | ✅ (2026-10-12) | Measured at the protocol resolution 1024×1024: **90.10 M / 747.0 GFLOPs** — `evaluation/reports/complexity_m3fd.json`. (VEDAI and DVTOD re-measured in the same pass and match the manuscript: 90.11 M / 747.1 G and 90.10 M / 1485.5 G.) |
| **Multi-seed mean ± std** | ✅ (2026-10-14) | Seeds 42 / 3407 / 2024 trained; test AP **55.42 ± 0.51** — `evaluation/reports/multiseed_and_10fold_summary.json`. |
| **Latency and peak memory** | ✅ (2026-10-12) | Same protocol, 1024×1024: **median 37.70 ms (26.5 fps), peak ~0.8 GiB** — `evaluation/reports/latency_m3fd.json`. 🔒 was resolved on an idle card (GPU util 0%, 4 MiB in use). |
| **Software lock** | ✅ | see `environment/` |
| **De-duplicated test-set generator + manifest** | ❌ | Searched the server for `*dedup*` artefacts on 2026-10-12 — the generator script is gone and only stray GT-id lists survive (`outputs/compare_fill_20260802/m3fd_s42_*_gt*.json`). It must be rewritten from scratch if the de-duplicated variant is ever needed again; the frozen manuscript number does not depend on it. |

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
| **Runnable formal config** | ✅ (2026-10-12) | `configs/RSC-DETR_DVTOD.yml` is the canonical paper config, recovered from the run's own tensorboard config dump and renamed to the public symbols; the original `rtdod_1920_shdetr_c2_15e_b2_seed42.yml` was deleted on the server. Training used the retired `v19c` balance variant — the shipped config carries `final_harmonization`, which affects training-time loss routing only. |
| **Released weights** | ✅ (2026-10-10) | `weights/dvtod_rscdetr_s42_1920_eval.pth` (Git LFS), from `result/RTDOD_HBB_3class/shdetr_c2_1920_b2_15e_seed42/best.pth`; see `registry/experiments.yaml`. |
| **Training and evaluation logs** | ✅ | `result/RTDOD_HBB_3class/*.log` (e.g. `concat_seed42_1920x1080_45e.log`, 800 KB) |
| **Hashes for config / checkpoint / predictions** | ✅ (2026-10-12) | `registry/experiments.yaml` records SHA-256 for `configs/RSC-DETR_DVTOD.yml`, the LFS checkpoint copy and `perclass_sh_c2_best/predictions.json`. |
| **Explicit random seed** | 🟡 | Seed 42 is in every config filename and log, but not stated in the manuscript's experimental-setup section. |
| **Multi-seed mean ± std** | ✅ (2026-10-14) | Seeds 42 / 3407 / 2024 trained; val AP **55.45 ± 1.05** — `evaluation/reports/multiseed_and_10fold_summary.json`. |
| **Latency and peak memory** | ✅ (2026-10-12) | Same protocol, 1088×1920: **median 61.33 ms (16.3 fps), peak ~1.6 GiB** — `evaluation/reports/latency_dvtod.json`. |

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
| 1 | Resolve the M3FD AP discrepancy (55.90 vs 55.95) and freeze one number | **done** (2026-10-12) — re-evaluated from the released weights: 55.897 → frozen as 55.90 | — |
| 2 | Archive DVTOD config + weights + logs + hashes into the registry | **done** (2026-10-12) — `configs/RSC-DETR_DVTOD.yml` recovered from the run's config dump; registry filled | — |
| 3 | Run ≥3 seeds for all three datasets | GPU time | ~3x current training budget |
| 4 | Lock the software environment | **done** | — |
| 5 | Stand up benchmarking and regression testing | **done** (2026-10-12) — latency + complexity recorded on an idle RTX 4090 for all three datasets | — |

Steps 1, 2, 3 and 5 are **all complete** — multi-seed and the full 10-fold
sweep were trained and scored 2026-10-12/14. The only remaining structural
gap is the DVTOD independent test split, which requires re-scoring every
method against a split that does not exist yet. No bookkeeping debt is left.
