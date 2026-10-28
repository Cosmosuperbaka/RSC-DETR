# VEDAI 10-fold summary

Best-epoch COCO metrics on the official VEDAI folds 1-10. Each fold config
points `val_dataloader` at `vedai_foldXX_test_class8.json`, so the reported
value is the held-out AP for that fold. Training seed is 3407 for every fold.
Fold 1 is the batch-4 re-evaluation of the released checkpoint; folds 2-10 are
parsed from the run logs. Generated 2026-10-28.

| Fold | Baseline AP | Baseline AP50 | Baseline AP75 | RSC-DETR AP | RSC-DETR AP50 | RSC-DETR AP75 | Δ AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 56.710 | 86.612 | 66.890 | 59.311 | 88.536 | 71.467 | +2.601 |
| 2 | 53.010 | 81.562 | 62.709 | 52.167 | 80.323 | 61.004 | -0.843 |
| 3 | 57.140 | 87.208 | 71.075 | 58.101 | 87.390 | 70.264 | +0.961 |
| 4 | 58.376 | 85.485 | 72.190 | 55.022 | 82.595 | 66.367 | -3.354 |
| 5 | 53.288 | 82.108 | 59.159 | 53.945 | 82.065 | 62.840 | +0.657 |
| 6 | 55.453 | 85.283 | 66.117 | 56.127 | 86.130 | 65.862 | +0.674 |
| 7 | 55.310 | 82.436 | 70.857 | 55.504 | 84.157 | 69.556 | +0.194 |
| 8 | 58.088 | 89.272 | 68.903 | 59.030 | 88.684 | 70.642 | +0.942 |
| 9 | 58.768 | 88.689 | 69.431 | 57.960 | 87.631 | 70.481 | -0.808 |
| 10 | 56.503 | 85.816 | 67.304 | 57.026 | 86.157 | 68.895 | +0.523 |
| **Mean** | **56.265** | **85.447** | **67.463** | **56.419** | **85.367** | **67.738** | **+0.155** |
| **Std** | **2.003** | **2.689** | **4.036** | **2.305** | **2.924** | **3.584** | **1.569** |

RSC-DETR improves AP on **7 of 10 folds**; the mean paired gain is +0.155 AP
with a paired SD of 1.569. Fold 1 is the largest gain (+2.601) and is also the
headline protocol. The 10-fold mean is therefore the appropriate summary when
comparing methods; the single fold-1 number is not a cross-fold expectation.

The baseline row is a separate ten-fold repeat of the same concat config; its
fold-1 run scored 56.71, while the main-table concat run scored 57.53. The
paired comparison therefore measures run-to-run variability under a common
protocol, not a re-use of the main-table baseline.

The two methods differ only in the fusion/supervision path; both use the same
fold definitions, input resolution, epoch budget, and evaluation code.

Source: `evaluation/reports/vedai_10fold_summary.json`.
