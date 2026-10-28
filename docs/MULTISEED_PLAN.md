# Multi-seed and 10-fold protocol

**Status: executed (2026-10-12/14).** Training-seed repeats and the full VEDAI
10-fold sweep are complete. The results below supersede the earlier
"not yet executed" status of this file. The plan text is retained as the
protocol description.

Data splits are fixed across seeds: VEDAI official fold 1 (and folds 1-10 for
the fold sweep), M3FD-LT20's fixed `lt20_seed42` split, and the DVTOD
train/validation split. Only the model-training seed changes.

## Results

| Dataset / protocol | Seeds | AP (mean ± std) | Comparison |
|---|---|---:|---|
| VEDAI fold-1 test | 42 / 3407 / 2024 | **57.72 ± 1.77** | headline seed 3407 = 59.31 |
| VEDAI all 10 official folds | 3407 | **56.42 ± 2.30** | concat baseline 56.26 ± 2.00 |
| M3FD-LT20 final 419 test | 42 / 3407 / 2024 | **55.62 ± 0.46** | concat 55.30 ± 0.41 |
| DVTOD validation | 42 / 3407 / 2024 | **55.45 ± 1.05** | no independent test split |

The 10-fold RSC-DETR values are 59.31, 52.17, 58.10, 55.02, 53.95, 56.13,
55.50, 59.03, 57.96, 57.03. RSC-DETR is better than the concat baseline on 7 of
10 folds; the mean paired gain is +0.16 AP (paired SD 1.57). The baseline row
is a separate ten-fold repeat of the concat config (fold-1 AP 56.71), not the
main-table concat run (57.53); the paired comparison measures run-to-run
variability under a common protocol. The fold-1 +2.60 gain is the largest of
the ten, which is why the headline 59.31 must not be read as a cross-fold
expectation.

The M3FD-LT20 three-seed values are 55.945 / 55.833 / 55.090 for seeds
42 / 3407 / 2024, measured with the same final test set, checkpoint-selection
rule, and batch size. The single-seed headline 55.90 comes from a batch-4
re-evaluation and differs from the batch-8 seed-42 value by 0.05 AP
(batching/padding numerics, not split membership).

Sources: `evaluation/reports/multiseed_and_10fold_summary.json`,
`evaluation/reports/vedai_10fold_summary.json`, per-run logs on the experiment
server. M3FD split details: `datasets/M3FD/reproducibility/README.md`.

---

## Original plan

The text below specifies the protocol. It is kept for provenance.

## Why this is required

A single-seed result cannot distinguish "our module helps" from "this seed was
lucky". Reviewers ask for it, and more importantly it is the honest thing to
report: DETR-family detectors vary by roughly 0.3–0.8 AP across seeds on sets of
this size, which is the same order as several of the ablation deltas.

The target is **mean ± standard deviation over three seeds**, reported for the
main results table and for every ablation row.

## Seeds

| Seed | Rationale |
|---|---|
| 3407 | already run — the current reported numbers |
| 42 | already run for some configs (DVTOD) |
| 2024 | new |

Three seeds is the minimum defensible number. If the standard deviation turns
out to be larger than the smallest ablation delta, the ablation needs more
seeds, not a stronger claim.

## Fixed across seeds

Everything except `seed` in the config. In particular:

- the data split (VEDAI fold 1; M3FD-LT20's fixed `lt20_seed42` split; DVTOD
  train/val)
- number of epochs, batch size, learning rate schedule, warm-up length
- the hyperparameters in `environment/env-lock.yaml`'s environment
- AMP off for DVTOD, as in the paper

If a config has the seed baked into the filename, create sibling files rather
than editing in place — the existing runs must stay addressable.

## What to run

| Dataset | Config family | Seeds | Approx. cost |
|---|---|---|---|
| VEDAI | `rtdetrv2_r50vd_vedai_1024_test-x1x3-v19c-spsf_seed*_fold01_30e.yml` | 3407, 42, 2024 | 3 x 30 epochs, 1024 px |
| VEDAI | concat baseline equivalent | 3407, 42, 2024 | 3 x 30 epochs |
| M3FD-LT20 | `rtdetrv2_r50vd_m3fd_lt20_seed*_x1x3_v19c_spsf_native_res_b8_45e.yml` | 3407, 42, 2024 | 3 x 45 epochs, 1024 px |
| M3FD-LT20 | concat baseline equivalent | 3407, 42, 2024 | 3 x 45 epochs |
| DVTOD | `rtdetrv2_r50vd_rtdod_hbb_seed*_1920_shdetr_15e_scaled.yml` | 42, 2024 (+3407 if available) | 3 x 15 epochs, 1088x1920 |

At minimum, the headline method and the strongest baseline must both be
multi-seed. Ablations can follow once the main table is stable.

## Submission

```bash
# one seed per GPU; the host has eight, so a dataset finishes in one wave
for seed in 3407 42 2024; do
  CUDA_VISIBLE_DEVICES=$GPU python3 rdetrv2_pytorch/tools/train.py \
      -c <config with that seed> --seed $seed \
      -t <output dir>/seed_$seed
done
```

Record every launch in `registry/experiments.yaml` with a distinct `id`
(`<dataset>-<method>-s<seed>`), then fingerprint the outputs with
`tools/hash_artifacts.py`.

## Reporting

Once all seeds are in, replace each single number in the main table with
`mean ± std`, and state the number of seeds in the caption. Where a method was
only run once, say so explicitly in the caption rather than leaving the reader
to assume the same protocol.

Also worth reporting alongside the table: the **spread between seeds** for the
baseline. If the baseline's own seed-to-seed variation approaches the claimed
improvement, that is a finding the paper should acknowledge, not hide.
