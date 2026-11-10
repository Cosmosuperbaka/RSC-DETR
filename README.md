# RSC-DETR

Reference code for the paper

> **RSC-DETR: Symmetric Harmonized Dual-Stream DETR for Long-Tailed RGB-IR
> Object Detection**

RSC-DETR is a dual-stream RGB–IR detector for aerial remote sensing. It keeps the
RT-DETR inference pathway unchanged and adds three training-side components plus
a harmonizer that coordinates them.

| Component | Role | Stage |
|---|---|---|
| **SPSF** — Shared–Private Symmetric Fusion | Models a shared semantic centre and symmetric modality-specific deviations at the `S5` feature level, reducing redundant cross-modal responses before the encoder. | Feature fusion |
| **PFHM** — Prototype-guided Fine-localization Harmonization | Transfers the localization quality of matched queries into prototype-based classification supervision through IoU-aware soft targets. | Training supervision |
| **ACRHM** — Annotation-guided Class-reliability Harmonization | Combines training-annotation priors with detached online matching states to estimate class reliability and regulate class-wise supervision under long-tailed distributions. | Training supervision |
| **Harmonizer** | Derives gates and a bounded class-wise supervision budget from training progress, prototype maturity, localization quality, assignment stability, and annotation evidence, and routes the total loss between the native and budget forms. | Training control |

Inference uses only the two-stream backbones, SPSF, the encoder, the decoder and
the main detection heads — no auxiliary branch is added at test time.

Reported results (AP, %):

| Dataset | Backbone-scale setting | AP | AP₅₀ | AP₇₅ | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| VEDAI | 1024 | **59.31** | 88.53 | 71.47 | 89.6 | 748 |
| M3FD-LT20 | 1024 | **55.90** | 84.09 | 60.51 | — | — |
| DVTOD | 1920 | **56.66** | 89.49 | 61.21 | 90.10 | 1485.5 |

The single-run RT-DETR concat baseline is improved by +1.78 / +0.88 / +0.40 AP
points on the three datasets respectively.

**Protocol variability (recorded in this repository).** The headline AP of
59.31 is a single fold-1 run, and 55.90 is a single training seed; neither is a
cross-fold or cross-seed estimate. The complete runs give:

| Dataset | Protocol | AP |
|---|---|---:|
| VEDAI | official fold 1, training seed 3407 (headline) | 59.31 |
| VEDAI | all 10 official folds, training seed 3407 | **56.42 ± 2.30** |
| VEDAI | fold 1, training seeds 42 / 3407 / 2024 | 57.72 ± 1.77 |
| VEDAI | RT-DETR concat, all 10 official folds | 56.26 ± 2.00 |
| M3FD-LT20 | final 419-pair test, training seed 42 (headline) | 55.90 |
| M3FD-LT20 | final 419-pair test, training seeds 42 / 3407 / 2024 | **55.62 ± 0.46** |
| M3FD-LT20 | RT-DETR concat, same three training seeds | 55.30 ± 0.41 |
| DVTOD | validation, training seed 42 (headline) | 56.66 |
| DVTOD | validation, training seeds 42 / 3407 / 2024 | 55.45 ± 1.05 |

The ten-fold result is the more robust VEDAI estimate. RSC-DETR improves on 7
of 10 folds over the concat baseline, with a mean paired gain of +0.16 AP and a
paired SD of 1.57 (paired t ≈ 0.31, n = 10 — the sweep shows no statistically
detectable mean gain); the +2.60 AP fold-1 gain is the largest of the ten. The
baseline row in the 10-fold summary is a separate repeat of the concat config
(fold-1 AP 56.71), not the main-table concat run (57.53); the paired ten-fold
comparison is therefore a run-to-run variability study, not a re-use of the
main-table baseline. On M3FD-LT20 the three-seed gain is +0.32 AP, smaller than
the single-seed +0.88 AP headline.
The underlying per-fold and per-seed records are in
[`evaluation/reports/multiseed_and_10fold_summary.json`](evaluation/reports/multiseed_and_10fold_summary.json)
and [`evaluation/reports/vedai_10fold_summary.json`](evaluation/reports/vedai_10fold_summary.json).
These are training-seed repeats on a fixed data split; the split seed is a
different quantity (see below).

The ten-fold sweep covers RSC-DETR and the concat baseline. The remaining
comparison methods are reported on fold 1 only; the per-fold table must not be
read as a full cross-method benchmark.

Scope of what is verifiable from this repository: all **ten** official folds
(1-10) were trained and scored, each with training seed 3407; a description of
this sweep as covering only a subset of folds (for example the odd-numbered
ones) would be incorrect. What is shipped here is the *parsed* summary:
`evaluation/summarize_vedai_10fold.py` reads the per-fold run logs on the
experiment server
(`results/VEDAI/v19c-spsf-s3407-10fold/fold02..fold10/log.txt`), and only the
fold-1 and fold-2 checkpoints are released through Git LFS. The fold-3-10
numbers can therefore be recomputed only with server access to those logs; they
are recorded in the reports and registry, not reproducible from the public tree.

Post-hoc measurements recorded in this repository (FP32, batch 1, idle RTX 4090,
protocol in [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)) — see
[evaluation/reports/](evaluation/reports/):

| Dataset | Resolution | Params (M) | GFLOPs | Latency median |
|---|---|---|---|---|
| VEDAI | 1024×1024 | 90.11 | 747.1 | 35.3 ms |
| M3FD-LT20 | 1024×1024 | 90.10 | 747.0 | 37.7 ms |
| DVTOD | 1088×1920 | 90.10 | 1485.5 | 61.3 ms |

---

## Repository layout

This repository contains the **complete model implementation** of RSC-DETR
(`src/`, `configs/`, the `tools/train.py` entry point) together with the
**figure-rendering, evaluation and reproducibility tooling** for the paper.
Datasets are not redistributed — download links and preparation steps live in
[datasets/README.md](datasets/README.md). The released model weights are
tracked through **Git LFS** under `weights/` (see below). Experiment paths on
the server are referenced through [`rscdetr_paths.py`](rscdetr_paths.py) rather
than hard-coded paths.

```
RSC-DETR/
├── src/                                # model, losses, data, solvers
│   ├── core/                           # config / workspace plumbing
│   ├── data/                           # dual-stream dataloaders & transforms
│   ├── nn/                             # backbones (PResNet) & common blocks
│   ├── optim/                          # optimizer / EMA / AMP / warmup
│   ├── solver/                         # train & evaluation engines
│   └── zoo/rtdetr/                     # RSC-DETR, SPSF, PFHM, ACRHM, harmonizer
├── configs/
│   ├── RSC-DETR_VEDAI.yml              # final VEDAI config (1024, fold-1)
│   ├── RSC-DETR_M3FD-LT20.yml          # final M3FD-LT20 config (b8, 45e)
│   └── RSC-DETR_DVTOD.yml              # final DVTOD config (1088x1920, seed 42)
├── weights/                            # released checkpoints (Git LFS, *.pth)
├── datasets/
│   ├── README.md                       # dataset download & preparation guide
│   └── M3FD/reproducibility/           # fixed M3FD-LT20 lists, manifest, counts
├── scripts/m3fd/                       # split construction, COCO conversion, verification
├── tools/
│   ├── train.py                        # training / evaluation entry point
│   ├── collect_env.py                  # re-capture the environment lock
│   ├── check_dataset.py                # dataset integrity gate
│   ├── eval_released.py                # re-score a released checkpoint, write a hashed report
│   └── hash_artifacts.py               # SHA-256 manifests for run directories
├── rscdetr_paths.py                    # every experiment path, env-var driven
├── figures/
│   ├── render_fig1_motivation.py       # Fig. 1  (cross-modal confidence gap)
│   ├── render_vedai_qualitative.py     # Fig. 7  (VEDAI qualitative comparison)
│   ├── render_m3fd_qualitative.py      # Fig. 8  (M3FD-LT20 qualitative comparison)
│   └── render_dvtod_qualitative.py     # Fig. 9  (DVTOD qualitative comparison)
├── evaluation/
│   ├── eval_coco_preds.py              # COCO AP / AP50 / AP75 and per-class AP
│   ├── build_dvtod_table.py            # assembles the DVTOD comparison table
│   ├── measure_complexity.py           # Params / GFLOPs / latency
│   ├── summarize_yolo_compare_coco.py  # aggregates YOLO-family comparison runs
│   ├── summarize_vedai_10fold.py       # parses the per-fold run logs into the 10-fold summary
│   ├── predict_yolo_best.py            # runs inference from a YOLO best.pt
│   └── reports/                        # recorded re-evaluation, complexity, latency, multi-seed and 10-fold results
├── analysis/
│   ├── generate_combined_distribution.py  # class-distribution figure
│   ├── infer_scene000004_baseline.py
│   ├── prepare_dvtod_compare.py
│   ├── scan_vedai_yellow.py            # scene ranking for qualitative figures
│   ├── scan_m3fd_yellow.py
│   ├── scan_vedai_margin.py
│   └── export_vedai_1033_materials.py
├── environment/                        # the locked software stack
│   ├── env-lock.yaml                   # OS / interpreter / CUDA / cuDNN / GPU
│   ├── pip-freeze-rtdetrv2-py310.txt   # verbatim freeze, 122 packages
│   └── README.md
├── registry/
│   └── experiments.yaml                # run -> config -> weights -> log -> metric
├── benchmark/
│   └── benchmark_latency.py            # enforces docs/BENCHMARK_PROTOCOL.md
├── export/
│   └── export_model.py                 # ONNX + TorchScript, with provenance
├── tests/
│   └── test_smoke.py                   # pytest regression gate
├── docs/
│   ├── paper/main.tex                  # the manuscript source (figure image files not included)
│   ├── FINAL_VERSION_AUDIT.json        # source-file / class-name mapping record
│   ├── VALIDATION.json                 # packaging validation record
│   ├── PATHS.md                        # dataset / checkpoint / prediction layout
│   ├── REPRODUCIBILITY.md              # what exists, what is missing, in what order
│   ├── BENCHMARK_PROTOCOL.md           # the timing protocol — cite this
│   └── MULTISEED_PLAN.md               # multi-seed / 10-fold protocol and results
├── LICENSE / NOTICE / LICENSES/        # licensing and third-party notices
└── MANIFEST.json                       # SHA-256 manifest of the packaged model sources (src/, VEDAI and M3FD-LT20 configs, tools/train.py, licences)
```

**Start with [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** if you are
trying to reproduce a number. It tracks every artefact, says which ones are
still missing, and orders the remaining work by cost.

### About the three qualitative figures

All three figures share one visual language:

* one panel per method, **visible (RGB) on top and infrared (IR) below**;
* **red** = correct detection, **blue** = false detection,
  **green dashed** = ground truth,
  **yellow** = a true positive recovered only by RSC-DETR;
* every geometry constant is multiplied by `SCALE = 2`, so the exported PNG
  carries roughly twice the pixels of the earlier 240 px / 330 px / 400 px
  thumbnails and stays crisp at `\textwidth` in print.

Each renderer is standalone and only needs `Pillow` plus the paths defined at the
top of the file.

```bash
# on the experiment server, from any directory
python3 render_vedai_qualitative.py     # -> out_qual/fig7_vedai_1033_v3.png
python3 render_m3fd_qualitative.py      # -> out_qual/fig8_m3fd_00400_v3.png
python3 render_dvtod_qualitative.py     # -> out_qual/fig9_dvtod_1857_v3.png
```

The three scripts were validated against Pillow 9.0.1 (Python 3.10); they use
`Image.LANCZOS`, which works on both the 9.x and 10.x series.

---

## Environment

The reported results were produced with a **locked** stack — conda env
`rtdetrv2`, CPython 3.10.20, torch 2.1.2+cu121, cuDNN 8.9.2, on 8x RTX 4090
(Ubuntu 22.04.5). The full record lives in
[`environment/`](environment/README.md):

| File | Purpose |
|---|---|
| [`environment/env-lock.yaml`](environment/env-lock.yaml) | OS, interpreter, framework, CUDA/cuDNN, GPU and driver versions |
| [`environment/pip-freeze-rtdetrv2-py310.txt`](environment/pip-freeze-rtdetrv2-py310.txt) | verbatim `pip freeze`, 122 packages |

```bash
# quick path: top-level dependencies only
python3 -m pip install -r requirements.txt

# exact rebuild of the environment that produced the paper
conda create -n rscdetr python=3.10.20 -y && conda activate rscdetr
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r environment/pip-freeze-rtdetrv2-py310.txt
```

The `+cu121` wheel is not interchangeable with `+cu118`: it pulls a different
cuDNN, which changes both throughput and (slightly) kernel numerics. If you
change it, any latency figure from this repository becomes non-comparable, and
the lock has to be re-captured with `python3 tools/collect_env.py --write`.

For the figure renderers alone, `Pillow` is the only hard requirement.

## Training and evaluation

All commands run from the repository root. Three final configs are provided
(VEDAI, M3FD-LT20 and DVTOD; see the config audit trail in
[docs/FINAL_VERSION_AUDIT.json](docs/FINAL_VERSION_AUDIT.json)):

```bash
pip install -r requirements.txt          # install an CUDA-matched torch first

# training
python tools/train.py -c configs/RSC-DETR_VEDAI.yml -d cuda --use-amp --seed 3407
python tools/train.py -c configs/RSC-DETR_M3FD-LT20.yml -d cuda --use-amp --seed 42

# evaluation with a released checkpoint
python tools/train.py -c configs/RSC-DETR_VEDAI.yml -d cuda --test-only -r weights/<vedai>.pth
python tools/train.py -c configs/RSC-DETR_M3FD-LT20.yml -d cuda --test-only -r weights/<m3fd>.pth
python tools/train.py -c configs/RSC-DETR_DVTOD.yml -d cuda --test-only -r weights/dvtod_rscdetr_s42_1920_eval.pth
```

`--seed` is the **model-training seed** and is independent of the dataset
construction. In particular, all M3FD-LT20 runs use the same fixed split
defined by [`datasets/M3FD/reproducibility/splits/`](datasets/M3FD/reproducibility/splits/);
the directory name `lt20_seed42` refers to the **data-split sampling seed**, not
to `--seed`. Training seeds 42, 3407, and 2024 were run on that same split.
The VEDAI names `s3407` / `seed3407` are likewise training seeds on the
official fold-1 split.

Recipe summary — VEDAI: dual-stream PResNet-50, 1024 input, batch 4, 30 epochs,
LR decay at epoch 27. M3FD-LT20: dual-stream PResNet-50, native-resolution
input/padding, batch 8, 45 epochs, LR decay at epoch 36. DVTOD: dual-stream
PResNet-50, 1088×1920 input, batch 2, 15 epochs, LR decay at epoch 13. With
`pretrained: true` the ImageNet PResNet-50 backbone weights must be fetched
separately. Note: the DVTOD training run used a retired internal balance
variant; the shipped config carries the final harmonization strategy, which
affects training-time loss routing only — the released DVTOD checkpoint
reproduces the reported evaluation numbers under `--test-only`.

## Datasets

None of the three datasets is redistributed with this repository. Download them
from the official sources and lay them out as described in
[datasets/README.md](datasets/README.md):

| Dataset | Official source |
|---|---|
| VEDAI | <https://downloads.greyc.fr/vedai/> (GREYC, Razakarivony & Jurie 2016) |
| M3FD | <https://github.com/JinyuanLiu-CV/TarDAL> (TarDAL, Liu et al., CVPR 2022) |
| DVTOD | <https://github.com/VDT-2048/DVTOD> (Song et al., IEEE T-IV 2024) |

The COCO-format annotation files used by this project (VEDAI fold-1 / 8-class
HBB; M3FD-LT20 long-tailed split; DVTOD three-class train/validation) are
prepared from the official releases; the conversion rules are documented in the
dataset loader sources.

### M3FD-LT20 reproducibility package

The fixed split is released under
[`datasets/M3FD/reproducibility/`](datasets/M3FD/reproducibility/):

| Artefact | Purpose |
|---|---|
| `splits/train.txt` (1,428), `splits/val.txt` (420), `splits/test.txt` (419) | The exact image stems used in the paper |
| `splits/base/` | Pre-long-tail 3,359/420/420 split |
| `split_manifest.json` | Dataset version, counts, IF, data-split seeds, SHA-256, reconstruction status, dedupe record |
| `class_counts.csv`, `image_class_counts.csv` | Training distribution audit |
| `ir_deduplication_report.json` | Why test `00539` was removed |
| `verify_report.json` | Verification output |
| [`README.md`](datasets/M3FD/reproducibility/README.md) | Protocol and rebuild commands |

The construction scripts are in [`scripts/m3fd/`](scripts/m3fd/). The split is
**project-generated**, not an official M3FD split. RGB--IR pairs are the
sampling unit; multi-class images keep all annotations; class targets are
optimization references, not hard quotas. The final test set is 419 rather
than 420 because test `00539` has an IR file byte-identical to validation
`01300` and is removed with its 14 annotations. File-byte deduplication does
**not** prove scene-level isolation.

Only the split lists, summaries, and code are released. The per-box COCO JSONs
are regenerated locally from licensed M3FD data; their SHA-256 values are
recorded in the manifest and README.

## Model weights (Git LFS)

The released checkpoints are tracked with **Git LFS** in `weights/`:

```bash
git lfs install          # once per machine, before the first clone
git clone https://github.com/Cosmosuperbaka/RSC-DETR.git
cd RSC-DETR && git lfs pull   # fetch the weights if they were skipped
```

The mapping between each weight file, its config, seed and evaluation artefacts
is indexed in [`registry/experiments.yaml`](registry/experiments.yaml); hashes
are recorded with `tools/hash_artifacts.py`.

## Data and checkpoints

Nothing in this repository downloads data automatically, and **no experiment
path is hard-coded**. Every location is resolved at import time by
[`rscdetr_paths.py`](rscdetr_paths.py), which reads environment variables and
falls back to defaults derived from the current user's home directory:

| Variable | Meaning | Default |
|---|---|---|
| `RSCDETR_WORKSPACE` | Root holding all experiment trees | `~` |
| `RSCDETR_ROOT` | Main RSC-DETR working repository | `$RSCDETR_WORKSPACE/rsc-detr` |
| `RSCDETR_DATASETS` | Dataset root containing `VEDAI/`, `M3FD/` | `$RSCDETR_WORKSPACE/vedai_data/datasets` |
| `RSCDETR_CFT` | CFT comparison checkout | `$RSCDETR_WORKSPACE/CFT` |
| `RSCDETR_LCAFNET` | LCAFNet comparison checkout | `$RSCDETR_WORKSPACE/LCAFNet` |
| `RSCDETR_MSOD` | Multispectral-object-detection checkout | `$RSCDETR_WORKSPACE/multispectral-object-detection` |
| `RSCDETR_PAPER` | Unpacked paper sources (figure target) | `$RSCDETR_ROOT/RSC_DETR` |
| `RSCDETR_OUT_QUAL` | Qualitative-figure output root | `$RSCDETR_ROOT/out_qual` |

Print the resolved values before running anything, and export the ones your
machine needs:

```bash
python3 rscdetr_paths.py                 # show every resolved path
export RSCDETR_ROOT=/data/shd/detr       # override a single location
export RSCDETR_WORKSPACE=/mnt/experiments
```

Inside the repository the experiment trees are referenced as
`ROOT/datasets/` (DVTOD, RTDOD), `ROOT/compare/` (comparison methods),
`ROOT/outputs/` (RSC-DETR / RT-DETR predictions) and `ROOT/result/`
(RTDOD predictions). Scripts are portable as long as these trees exist
somewhere and the variables point at them.

See [docs/PATHS.md](docs/PATHS.md) for the full list, including which
`predictions.json` file feeds which panel of each figure.

## Reproducing the evaluation tables

```bash
# COCO-style metrics from a predictions.json
python3 evaluation/eval_coco_preds.py --help

# complexity (params / GFLOPs / latency)
python3 evaluation/measure_complexity.py --help

# assemble the DVTOD comparison table
python3 evaluation/build_dvtod_table.py

# re-derive the VEDAI 10-fold means from the run logs
python3 evaluation/summarize_vedai_10fold.py --help

# verify the released M3FD-LT20 split against locally generated COCO JSON
python3 scripts/m3fd/verify_lt20_final.py --help
```

Thresholds used by the qualitative renderers are a **confidence threshold of
0.50** (0.70 for the VEDAI panel, matching the released VEDAI comparison
predictions) and a **same-class IoU threshold of 0.50**.

## Notes on reproducibility

**For the full picture, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).**
It lists every artefact behind every reported number, flags the ones that are
still missing, and orders the remaining work. Short version:

| Area | State |
|---|---|
| Dataset paths, configs, weights, logs | exist on the experiment server, indexed in [`registry/experiments.yaml`](registry/experiments.yaml) |
| Software environment | **locked** — `environment/` |
| Dataset integrity / regression gates | **in place** — `tools/check_dataset.py`, `tests/test_smoke.py` |
| Timing protocol | **defined** — [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md) |
| Multi-seed mean ± std | **complete** — VEDAI 57.72 ± 1.77, M3FD-LT20 55.62 ± 0.46, DVTOD 55.45 ± 1.05 (training-only repeats on fixed splits) |
| VEDAI 10-fold mean ± std | **complete** — 56.42 ± 2.30; concat baseline 56.26 ± 2.00 |
| M3FD-LT20 fixed split | **released** — [datasets/M3FD/reproducibility/](datasets/M3FD/reproducibility/) |
| M3FD-LT20 headline AP | **resolved** — 55.90 on the final 419-pair test set (batch-4 re-evaluation); the uniform batch-8 three-seed suite gives 55.95 for the same seed |
| DVTOD independent test split | **does not exist** — reported numbers are validation numbers; DVTOD multi-seed repeats are validation-only |

Specific caveats about the figures:

* The qualitative panels are selected illustrative scenes, not dataset-level
  recall measurements. The scripts import cached `predictions.json` /
  YOLO `labels/*.txt` outputs; re-running inference may change individual scores.
* Panels that draw a yellow box re-derive "recovered only by RSC-DETR" by matching
  detections to ground truth at IoU ≥ 0.50 and checking whether any other method
  matches the same target.
* Class-index conventions differ between the RT-DETR family (1-based COCO ids)
  and the YOLO-family runs (0-based). Each renderer normalises them locally.
* **Fig. 1 is generated from real inference output, not hand-typed numbers.**
  `figures/render_fig1_motivation.py` reads every score out of
  `outputs/vedai_scene000004_single_modality/*.json`. If you change that figure,
  keep it that way — a motivation figure whose numbers cannot be reproduced is
  worse than no figure.

## Checks you can run right now

```bash
python3 tools/check_dataset.py --dataset vedai     # dataset integrity gate
pytest tests/ -v                                   # repository + numeric regression
python3 tools/hash_artifacts.py --run <run_dir> --out manifests/<name>.json
python3 tools/hash_artifacts.py --verify manifests/<name>.json
python3 benchmark/benchmark_latency.py --help      # read the protocol first
```

## Citation

```bibtex
@article{chen2026rscdetr,
  title   = {{RSC-DETR}: Symmetric Harmonized Dual-Stream {DETR} for
             Long-Tailed {RGB-IR} Object Detection},
  author  = {Chen, Yi and Deng, Lingjun and Zhong, Chuen-Ho and Liu, Chang and Dong, Yanni},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## Acknowledgements

The numerical calculations in this work were carried out on the supercomputing
system of the Supercomputing Center of Wuhan University. This study was supported
by the National Natural Science Foundation of China under Grant U2541203.
