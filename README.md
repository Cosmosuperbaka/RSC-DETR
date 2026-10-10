# RSC-DETR

Reference code for the paper

> **Symmetric Fusion and Reliability-Aware Supervision for Long-Tailed RGB–IR
> Remote-Sensing Detection**

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
| VEDAI | 1024 | **59.31** | — | — | 89.6 | 748 |
| M3FD-LT20 | 1024 | **55.90** | — | — | — | — |
| DVTOD | 1920 | **56.66** | 89.49 | 61.21 | 90.10 | 1485.5 |

The RT-DETR concat baseline is improved by +1.78 / +0.88 / +0.40 AP points on the
three datasets respectively.

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
│   └── RSC-DETR_M3FD-LT20.yml          # final M3FD-LT20 config (b8, 45e)
├── weights/                            # released checkpoints (Git LFS, *.pth)
├── datasets/README.md                  # dataset download & preparation guide
├── tools/
│   ├── train.py                        # training / evaluation entry point
│   ├── collect_env.py                  # re-capture the environment lock
│   ├── check_dataset.py                # dataset integrity gate
│   └── hash_artifacts.py               # SHA-256 manifests for run directories
├── rscdetr_paths.py                    # every experiment path, env-var driven
├── figures/
│   ├── render_fig1_motivation.py       # Fig. 1  (cross-modal confidence gap)
│   ├── render_vedai_qualitative.py     # Fig. 7  (VEDAI qualitative comparison)
│   ├── render_m3fd_qualitative.py      # Fig. 8  (M3FD-LT20 qualitative comparison)
│   ├── render_dvtod_qualitative.py     # Fig. 9  (DVTOD qualitative comparison)
│   └── legacy/                         # earlier renderers, kept for provenance
├── evaluation/
│   ├── eval_coco_preds.py              # COCO AP / AP50 / AP75 and per-class AP
│   ├── build_dvtod_table.py            # assembles the DVTOD comparison table
│   ├── measure_complexity.py           # Params / GFLOPs / latency
│   ├── summarize_yolo_compare_coco.py  # aggregates YOLO-family comparison runs
│   └── predict_yolo_best.py            # runs inference from a YOLO best.pt
├── analysis/
│   ├── plot_error_composition.py       # error-composition breakdown
│   ├── plot_paper_analysis_candidates.py
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
│   ├── paper/main.tex                  # the manuscript source
│   ├── FINAL_VERSION_AUDIT.json        # source-file / class-name mapping record
│   ├── VALIDATION.json                 # packaging validation record
│   ├── PATHS.md                        # dataset / checkpoint / prediction layout
│   ├── REPRODUCIBILITY.md              # what exists, what is missing, in what order
│   ├── BENCHMARK_PROTOCOL.md           # the timing protocol — cite this
│   └── MULTISEED_PLAN.md               # the not-yet-run multi-seed job
├── LICENSE / NOTICE / LICENSES/        # licensing and third-party notices
└── MANIFEST.json                       # SHA-256 manifest of the packaged sources
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

All commands run from the repository root. Two final configs are provided
(VEDAI and M3FD-LT20; see the config audit trail in
[docs/FINAL_VERSION_AUDIT.json](docs/FINAL_VERSION_AUDIT.json)):

```bash
pip install -r requirements.txt          # install an CUDA-matched torch first

# training
python tools/train.py -c configs/RSC-DETR_VEDAI.yml -d cuda --use-amp --seed 3407
python tools/train.py -c configs/RSC-DETR_M3FD-LT20.yml -d cuda --use-amp --seed 42

# evaluation with a released checkpoint
python tools/train.py -c configs/RSC-DETR_VEDAI.yml -d cuda --test-only -r weights/<vedai>.pth
python tools/train.py -c configs/RSC-DETR_M3FD-LT20.yml -d cuda --test-only -r weights/<m3fd>.pth
```

Recipe summary — VEDAI: dual-stream PResNet-50, 1024 input, batch 4, 30 epochs,
LR decay at epoch 27. M3FD-LT20: dual-stream PResNet-50, native-resolution
input/padding, batch 8, 45 epochs, LR decay at epoch 36. With
`pretrained: true` the ImageNet PResNet-50 backbone weights must be fetched
separately.

## Datasets

Neither dataset is redistributed with this repository. Download them from the
official sources and lay them out as described in
[datasets/README.md](datasets/README.md):

| Dataset | Official source |
|---|---|
| VEDAI | <https://downloads.greyc.fr/vedai/> (GREYC, Razakarivony & Jurie 2015) |
| M3FD | <https://github.com/JinyuanLiu-CV/TarDAL> (TarDAL, Liu et al., ACCV 2022) |

The COCO-format annotation files used by this project (VEDAI fold-1 / 8-class
HBB; M3FD-LT20 long-tailed split, seed 42) are prepared from the official
releases; the conversion rules are documented in the dataset loader sources.

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
| Multi-seed mean ± std | **not yet run** — plan in [docs/MULTISEED_PLAN.md](docs/MULTISEED_PLAN.md) |
| M3FD-LT20 headline AP | **unresolved** — manuscript says 55.90, the weight log says 55.95 |
| DVTOD independent test split | **does not exist** — reported numbers are validation numbers |

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
  title   = {Symmetric Fusion and Reliability-Aware Supervision
             for Long-Tailed {RGB--IR} Remote-Sensing Detection},
  author  = {Chen, Yi and Deng, Lingjun and Zhong, Chuen-Ho and Liu, Chang and Dong, Yanni},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations
             and Remote Sensing},
  year    = {2026}
}
```

## Acknowledgements

The numerical calculations in this work were carried out on the supercomputing
system of the Supercomputing Center of Wuhan University. This study was supported
by the National Natural Science Foundation of China under Grant U2541203.
