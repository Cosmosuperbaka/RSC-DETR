# M3FD-LT20 reproducibility record

This directory defines the **fixed M3FD-LT20 split used in the paper**. It
contains the split lists, construction scripts, count tables, manifests, and
the file-level duplicate report. It does **not** contain M3FD images or the
per-box annotation files; obtain the dataset from the official source and
regenerate the COCO annotations locally with the scripts below.

The split is a **project-generated fixed split**, not an official M3FD split.
M3FD itself does not publish train/validation/test lists for this protocol.

## Protocol at a glance

| Item | Value |
|---|---:|
| Raw RGB--IR pairs | 4,200 |
| Unreadable pair excluded before splitting | `00375` (bad IR) |
| Valid pairs | 4,199 |
| Base split (train / val / test) | 3,359 / 420 / 420 |
| Final LT20 split (train / val / test) | **1,428 / 420 / 419** |
| Annotations (train / val / test) | 16,406 / 4,119 / 4,095 |
| Training instances, largest / smallest class | 8,379 (Car) / 407 (Motorcycle) |
| Actual imbalance factor (IF) | **20.59** |
| Classes | People, Car, Bus, Lamp, Motorcycle, Truck |

The sampling unit is a **complete RGB--IR image pair**. Every valid
annotation in a selected image is retained, including co-occurring classes.
The long-tail class targets are optimization references, **not hard quotas**;
no boxes are added or deleted to force a target.

The validation set is not long-tail resampled. The final test set contains 419
pairs rather than 420 because one exact-file duplicate was removed (see below).

## Seed semantics: two different things are called "seed"

This is the most common source of confusion in this release. The directory
name `lt20_seed42` and all `seed` fields inside `split_manifest.json` refer to
**data-split sampling seeds**. Training runs use a separate `--seed` argument.

| Role | Values | Where it appears |
|---|---|---|
| Base-split validation sampling | 42 | `split_manifest.json` -> `random_seeds.base_validation` |
| Base-split test sampling | 1042 | `split_manifest.json` -> `random_seeds.base_test` |
| LT20 training-subset sampling | 42 | `split_manifest.json` -> `random_seeds.long_tail_train` |
| Model-training seed, headline M3FD-LT20 run | 42 | `--seed 42` on the training command |
| Additional model-training seeds | 3407, 2024 | `--seed 3407`, `--seed 2024` |

Consequences:

* **All M3FD-LT20 training runs in the paper use the same fixed data split**
  (`lt20_seed42`), regardless of the training seed.
* Run names such as `..._seed3407_native_res_b8_45e` and
  `v19c_spsf_seed2024_...` are **training-seed** variants. They do not use a
  different data split.
* `lt20_seed3407/` and `lt20_seed1234/` are alternative **data-split**
  protocols that `build_lt_split.py --seeds 1234,3407` can produce. They are
  not included here and are not used in the paper; do not substitute them for
  `lt20_seed42`.
* VEDAI run names such as `s3407` / `seed3407` are also training seeds; VEDAI
  uses the official fold-1 split.

## Directory layout

```text
datasets/M3FD/
├── raw/                            # not distributed; place official data here
│   ├── vi/                         # visible images
│   ├── ir/                         # infrared images
│   └── labels/                     # YOLO-format labels, class ids 0-5
└── reproducibility/
    ├── README.md                   # this file
    ├── splits/
    │   ├── train.txt               # 1,428 stems (final)
    │   ├── val.txt                 # 420 stems
    │   ├── test.txt                # 419 stems (final; 00539 excluded)
    │   └── base/                   # pre-long-tail base lists (3,359/420/420)
    ├── split_manifest.json         # dataset version, counts, IF, seeds, SHA-256, dedupe
    ├── class_counts.csv            # training instances/images per class
    ├── image_class_counts.csv      # per-training-image class counts
    ├── ir_deduplication_report.json
    ├── verify_report.json
    └── history/
        └── test.pre_ir_dedupe.txt  # historical 420-image test list, NOT used
```

Each line of a split file is a bare image stem such as `00001`. The COCO
`file_name` is `00001.png` (or the source extension); the IR field is
`ir_file_name`.

`history/test.pre_ir_dedupe.txt` is archived **only** so the removal of
`00539` can be audited. `splits/test.txt` is the evaluation list. Do not use
the historical file for training or evaluation.

## Obtain the raw data

Download M3FD from the official TarDAL release:

<https://github.com/JinyuanLiu-CV/TarDAL>

Follow that project's license and terms of use. The TarDAL download provides
`ir/`, `vi/`, `labels/` (YOLO txt), and `meta/`. Copy the contents of its
`m3fd` directory into `datasets/M3FD/raw/` so that the three directories above
exist. If you start from the `Annotation/` XML files instead, convert them to
the TarDAL six-class YOLO layout first; the scripts expect the published
TarDAL `labels/*.txt` format with class ids

```text
0 People, 1 Car, 2 Bus, 3 Lamp, 4 Motorcycle, 5 Truck
```

The scripts discover `vi/`, `ir/`, and `labels/` below `datasets/M3FD/raw/`
(case-insensitive; `visible/`, `infrared/`, `rgb/` and `thermal/` are also
accepted).

## Rebuild the split

All commands run from the repository root. Use Python 3 with `Pillow` and
`NumPy` installed (the same dependencies as the data-preparation utilities).

```bash
# 0. Optional: audit the local raw copy and write reports/ + figures/
python scripts/m3fd/audit_m3fd.py --root datasets/M3FD

# 1. Base 3,359 / 420 / 420 split
python scripts/m3fd/build_base_split.py \
    --root datasets/M3FD \
    --seed 42 \
    --out-dir datasets/M3FD/reproducibility/splits/base

# 2. Long-tail training subset from the base training list (data-split seed 42)
python scripts/m3fd/build_lt_split.py \
    --root datasets/M3FD \
    --base-dir datasets/M3FD/reproducibility/splits/base \
    --seeds 42 \
    --out-dir datasets/M3FD/reproducibility/build/lt20

# 3. File-level deduplication, final lists, paired COCO JSON, counts, manifest
python scripts/m3fd/finalize_lt20.py \
    --root datasets/M3FD \
    --split-dir datasets/M3FD/reproducibility/build/lt20/lt20_seed42 \
    --report-dir datasets/M3FD/reproducibility/build/final \
    --output-dir datasets/M3FD/processed/lt20_seed42

# 4. Strict verification (counts, overlap, hashes, duplicate bytes)
python scripts/m3fd/verify_lt20_final.py \
    --split-dir datasets/M3FD/reproducibility/build/final \
    --dataset-dir datasets/M3FD/processed/lt20_seed42 \
    --report datasets/M3FD/reproducibility/build/final/verify_report.json
```

The finalizer writes the paired COCO files to
`datasets/M3FD/processed/lt20_seed42/annotations/instances_{train,val,test}.json`
and links/copies the images into `visible/` and `infrared/`. The training
configs read from `datasets/M3FD lt20`, which should point at that processed
directory.

The commands above build into `reproducibility/build/` so the released
reference lists are not overwritten. To check a rebuild, compare it with the
released lists:

```bash
sha256sum datasets/M3FD/reproducibility/splits/*.txt \
          datasets/M3FD/reproducibility/build/final/*.txt
# train 066c7b6d570fa3001f8b68348789d0e2861909283bb4a33c900834b45d6c6dbf
# val   03ccf6f572d26fa3df88fcaea5196aeef2bcce7ae6ea6d55564a1759ad44258f
# test  81326b577accbe015acc18bfc6bdbfb684474160db354dc67d2043a96b5b6f6d
```

`test.txt` is sorted, so the list hash is stable across platforms.

## Expected SHA-256 values

| Artefact | SHA-256 |
|---|---|
| `splits/train.txt` | `066c7b6d570fa3001f8b68348789d0e2861909283bb4a33c900834b45d6c6dbf` |
| `splits/val.txt` | `03ccf6f572d26fa3df88fcaea5196aeef2bcce7ae6ea6d55564a1759ad44258f` |
| `splits/test.txt` | `81326b577accbe015acc18bfc6bdbfb684474160db354dc67d2043a96b5b6f6d` |
| `instances_train.json` (LF, as generated) | `4403aca1602dfa69f9b7a1319a47d4742b53ac304c2203f58b5145a93829c8b5` |
| `instances_val.json` (LF, as generated) | `c3d548bf6c0ede36f34ca37d0ab52893d2b8ae3b15e1a5f81bf774bc0f5d1a55` |
| `instances_test.json` (LF, as generated) | `9f515d6c251597f10b2f2b2bea8eee33e29db85f5e0f8d4794d8297da534641e` |

The annotation JSON hashes refer to the exact JSON files produced by
`finalize_lt20.py`. Regenerating on another platform may change the file bytes
if the JSON formatting changes, but the split-list hashes and the
image/annotation counts must remain identical.

`split_manifest.json` records `reconstruction_verified: true`: the released
scripts were rerun on the same raw snapshot and reproduced the base lists, the
final lists, and all three COCO JSON hashes exactly.

## Final training distribution

| Class | Instances | Images |
|---|---:|---:|
| People | 4,767 | 866 |
| Car | 8,379 | 1,279 |
| Bus | 499 | 452 |
| Lamp | 1,657 | 501 |
| Motorcycle | 407 | 335 |
| Truck | 697 | 463 |
| **Total** | **16,406** | **1,428** |

IF = 8,379 / 407 = **20.59**.

## Why the test set is 419, not 420

The finalizer hashes the bytes of every RGB and IR file independently across
all three subsets, with priority `val > test > train` and then the
lexicographically first stem within a split. If either modality of a pair is
byte-identical to an already-kept file, the whole lower-priority pair is
removed.

On this snapshot:

* validation `01300` and test `00539` have byte-identical IR files
  (`55d24efe09e7456a60361d908c3307475edc68f45a07bbb819213b70d3c1ff78`);
* validation is kept, so test `00539` is removed together with its 14
  annotations;
* `removed_pair_count = 1`, hence 420 - 1 = **419**.

The 420-image list is preserved as
`history/test.pre_ir_dedupe.txt` for audit only.

**This byte-level check does not prove scene-level isolation.** It only
detects exact duplicate files. Near-duplicate scenes, overlapping captures, or
semantically related images can remain across splits. The paper reports the
file-level result; users requiring scene-level isolation must perform an
additional scene/cluster analysis.

## What is not released

* Raw RGB/IR images: obtain them from the official M3FD release.
* Per-box annotations: regenerate them locally with the scripts above; M3FD's
  license is not assumed to permit redistribution of derived annotations.
* Model weights, training logs, and visualisations: see the main repository
  (`weights/`, `registry/experiments.yaml`) and the paper.

## Verification checklist

`verify_lt20_final.py` checks all of the following and exits non-zero on any
failure:

1. list counts are 1,428 / 420 / 419;
2. no duplicate stems within a list;
3. the three lists are pairwise disjoint;
4. every list stem appears exactly once in the corresponding COCO JSON;
5. annotation counts are 16,406 / 4,119 / 4,095;
6. training class counts and IF match the table above;
7. `00539` is absent from the final test list;
8. RGB/IR/pair bytes have no duplicates;
9. the list and annotation SHA-256 values match `split_manifest.json`.

## Citation of the protocol

If you use this split, cite the RSC-DETR paper and state that the split is
**project-generated**, not an official M3FD benchmark split.
