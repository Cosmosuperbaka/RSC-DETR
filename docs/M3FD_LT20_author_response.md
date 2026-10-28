# Author response draft: M3FD-LT20 construction and reproducibility

**Reviewer comment:** Please clarify how M3FD-LT20 was constructed, how the
long-tailed sampling handles images containing multiple classes, and how the
reported split and imbalance statistics can be reproduced.

**Response:** Thank you for identifying that the original description did not
provide enough detail to reproduce the M3FD-LT20 protocol. We have revised
Section IV-A and released a reproducibility package containing the
split-generation and finalization scripts, fixed filename-stem lists, count
tables, manifests, and verification code under
`datasets/M3FD/reproducibility/` and `scripts/m3fd/`.

M3FD does not provide official split lists in the source snapshot used here.
We therefore generate a project-defined base split after excluding the
unreadable-IR pair `00375`. The raw collection has 4,200 RGB--IR pairs; 4,199
are valid. The base train/validation/test split contains 3,359/420/420 pairs.
The base validation and test selections use Python `random.Random` seeds 42
and 1042, respectively. Long-tail sampling uses seed 42 and selects complete
RGB--IR pairs from the base training list. Every valid annotation in a
selected image is retained, including co-occurring classes; the class targets
guide a greedy objective and are not per-class box quotas. Validation is not
long-tail resampled.

The data-split seed 42 is a dataset-construction parameter and is independent
of the model-training seed (`--seed`). All M3FD-LT20 training runs in the paper
use the same fixed split; training seeds 42, 3407, and 2024 are separate runs
on that split. The directory name `lt20_seed42` and the `seed` fields in the
manifest refer to the data-split seed.

We also clarified the exact-file duplicate repair. Test pair `00539` is removed
with its 14 annotations because its IR file is byte-identical to validation
pair `01300`, which is retained. The final split contains
1,428/420/419 train/validation/test pairs and 16,406 training instances. The
Car and Motorcycle counts are 8,379 and 407, respectively, yielding
IF = 20.59. The released verification report checks split/list/annotation
consistency and RGB/IR byte hashes; we explicitly state that this check does
not establish scene-level isolation.

The released materials contain the fixed lists, scripts, manifests, and
count tables, but not the M3FD images or the per-box annotations. Users with
licensed access to M3FD can regenerate the COCO annotations locally with
`scripts/m3fd/finalize_lt20.py`; the README records the expected SHA-256 values
and the full command sequence.

**Suggested manuscript location:** Section IV-A, dataset construction paragraph
for M3FD-LT20.
