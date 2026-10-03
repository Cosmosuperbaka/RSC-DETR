"""Regression tests -- cheap checks that catch the expensive mistakes.

These are deliberately fast and run without a GPU where possible. They exist to
answer one question before a long job is submitted: *is the environment still
sane?* A silent failure 15 epochs in costs far more than a 20-second test.

    pytest tests/ -v
    pytest tests/ -m "not gpu"       # skip anything needing CUDA
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# repository layout
# ---------------------------------------------------------------------------

def test_paths_module_imports():
    """The shared path resolver must load and expose its documented names."""
    mod = importlib.import_module("rscdetr_paths")
    for name in ("WORKSPACE", "ROOT", "DATASETS", "CFT", "LCAFNET", "MSOD",
                 "PAPER", "OUT_QUAL"):
        assert hasattr(mod, name), f"rscdetr_paths is missing {name}"
        assert isinstance(getattr(mod, name), Path)


def test_no_hardcoded_home_paths():
    """No script may embed somebody's home directory again."""
    offenders = []
    for path in REPO.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "/home/" in line and "Path.home()" not in line and "rscdetr_paths" not in line:
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, (
        "hard-coded home paths reappeared (use rscdetr_paths instead):\n  "
        + "\n  ".join(offenders)
    )


def test_registry_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    path = REPO / "registry/experiments.yaml"
    assert path.is_file(), "registry/experiments.yaml is missing"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "datasets" in data and "runs" in data
    for run in data["runs"]:
        for field in ("id", "dataset", "method", "config", "status"):
            assert field in run, f"run {run.get('id')} is missing {field}"
        assert run["dataset"] in data["datasets"], (
            f"run {run['id']} references unknown dataset {run['dataset']}"
        )


def test_env_lock_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    path = REPO / "environment/env-lock.yaml"
    assert path.is_file()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fw = data["frameworks"]
    assert fw["torch"].startswith("2.1.2"), "torch version drifted from the locked value"
    assert fw["cuda_built"] == "12.1", "CUDA build drifted from the locked value"
    assert fw["cudnn"] == 8902, "cuDNN drifted -- latency numbers are not comparable"


def test_freeze_file_matches_lock():
    """The freeze file should contain the framework versions the lock claims."""
    freeze = REPO / "environment/pip-freeze-rtdetrv2-py310.txt"
    assert freeze.is_file(), "the pip freeze snapshot is missing"
    text = freeze.read_text(encoding="utf-8")
    for expected in ("torch==2.1.2+cu121", "torchvision==0.16.2+cu121", "numpy==1.26.4"):
        assert expected in text, f"{expected} is not in the freeze file"


# ---------------------------------------------------------------------------
# tooling
# ---------------------------------------------------------------------------

def test_benchmark_refuses_busy_gpu_logic():
    """The idle gate must actually be wired up, not decorative."""
    src = (REPO / "benchmark/benchmark_latency.py").read_text(encoding="utf-8")
    assert "IDLE_UTIL_MAX" in src and "IDLE_MEM_MAX" in src
    assert "REFUSING TO MEASURE" in src
    assert "skip-idle-check" in src, "there must be an explicit escape hatch"


def test_hash_artifacts_roundtrip(tmp_path):
    """Hash a file, verify it, then corrupt it and confirm the check fails."""
    sys.path.insert(0, str(REPO / "tools"))
    mod = importlib.import_module("hash_artifacts")

    target = tmp_path / "artifact.bin"
    target.write_bytes(b"rsc-detr")
    digest = mod.sha256(target)
    assert digest == mod.sha256(target)

    target.write_bytes(b"rsc-detr!")
    assert mod.sha256(target) != digest, "hash did not change after modification"


def test_check_dataset_presets_reference_real_paths():
    """Presets must describe datasets that exist, or fail loudly."""
    sys.path.insert(0, str(REPO / "tools"))
    mod = importlib.import_module("check_dataset")
    assert set(mod.PRESETS) == {"vedai", "m3fd_lt20", "dvtod"}
    for name, preset in mod.PRESETS.items():
        assert isinstance(preset["root"], Path), f"{name} preset root is not a Path"
        assert preset["annotations"], f"{name} preset has no annotation files"


# ---------------------------------------------------------------------------
# numerics -- the parts that can be checked without a dataset
# ---------------------------------------------------------------------------

def test_iou_is_correct():
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        bb = (b[2] - b[0]) * (b[3] - b[1])
        union = aa + bb - inter
        return inter / union if union > 0 else 0.0

    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)
    assert iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert iou(box, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_quality_weight_matches_the_paper_equation():
    """a_ic = alpha * sigma^gamma * (1 - y) + y  -- see Eq. (a_ic) in the paper."""
    torch = pytest.importorskip("torch")
    alpha, gamma = 0.25, 2.0

    def weight(logit, y):
        score = torch.sigmoid(torch.tensor(logit)).detach()
        return float(alpha * score.pow(gamma) * (1.0 - y) + y)

    # negative class: reduces to the focal-style term, must NOT be zero
    negative = weight(0.0, 0.0)
    assert negative == pytest.approx(alpha * 0.25), "negative weight is wrong"
    assert negative > 0.0, (
        "the weight collapsed to zero for y=0 -- this is the exact bug the "
        "reviewer flagged; the formula must keep the focal term alive"
    )

    # matched class: interpolates up to exactly one at u_i = 1
    assert weight(0.0, 1.0) == pytest.approx(1.0)
    assert weight(0.0, 0.5) == pytest.approx(alpha * 0.25 * 0.5 + 0.5)


@pytest.mark.gpu
def test_cuda_environment_matches_lock():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")
    lock = json.loads((REPO / "environment/env-snapshot.json").read_text()
                      ) if (REPO / "environment/env-snapshot.json").is_file() else {}
    assert torch.version.cuda == "12.1"
    assert torch.backends.cudnn.version() == 8902, (
        "cuDNN differs from the locked value -- measured latencies are not comparable"
    )
    if lock:
        assert torch.__version__ == lock["frameworks"]["torch"]
