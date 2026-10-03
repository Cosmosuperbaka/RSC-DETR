#!/usr/bin/env python3
"""Capture the software and hardware stack into environment/env-lock.yaml.

Run this on the machine that produces results, so the lock always reflects
reality rather than intent::

    python3 tools/collect_env.py            # print a report, write nothing
    python3 tools/collect_env.py --write    # refresh environment/ files

The script refuses to write when CUDA is unavailable, because a lock captured
on a machine that cannot run the code is worse than no lock at all.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import sys as _sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        _sys.path.insert(0, str(_parent))
        break

REPO = Path(__file__).resolve().parents[1]
ENV_DIR = REPO / "environment"


def run(cmd: list[str]) -> str:
    """Return stdout of *cmd*, or '' when the command is unavailable."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect() -> dict:
    import torch
    import torchvision
    info: dict = {
        "captured_at": date.today().isoformat(),
        "host": {
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "frameworks": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_built": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        },
        "cuda_available": torch.cuda.is_available(),
    }

    for name in ("numpy", "PIL", "cv2", "yaml", "onnx", "onnxruntime",
                 "pycocotools", "faster_coco_eval"):
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", None)
            if version is None and name == "PIL":
                import PIL
                version = PIL.__version__
            info.setdefault("packages", {})[name] = version
        except Exception:  # noqa: BLE001
            info.setdefault("packages", {})[name] = None

    if info["cuda_available"]:
        info["hardware"] = {
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
        }
        nvidia = run([
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if nvidia:
            driver, mem = nvidia.splitlines()[0].split(",")
            info["hardware"]["driver_version"] = driver.strip()
            info["hardware"]["gpu_memory_mib"] = int(mem.strip())

    return info


def report(info: dict) -> str:
    lines = ["Captured environment", "=" * 60]
    lines.append(f"  os        : {info['host']['os']}")
    lines.append(f"  python    : {info['host']['python']}")
    for key, value in info["frameworks"].items():
        lines.append(f"  {key:<10}: {value}")
    lines.append(f"  cuda?     : {info['cuda_available']}")
    for key, value in (info.get("hardware") or {}).items():
        lines.append(f"  {key:<10}: {value}")
    lines.append("-" * 60)
    for key, value in sorted((info.get("packages") or {}).items()):
        flag = " " if value else "!"
        lines.append(f" {flag}{key:<18}: {value}")
    return "\n".join(lines)


def write_files(info: dict) -> list[Path]:
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    # 1) machine-readable snapshot
    snapshot = ENV_DIR / "env-snapshot.json"
    snapshot.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    written.append(snapshot)

    # 2) the freeze of the running interpreter
    freeze = run([sys.executable, "-m", "pip", "freeze"])
    if freeze:
        target = ENV_DIR / f"pip-freeze-{info['host']['python'].replace('.', '')}.txt"
        target.write_text(freeze + "\n", encoding="utf-8")
        written.append(target)

    # 3) echo the resolved values back into the human-readable lock
    lock = ENV_DIR / "env-lock.yaml"
    if lock.exists():
        text = lock.read_text(encoding="utf-8")
        lines = text.splitlines()
        out = []
        for line in lines:
            if line.startswith("captured_at:"):
                out.append(f'captured_at: "{info["captured_at"]}"')
            else:
                out.append(line)
        lock.write_text("\n".join(out) + "\n", encoding="utf-8")
        written.append(lock)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write the captured values into environment/")
    args = parser.parse_args()

    try:
        info = collect()
    except ImportError as exc:
        print(f"Cannot capture the environment: {exc}")
        print("\nThis tool inspects the stack that actually runs the code, so it")
        print("has to run inside it -- activate the locked conda environment first:")
        print("    conda activate rscdetr && python3 tools/collect_env.py --write")
        return 1

    print(report(info))

    if not info["cuda_available"]:
        print("\nCUDA is not available on this machine.")
        if args.write:
            print("Refusing to write the lock -- capture it where the code actually runs.")
            return 1
        return 0

    if args.write:
        for path in write_files(info):
            print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
