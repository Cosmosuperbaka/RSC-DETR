#!/usr/bin/env python3
"""Produce SHA-256 manifests for the artefacts behind a reported number.

A metric is only trustworthy if you can point at the exact files that produced
it. This tool walks a run directory and writes a manifest linking the trio that
matters:

    config.yml  ->  best.pth  ->  predictions.json

Usage::

    python3 tools/hash_artifacts.py --run result/RTDOD_HBB_3class/<run> \\
        --out manifests/dvtod_rscdetr.json

    # verify later that nothing moved
    python3 tools/hash_artifacts.py --verify manifests/dvtod_rscdetr.json

Large files are streamed in chunks, so a 200 MB checkpoint does not blow up
memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import sys as _sys

for _parent in Path(__file__).resolve().parents:
    if (_parent / "rscdetr_paths.py").is_file():
        _sys.path.insert(0, str(_parent))
        break

from rscdetr_paths import ROOT  # noqa: E402

CHUNK = 1 << 20          # 1 MiB
INTERESTING = {".yml", ".yaml", ".json", ".pth", ".pt", ".log", ".txt", ".cfg"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def collect(run_dir: Path, max_depth: int) -> list[dict]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in INTERESTING:
            continue
        if len(path.relative_to(run_dir).parts) > max_depth:
            continue
        entries.append({
            "path": path.relative_to(ROOT).as_posix()
                    if ROOT in path.parents else str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return entries


def resolve(rel: str) -> Path:
    """Turn a manifest path back into something openable."""
    candidate = Path(rel)
    return candidate if candidate.is_absolute() else ROOT / rel


def verify(manifest_path: Path) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = 0
    for entry in data["artifacts"]:
        path = resolve(entry["path"])
        if not path.is_file():
            print(f"MISSING  {entry['path']}")
            bad += 1
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            print(f"CHANGED  {entry['path']}")
            print(f"         expected {entry['sha256'][:16]}...")
            print(f"         actual   {actual[:16]}...")
            bad += 1
        else:
            print(f"ok       {entry['path']}")
    print(f"\n{len(data['artifacts']) - bad}/{len(data['artifacts'])} intact")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="run directory to fingerprint")
    parser.add_argument("--out", type=Path, help="where to write the manifest")
    parser.add_argument("--verify", type=Path, help="re-check an existing manifest")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    if args.verify:
        return verify(args.verify)

    if not args.run:
        parser.error("either --run or --verify is required")

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}")
        return 1

    artifacts = collect(run_dir, args.max_depth)
    if not artifacts:
        print(f"no fingerprintable files under {run_dir}")
        return 1

    manifest = {
        "run": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": len(artifacts),
        "total_bytes": sum(a["size_bytes"] for a in artifacts),
        "artifacts": artifacts,
    }

    print(f"{len(artifacts)} files, "
          f"{manifest['total_bytes'] / 1024 / 1024:.1f} MB")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        for a in artifacts:
            print(f"  {a['sha256'][:12]}  {a['size_bytes']:>12,}  {a['path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
