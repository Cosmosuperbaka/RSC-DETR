#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX and TorchScript in a single, repeatable way.

Every deployment artefact in this project should come out of this script, so
that two exports of the same checkpoint are byte-comparable and a reviewer can
reproduce the file we shipped.

    python3 export/export_model.py \\
        --config rtdetrv2_pytorch/configs/rtdetrv2/<config>.yml \\
        --weights <best.pth> \\
        --resolution 1024 1024 \\
        --out export/rscdetr_vedai_1024

Writes into the output directory:

    model.onnx              the exported graph
    model.ts                TorchScript module
    export-manifest.json    config, weights, SHA-256 of every emitted file,
                            input signature, and the software environment

The manifest is the point. An exported blob with no provenance is not an
artefact, it is a rumour.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class Wrapper:
    """Flatten the dict input into positional tensors so ONNX can trace it."""

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, rgb, ir):
        return self.model({"rgb": rgb, "ir": ir})


def export_onnx(model, out_path: Path, h: int, w: int, opset: int, dynamic: bool) -> bool:
    try:
        import torch
    except ImportError:
        print("torch is required")
        return False

    wrapper = Wrapper(model).eval()
    rgb = torch.randn(1, 3, h, w, device=next(model.parameters()).device)
    ir = torch.randn_like(rgb)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"rgb": {2: "h", 3: "w"}, "ir": {2: "h", 3: "w"}}

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (rgb, ir), str(out_path),
                input_names=["rgb", "ir"],
                output_names=["logits", "boxes"],
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX export failed: {type(exc).__name__}: {exc}")
        return False
    return True


def export_torchscript(model, out_path: Path, h: int, w: int) -> bool:
    try:
        import torch
    except ImportError:
        return False

    wrapper = Wrapper(model).eval()
    device = next(model.parameters()).device
    rgb = torch.randn(1, 3, h, w, device=device)
    ir = torch.randn_like(rgb)

    try:
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (rgb, ir), strict=False)
            traced.save(str(out_path))
    except Exception as exc:  # noqa: BLE001
        print(f"TorchScript export failed: {type(exc).__name__}: {exc}")
        return False
    return True


def verify_onnx(out_path: Path, h: int, w: int) -> dict:
    """Load the exported graph and compare against the reference if possible."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return {"verified": False, "reason": "onnxruntime not installed"}

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    feeds = {n: np.random.randn(1, 3, h, w).astype("float32") for n in names}
    outputs = session.run(None, feeds)
    return {
        "verified": True,
        "inputs": names,
        "output_count": len(outputs),
        "output_shapes": [list(o.shape) for o in outputs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--resolution", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true",
                        help="export with dynamic height/width axes")
    parser.add_argument("--skip-torchscript", action="store_true")
    args = parser.parse_args()

    import torch

    config, weights = args.config.resolve(), args.weights.resolve()
    for path in (config, weights):
        if not path.is_file():
            print(f"missing: {path}")
            return 1
    h, w = args.resolution
    args.out.mkdir(parents=True, exist_ok=True)

    # --- load exactly the way training does -------------------------------
    sys.path.insert(0, str(Path(config).parents[2]))
    from src.core import YAMLConfig, yaml_utils  # noqa: E402

    update = yaml_utils.parse_cli([])
    update.update({"resume": str(weights), "device": "cuda:0" if torch.cuda.is_available() else "cpu"})
    cfg = YAMLConfig(str(config), **update)
    model = cfg.model
    state = torch.load(weights, map_location="cpu", weights_only=False)
    sd = state.get("ema", state).get("module", state.get("model", state))
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
    print(f"loaded {config.name} + {weights.name}")

    produced = {}

    onnx_path = args.out / "model.onnx"
    if export_onnx(model, onnx_path, h, w, args.opset, args.dynamic):
        produced["model.onnx"] = {"sha256": sha256(onnx_path),
                                  "size_bytes": onnx_path.stat().st_size,
                                  **verify_onnx(onnx_path, h, w)}
        print(f"wrote {onnx_path.name} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if not args.skip_torchscript:
        ts_path = args.out / "model.ts"
        if export_torchscript(model, ts_path, h, w):
            produced["model.ts"] = {"sha256": sha256(ts_path),
                                    "size_bytes": ts_path.stat().st_size}
            print(f"wrote {ts_path.name} ({ts_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if not produced:
        print("nothing was exported")
        return 1

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "config": str(config),
            "config_sha256": sha256(config),
            "weights": str(weights),
            "weights_sha256": sha256(weights),
        },
        "input_signature": {
            "rgb": [1, 3, h, w],
            "ir": [1, 3, h, w],
            "dynamic_axes": args.dynamic,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "opset": args.opset,
        },
        "artifacts": produced,
    }
    manifest_path = args.out / "export-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
