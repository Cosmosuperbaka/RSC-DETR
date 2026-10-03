# Environment

This directory is the single source of truth for the software stack behind the
reported numbers. Two files, two jobs:

| File | What it is |
|---|---|
| `env-lock.yaml` | Human-readable lock: OS, interpreter, framework versions, CUDA/cuDNN, GPU, driver. Read this to know *what* the results were produced with. |
| `pip-freeze-rtdetrv2-py310.txt` | Verbatim `pip freeze` of the exact conda environment that ran the paper (122 packages). Read this to rebuild it byte-for-byte. |

## The environment that produced the paper

```
conda env:  rtdetrv2
python:     /home/<user>/miniconda3/envs/rtdetrv2/bin/python  (CPython 3.10.20)
torch:      2.1.2+cu121        torchvision: 0.16.2+cu121
CUDA built: 12.1               cuDNN:       8.9.2  (8700 -> 8902)
GPU:        8x NVIDIA RTX 4090 (24 GB, sm_89), driver 590.48.01
OS:         Ubuntu 22.04.5 LTS
```

## Rebuilding it

```bash
conda create -n rscdetr python=3.10.20 -y
conda activate rscdetr

# The +cu121 wheels are mandatory -- a +cu118 wheel pulls a different cuDNN and
# invalidates the latency protocol in docs/BENCHMARK_PROTOCOL.md.
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r environment/pip-freeze-rtdetrv2-py310.txt
```

> If you install the full freeze file first and it tries to downgrade torch,
> pin the two torch packages explicitly *before* running the freeze file, as
> shown above.

## Re-capturing the lock

Whenever the environment changes, refresh both files rather than editing them by
hand. The helper records the framework versions, CUDA/cuDNN, GPU properties and
a fresh `pip freeze`:

```bash
python3 tools/collect_env.py --write
```

It refuses to write if `torch.cuda.is_available()` is `False`, so a lock is
never captured on a machine that cannot actually run the code.

## Caveats worth knowing

- **The host is shared (8 GPUs).** Version numbers are stable, but anything
  time-related is not. Always use the benchmark protocol.
- **`torch._C._GLIBCXX_USE_CXX11_ABI` is `False`.** If you build a custom
  extension, compile it against the same ABI or the import will fail.
- **A second environment exists on the same host** (`~/.venv`, torch
  `2.0.1+cu118`). It was used for some early exploratory scripts only. It did
  **not** produce any reported result — do not use it to reproduce the paper.
