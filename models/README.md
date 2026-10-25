# Prepare local AI models

This project tracks its model JSON manifests and upstream license notices.
It does not redistribute upstream weights or generated ONNX binaries. Download
weights from their original hosts and convert them on your own machine.

`tools/prepare_models.py` records the pinned source revisions, weight URLs, and
expected SHA-256 checksums:

| Model | Original source | Weight host |
| --- | --- | --- |
| UDnet V1 | [alzayats/UDnet](https://github.com/alzayats/UDnet) | [Alzayats/UDnet](https://huggingface.co/Alzayats/UDnet) |
| FastDVDnet | [m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet) | The same upstream GitHub repository |

## Download and convert

Run from the repository root with Python 3.12+ and network access. A separate
CPU environment is sufficient for conversion, even if inference will use a GPU:

```bash
python3 -m venv .venv-models
.venv-models/bin/python -m pip install --upgrade pip
.venv-models/bin/python -m pip install -e '.[ml,vision]'
.venv-models/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-models/bin/python -m pip install onnx onnxscript
.venv-models/bin/python tools/prepare_models.py
.venv-models/bin/python -m uwcc models
```

On Windows, create the environment with `py -3.12 -m venv .venv-models` and
replace `.venv-models/bin/python` with `.venv-models\Scripts\python.exe`.
Use double quotes around the extras argument in PowerShell.

The script downloads source code and licenses at pinned revisions into
`models/.sources/`, verifies downloaded weight checksums, loads tensor state
dictionaries with `weights_only=True`, and exports ONNX models. It checks
inference parity before writing local manifests and copies upstream notices.
The fetched upstream Python code runs during conversion, which is an explicit
setup operation. UWCC itself uses the generated ONNX files and never downloads
weights automatically.

To download without installing conversion dependencies:

```bash
python3 tools/prepare_models.py --download-only
```

This requires only Python's standard library and does not produce runnable ONNX
models. Run the full preparation command afterward to convert; verified cached
weights are reused. Source code and licenses are fetched again from their
pinned revisions.

Use `--models-dir /path/to/models` for another output directory, then set
`UWCC_MODEL_DIR` to that directory when launching UWCC. Keep the generated ONNX
files beside their generated manifests: conversion tool versions can affect
binary hashes, so a checked-in manifest alone is not sufficient. Rebuilding
can update the tracked manifests with hashes and measurements for your output.

## Optional NVIDIA variants

After the standard models are prepared, generate the dynamic FP32/FP16
FastDVDnet graphs in the conversion environment:

```bash
.venv-models/bin/python tools/prepare_gpu_denoise.py
```

Then use a separate [NVIDIA runtime environment](../docs/gpu-setup.md) to validate
and enable them on the target CUDA device:

```bash
.venv-nvidia/bin/python tools/validate_gpu_denoise.py
```

Preparation marks these variants unvalidated; the CUDA validator registers
them only after accuracy checks pass. Standard CPU and DirectML inference
does not require this step. Rebuilding standard models replaces their manifest;
repeat GPU preparation and validation afterward if needed.

Downloaded sources, weights, and generated `*.onnx` files are ignored by Git.
Do not attach them to project releases; distribute the setup scripts, manifests,
and license notices instead. Once prepared, local inference works offline.
