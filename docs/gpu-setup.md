# Optional AI runtime and GPU setup

For the Ubuntu `.deb` installation, use `uwcc-model-tools` and `uwcc-nvidia`
with the [packaged setup commands](ubuntu-packaging.md#nvidia-runtime).
The Python extras and virtual environments below are the source/Python-package
alternative.

UWCC runs classical correction without an AI runtime. Local neural enhancement
and FastDVDnet denoise need model files and an inference runtime. Install from
the repository root with Python 3.12+; FFmpeg/ffprobe must also be on PATH.

| Extra | Runtime | Hardware and platform |
| --- | --- | --- |
| `ml` | ONNX Runtime CPU | CPU on supported Python platforms |
| `ml-nvidia` | ONNX Runtime CUDA, with pip CUDA/cuDNN libraries | Compatible NVIDIA GPU and driver, Linux or Windows |
| `ml-amd` | ONNX Runtime DirectML | DirectX 12 capable AMD GPU and driver, Windows |
| `vision` | OpenCV | Optional stabilization and OpenCV model runtime |

Choose **one** ML extra per virtual environment. `onnxruntime`,
`onnxruntime-gpu`, and `onnxruntime-directml` all install the `onnxruntime`
Python module; combining extras can overwrite it. Both GPU packages include a
CPU provider, so do not add `ml` for CPU fallback. Use separate environments to
try CUDA and DirectML on a machine with both vendors' GPUs. UWCC selects CUDA,
then DirectML, then CPU from the providers in the installed runtime; it does
not distribute inference across GPUs.

Platform markers skip GPU dependencies outside their listed operating systems;
installing `ml-amd` on Linux does **not** install an AMD runtime. UWCC currently
does not select ROCm or MIGraphX providers. On Linux with AMD hardware, use `ml`
for CPU inference. PyTorch/ROCm alone does not accelerate UWCC's ONNX inference.

## CPU

```bash
python3 -m venv .venv-ai
.venv-ai/bin/python -m pip install --upgrade pip
.venv-ai/bin/python -m pip install -e '.[ml,vision]'
.venv-ai/bin/python -m uwcc models
.venv-ai/bin/python -m uwcc
```

## NVIDIA on Linux or Windows

Install a compatible NVIDIA driver separately; `nvidia-smi` should recognize
the GPU before starting UWCC. The extra installs CUDA/cuDNN user-space
libraries, not the driver. Required driver and GPU support depend on the ONNX
Runtime/CUDA version pip resolves. Check the official
[CUDA provider compatibility and installation guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
when selecting a version for older hardware or drivers. UWCC preloads these
libraries before initializing CUDA.

Linux:

```bash
python3 -m venv .venv-nvidia
.venv-nvidia/bin/python -m pip install --upgrade pip
.venv-nvidia/bin/python -m pip install -e '.[ml-nvidia,vision]'
.venv-nvidia/bin/python -m uwcc models
.venv-nvidia/bin/python -m uwcc
```

Windows PowerShell (also requires the Microsoft Visual C++ runtime described
in the [ONNX Runtime installation guide](https://onnxruntime.ai/docs/install/)):

```powershell
py -3.12 -m venv .venv-nvidia
.venv-nvidia\Scripts\python.exe -m pip install --upgrade pip
.venv-nvidia\Scripts\python.exe -m pip install -e ".[ml-nvidia,vision]"
.venv-nvidia\Scripts\python.exe -m uwcc models
.venv-nvidia\Scripts\python.exe -m uwcc
```

## AMD on Windows

Use Windows 10 version 1903 or later (Windows 11 is the project portability
target), a DirectX 12 capable GPU, and a current AMD graphics driver. Install
the Microsoft Visual C++ runtime required by ONNX Runtime. CUDA and ROCm are
not required for this path. DirectML can also run on compatible NVIDIA or Intel
GPUs. See the official [DirectML requirements and limitations](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).

```powershell
py -3.12 -m venv .venv-amd
.venv-amd\Scripts\python.exe -m pip install --upgrade pip
.venv-amd\Scripts\python.exe -m pip install -e ".[ml-amd,vision]"
.venv-amd\Scripts\python.exe -m uwcc models
.venv-amd\Scripts\python.exe -m uwcc
```

UWCC disables memory patterns and uses sequential session execution as required
by DirectML. The default adapter is used, which may be an integrated GPU on a
multi-adapter laptop. Windows/AMD execution has not been hardware-validated in
this project. DirectML is maintained upstream in sustained engineering mode.

## Model files and verification

Extras install runtimes, not weights or model conversion tools. Use existing
checksum-verified models under `models/`, or follow
[Rebuild models on another machine](ai-workflow.md#rebuild-models-on-another-machine).
Model preparation is an explicit step; UWCC does not download models at launch.
Keep PyTorch and conversion dependencies in a separate model preparation
environment. `UWCC_MODEL_DIR` can point to a different models directory.

Run `python -m uwcc models` with the Python executable of the new environment.
The provider list should include `CUDAExecutionProvider` for NVIDIA or
`DmlExecutionProvider` for DirectML. This lists available providers; actual
session initialization and inference may still fail. Open a clip and choose
**Local AI enhancement + white balance**; analysis starts automatically once
the input color profile is known. For denoise,
enable **Use FastDVDnet AI denoise**, set a nonzero denoise strength, and export.
Export progress reports `CUDA GPU · ONNX Runtime` or
`DirectML GPU · ONNX Runtime` when that provider is active. Unsupported model
operations may still execute on CPU.

The optimized full-frame/FP16 FastDVDnet variants and
`tools/validate_gpu_denoise.py` are CUDA-only. DirectML uses the standard tiled
model; NVIDIA performance measurements in the
[AI workflow guide](ai-workflow.md#gpu-environment-on-this-machine) do not apply
to AMD. GPU inference does not enable GPU codecs or accelerate all color work.

If initialization or inference fails, UWCC falls back to CPU and reports the
actual backend. To force CPU, set `UWCC_INFERENCE_DEVICE=cpu` before launching:

```bash
UWCC_INFERENCE_DEVICE=cpu .venv-nvidia/bin/python -m uwcc
```

```powershell
$env:UWCC_INFERENCE_DEVICE = "cpu"
.venv-amd\Scripts\python.exe -m uwcc
```

Unset the variable or set it to `auto` to restore automatic provider selection.
Restart UWCC after changing environments or runtimes; a running export cannot
switch them. If the provider is missing, confirm the launch interpreter and
create a clean environment with exactly one ML extra. Avoid
`--system-site-packages` when it would expose another ONNX Runtime installation.
