# AI color, reference learning, and finishing

UWCC now offers three local color workflows: neural underwater enhancement,
automatic levels/white balance, and learning from a corrected version of the
same clip. The inspector exposes these alongside optional neural denoise,
background stabilization, and opt-in cloud grading advice.

## Run the installed local AI models

This checkout has converted, checksum-verified UDnet V1 and FastDVDnet models
in `models/`. UDnet needs ONNX Runtime; FastDVDnet can run through OpenCV DNN.
The isolated CPU model environment includes both runtimes and sees the system
Qt binding:

```bash
.venv-models/bin/python -m uwcc models
.venv-models/bin/python -m uwcc
```

Choose **Local AI enhancement + white balance**, confirm the input picture
profile, and press **Auto correct**. UDnet predicts color references on proxy
frames at approximately half-second intervals. Each reference receives a
levels/contrast/vibrance finish. White balance is not estimated a second time
after the network. A regularized polynomial fit captures the color transform,
and temporally interpolated LUTs apply it at full resolution. This preserves
source geometry and detail and avoids running a neural enhancer on every
100 fps full-resolution frame. Spatial effects in a network reference are
approximated by a global color transform, not copied into the source.

The engine label reports the actual model or fallback. If the built-in model
or runtime is unavailable, automatic selection reports the fallback and uses
levels plus gray-edge white balance. Explicitly selected invalid models fail
instead of being silently substituted.

The non-neural **Automatic levels / white balance** mode estimates illuminant
color from edge statistics and robust highlights, applies bounded channel gains,
sets black/white levels and midtone brightness, and adds a gentle S curve and
vibrance. Its parameters are smoothed within shots. **Legacy balance** remains
available for older looks. Manual controls and their keyframes apply afterward;
0% correction strength bypasses automatic correction.

The encoded LUT path is optimized for SDR Rec.709 input/output. Existing log/HDR
handling still uses the scene-linear matrix path. Verify the camera's recording
profile: Sony picture profiles cannot be inferred reliably from file names.

## Learn the look of a corrected reference

Choose **Learn from corrected reference…**, selecting a corrected version of
the open source clip. The reference must preserve the same framing, timing,
dimensions, and duration. It should not have stabilization, cropping, or edits
that change geometry. The current learning workflow requires SDR Rec.709.

This learns time-dependent color curves, channel mixing, level shifts and
contrast from aligned pixel pairs. It is supervised regression, not a neural
network. The fitted transforms are stored in the project, so the reference video
is no longer required to render that project. A preset excludes analysis and
does not transfer these clip-specific fits to a different video.

```bash
python3 -m uwcc learn-reference tests/clips/C0201.MP4 \
  '/data/photo/SynDrive-Photo/Camera Backup/Jupiter - Emerald Charter/Video/corrected_C0201.MP4' \
  --profile rec709 --project /tmp/C0201-reference.uwcc-project.json
python3 -m uwcc render /tmp/C0201-reference.uwcc-project.json /tmp/C0201-learned.mp4
```

On four C0201 timestamps between training samples, the learned grade's mean
absolute RGB error against the user's reference is about **0.0052** on a 0–1
scale. This measures similarity to that personal grade, not physical color
accuracy. The reference and original have unconfirmed color signaling, so the
Rec.709 interpretation used in this comparison is provisional.

Detailed measurements are in the [C0201 upgrade report](benchmarks/C0201-ai-upgrade.md).

Comparison images, projects, metrics and videos are in
`benchmarks/results/ai-upgrade/` (local generated artifacts, ignored by git).

## Denoise and stabilization

**Use FastDVDnet AI denoise** selects the installed five-frame temporal neural
model. Set denoise strength above zero; 3 corresponds to a noise-map standard
deviation of 7.5/255. The model processes full-resolution overlapping tiles with
halos, retains the center of each tile, and repeats endpoint frames without
changing frame count or timestamps. Cancellation is checked between tiles.
The current neural denoiser supports Rec.709; log/HDR clips can use conventional
spatial/temporal denoise. Denoise remains export-only.

```bash
python3 -m uwcc render /tmp/C0201-reference.uwcc-project.json /tmp/C0201-ai-denoised.mp4 \
  --ai-denoise --denoise 3
```

**Full-resolution neural denoise is much slower on CPU than conventional
`hqdn3d`.** It is an explicit choice, not enabled by automatic color analysis.
Color workers cannot accelerate the sequential temporal denoiser that feeds
them. The UI now reports the actual CPU/GPU backend, neural frame and tile
progress, and encoded frame count. Render updates are throttled by elapsed time
rather than source fps, so a slow 100 fps export no longer hides 25 frames of work
behind a stale frame counter. Ready color results are drained without waiting
for the result queue to fill.

### GPU environment on this machine

The working RTX 3060 Ti (8 GiB, driver 580.178.04) has been tested with a separate
`.venv-gpu` environment, ONNX Runtime GPU 1.31.0, and pip-installed CUDA/cuDNN
libraries. The original `.venv-models` remains CPU-only. To use the GPU, save your
project, cancel the current CPU export, and relaunch from this checkout:

```bash
UWCC_INFERENCE_DEVICE=auto .venv-gpu/bin/python -m uwcc
```

An already running export cannot switch Python runtimes. GPU support here
accelerates neural inference; color processing and the current video codecs
still use CPU work. The progress display confirms `CUDA GPU · ONNX Runtime` when
the denoiser session actually has a CUDA provider. GPU initialization or inference
failures fall back explicitly to bounded CPU tiles.

The validated GPU FastDVDnet variant processes complete frames up to 1080p,
avoiding 240 separate network calls per frame. Larger images use 512-pixel GPU
tiles with halos. The GPU arena is limited to 4 GiB. A mixed FP16 graph keeps
float32 input/output; its validation against float32 measured a maximum error
of 0.00292 and mean error below 0.00025 on random, dark and bright test inputs.
Use `UWCC_GPU_PRECISION=float32` to select the validated float32 GPU graph, or
`UWCC_INFERENCE_DEVICE=cpu` to force the original CPU model.

On five-frame inputs from C0204, the app's warm 1080p denoising stage measured
about **0.38 seconds/frame**, versus about **1.03 seconds/frame** with the original
small-tile GPU path. CUDA kernel execution was verified with an ORT profile.
For 2,784 frames, that is roughly **18 minutes of neural denoising alone**;
full-export time also includes decoding, transfer conversion, color processing,
and encoding. These short warm-up measurements are not a completed full-clip
GPU export benchmark. Conventional denoise remains the faster option.

To prepare the same setup on a compatible NVIDIA system, keep GPU and CPU ORT
packages in separate environments:

```bash
python3 -m venv --system-site-packages .venv-gpu
.venv-gpu/bin/python -m pip install 'onnxruntime-gpu[cuda,cudnn]'
.venv-models/bin/python tools/prepare_gpu_denoise.py
.venv-gpu/bin/python tools/validate_gpu_denoise.py
```

The GPU validator must pass before it registers the GPU variants. Preparation
uses the pinned tensor weights downloaded by `tools/prepare_models.py`. The
application never creates or converts models during export.
See the official [CUDA provider setup](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
and [float16 conversion guidance](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

Stabilization now uses OpenCV feature tracking, forward/backward consistency
checks, and RANSAC to estimate background motion. Spatially distributed features
reduce domination by one moving diver. Select **Seabed / lower half** when the
sand remains visible, or **Whole image** for other framing. A smoothed camera
path suppresses shake while allowing pans. A proxy analysis pass determines
all stabilization transforms, then calculates the smallest single fixed zoom
that keeps every output frame inside the original image with a bilinear sampling
margin. The whole export uses that fixed crop: there are no mirrored/black fill
edges and no per-frame zoom pumping. The selected zoom is shown in export
progress. Framing is tighter and an extra analysis pass is required.
The RGB image is warped as floating point, avoiding the previous FFmpeg
stabilizer's 8-bit processing limitation.

This stabilizer is **geometric vision, not a neural stabilization model**.
On matched reliable tracks in the first two seconds of C0201, the background
acceleration diagnostic falls from 2.274 to 1.615 proxy pixels (about 29%).
Independent tracking success differs between variants; this conditional metric
is not a full-video quality certification. Parallax, rolling shutter, occlusion,
weak texture, and large camera moves remain limitations. Stabilization remains
export-only.

## Optional OpenAI / Claude grading advice

The **Cloud grading advice…** dialog asks for a provider and a vision-capable
API model ID. Pressing **Send frames and get advice** sends **three JPEG proxies,
up to 640 pixels**, after local automatic correction and before manual grading.
The full video, audio, local paths, and metadata are not uploaded. A returned
proposal is validated against the application's numeric controls and shown for
review. Applying it replaces manual controls/keyframes and is undoable.

Credentials come only from environment variables:

- OpenAI: `OPENAI_API_KEY`, using the Responses API with image inputs and a strict
  JSON grading schema; `store` is disabled in the request.
- Anthropic: `ANTHROPIC_API_KEY`, using Messages image inputs and a constrained
  grading tool response.

Enter a model ID available to your API account; the application does not assume
a specific model version. API keys are never saved in projects or presets.
Neither provider has been called with live credentials during this work;
request/response handling is covered by mocked provider tests. Sending requests
may incur charges under your provider account.

```bash
python3 -m uwcc advise /tmp/C0201-reference.uwcc-project.json \
  --provider openai --model YOUR_VISION_MODEL --allow-upload \
  --preference 'Natural blue water, brighter sand, vivid fish, preserve highlights' \
  --output /tmp/C0201-cloud-proposal.uwcc-project.json
```

Use `--provider anthropic` for Claude. Without `--allow-upload`, the CLI refuses
to send images. Advice is a grading proposal, not a pixel restoration model;
still frames cannot establish camera shake or temporal noise.

API implementation references:
[OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision),
[Anthropic vision](https://platform.claude.com/docs/en/build-with-claude/vision),
and [Anthropic tool definitions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools).

## Rebuild models on another machine

The application never downloads weights automatically. Model preparation is an
explicit setup step, separate from opening videos or projects:

```bash
python3 -m venv --system-site-packages .venv-models
.venv-models/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-models/bin/python -m pip install onnx onnxscript onnxruntime
.venv-models/bin/python tools/prepare_models.py
python3 setup.py build_ext --inplace
```

Install OpenCV (`python3-opencv` on Ubuntu, or the `vision` extra) if unavailable.
The converter fetches pinned source revisions and verifies the upstream weight
checksums, loads tensor state dictionaries with `weights_only=True`, exports
ONNX, and compares ONNX against PyTorch on test tensors. The running application
loads only ONNX artifacts, never PyTorch pickle files. Model manifests record
source revisions, weights/ONNX SHA-256 hashes and conversion error; adjacent
LICENSE files retain upstream notices. Set `UWCC_MODEL_DIR` to use a different
models directory.

| Model | Upstream | License | Runtime tested | Conversion maximum error |
| --- | --- | --- | --- | ---: |
| UDnet V1 | [official repository](https://github.com/alzayats/UDnet), [published weights](https://huggingface.co/Alzayats/UDnet) | MIT | ONNX Runtime CPU | 0.00000194 |
| FastDVDnet | [official repository](https://github.com/m-tassano/fastdvdnet) | MIT | OpenCV DNN CPU | 0.00000206 |

Darktable's [color calibration documentation](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/color-calibration/)
provided guidance on edge/surface illuminant estimation. Darktable is not being
loaded as a pretrained neural model. Deep White-Balance was investigated but
not bundled because its [published license](https://github.com/mahmoudnafifi/Deep_White_Balance/blob/master/LICENSE.md)
is non-commercial, contrary to this project's redistributable open-source policy.
