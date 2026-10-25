# Running UWCC

## Installation

Python 3.12 or later and FFmpeg/ffprobe on PATH are required.

On Ubuntu, the distribution-package route is:

```bash
sudo apt install python3-numpy python3-pil python3-pyqt6 python3-pyqt6.qtmultimedia ffmpeg
python3 -m uwcc
```

Run the second command from this project directory. Qt Multimedia is optional:
it enables source-audio preview; exported audio is always handled by FFmpeg.

With package-network access, a separate environment can use the preferred
PySide6 binding:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m uwcc
```

The implementation also works with system PyQt6. An empty `.venv` created
without system packages cannot see those distribution packages; use `python3`
for the distribution-package route. Windows users can install FFmpeg on PATH
and use the same Python package, but a Windows build has not been validated.

For optional local AI, install one of the `ml`, `ml-nvidia`, or `ml-amd` extras.
The [GPU setup guide](gpu-setup.md) covers NVIDIA CUDA on Linux/Windows,
AMD DirectML on Windows, CPU fallback, and runtime verification. AMD GPU
inference on Linux is not currently supported by UWCC.

## First correction

1. Choose **Open video** or drop a video on the workspace.
2. Confirm the input profile. Select Rec.709 for standard footage or the exact
   Sony log profile used in the camera. UWCC does not infer a Sony picture
   profile from the camera model or filename.
3. Set the capture-filter hint if you used a red or magenta filter.
4. Choose **Auto correct**. The current engine, confidence, and signal warnings
   appear in the inspector. The supplied classical engine needs no model.
5. Drag the comparison divider, scrub the timeline, and adjust strength and
   finishing controls. A 0% strength disables automatic correction while keeping
   manual controls active. **Original** bypasses all corrections except input
   interpretation and the display transform.
6. Choose the output color intent and export format, then **Export**.

SDR displays always receive a tone-mapped SDR preview of HDR/log material. The
output-intent selector controls file rendering, not the monitor. For HDR output,
choose HEVC or FFV1. DNxHR and ProRes are SDR master profiles in this build.

Zoom refers to proxy pixels. Drag with the right or middle mouse button to pan.
Playback uses a 640-pixel proxy; paused preview uses up to 960 pixels. Large
exports use source resolution and may be substantially slower than real time.

## Export performance and video finishing

The optional C extension accelerates tetrahedral LUT interpolation and fuses SDR
tone mapping, gamut compression, and transfer encoding. It releases Python's GIL
and uses the same color graph as the NumPy fallback. Neutral manual controls skip
unnecessary calculations; automatic gains stay in float32. H.264/HEVC use the
`fast` encoding preset at the previous CRF settings (17/18), with at most eight
encoder threads. Files may be larger than with the previous `medium` preset.

Package installation builds the extension when a compiler and Python development
headers are available. For the distribution-package/source checkout workflow:

```bash
sudo apt install build-essential python3-dev python3-setuptools
python3 setup.py build_ext --inplace
python3 -m benchmarks.export
```

Build using the same Python interpreter used to launch UWCC. Without a compiled
extension the application uses NumPy automatically. To compare complete exports,
run `python3 -m benchmarks.export --project dive.uwcc-project.json`; this creates
temporary H.264 files and removes them afterward. Microbenchmarks isolate the
kernels; actual export speed also depends on resolution, controls, codecs, and
the source storage.

To benchmark a camera reference clip, including each finishing effect separately:

```bash
python3 -m benchmarks.export --clip tests/clips/C0201.MP4 --input-profile rec709 \
  --finishing --output-dir /tmp/uwcc-reference-exports \
  --report /tmp/uwcc-reference-report.json
```

Set `--input-profile` to the camera's actual recording profile. Without that
option the runner uses tagged metadata and rejects ambiguous input. Analysis
runs once; the same analysis is reused for NumPy/C comparisons. The finishing
runs use the C backend with stabilization, denoise at strength 3, and both.
Every export includes timing/audio validation and reports elapsed seconds,
frames per second, and speed relative to real time. The output folder retains
videos for visual inspection; use a fresh folder for each run. The JSON report
records source dimensions, frame count, fingerprint, and individual results.
Use `--backend native` or `--backend numpy` to restrict complete exports to one
backend. See the [C0201 camera benchmark](benchmarks/C0201.md) for measured real
footage results and the limits of the current stabilization implementation.

**Stabilize video on export** uses robust OpenCV background tracking, with a
seabed/lower-half or whole-image reference. An analysis pass chooses a single
fixed crop/zoom large enough to keep every frame inside the source image, so
no mirrored or exposed black borders appear and the zoom does not pump.
Tracking uses a small proxy; the image warp preserves float RGB precision.
OpenCV is required.

**Denoise on export** uses FFmpeg's native `hqdn3d` by default. Zero disables it;
start at 2–4 on the 0–10 scale. **Use FastDVDnet AI denoise** selects the installed
five-frame neural model instead. It is much slower on CPU. Both effects are
export-only and are saved with projects, presets, and batch jobs.

The [AI workflow guide](ai-workflow.md) describes local neural enhancement,
automatic levels/white balance, learning a corrected reference, AI denoise,
and optional OpenAI/Claude advice.

### Parallel export

Decode, timestamp validation, filtering, and encoding now have bounded native
thread pools (up to eight codec threads and four filter threads by default).
Independent color frames run concurrently in a bounded worker pool; temporal
stabilization and denoise run in order before that pool. Output frames are
written in their original order.

```bash
python3 -m uwcc render project.json output.mp4 --workers 3
```

`--workers` sets the requested number of concurrent color frames. The default
requests up to four and caps that number using a 768 MiB estimated working
memory budget (normally three workers at 1080p). Higher resolutions reduce the
worker count. The estimate is not a hard process RSS limit. Neural denoise can
allocate additional memory for its temporal window and network.

Environment settings are `UWCC_THREADS` (1–32, bounded by CPU affinity),
`UWCC_RENDER_WORKERS` (1–16), and `UWCC_RENDER_MEMORY_MB` (at least 64).
`UWCC_THREADS=1` and `--workers 1` provide a serial diagnostic. Some FFmpeg
filters and the temporal tracking path inherently require sequential work;
more encoder threads alone cannot remove a color-processing bottleneck.

## Animation and presets

Press the diamond next to a control to capture its current value at the
playhead. Add another keyframe later to interpolate between them. Editing a
control that already has animation adds or updates a keyframe at the playhead.
Outside its keyframes, the control holds the nearest keyframe value.

Choose a keyframe in the list below the timeline to jump to it. Use **Delete
selected keyframe** to remove it. The reset arrow removes that control's
animation and restores its neutral value. Ctrl+Z and Ctrl+Shift+Z undo and redo.

Presets store settings and input/output intent, but exclude source paths,
automatic curves, and manual animation. Batch items each receive their own
automatic analysis. Choose the required input profile and export format before
adding files to a batch. Pause stops the current item safely; resume restarts
that item, leaving completed outputs alone.

**Load preset…** opens a file chooser even before a video is open. In that case,
the selected settings are applied to the next video you open. **Save preset…**
becomes available once a video is open.

## Projects and recovery

Ctrl+S saves a `.uwcc-project.json` file beside the source by default. The project
contains the editing settings and generated curve, not the original video.
Recovery state is saved after edits settle for two seconds in the user's data
directory (`~/.local/share/uwcc` by default on Linux). On next launch, UWCC
offers to recover that session.

A source fingerprint prevents stale analysis from being applied to a changed
video. To relink a moved or deliberately replaced source:

```bash
python3 -m uwcc relink dive.uwcc-project.json /new/path/dive.mp4
```

For changed media the command preserves manual edits and discards old analysis.
Reopen the project and run Auto correct again.

## Command line

```bash
python3 -m uwcc probe dive.mp4
python3 -m uwcc analyze dive.mp4 --profile rec709 --filter red --project dive.uwcc-project.json
python3 -m uwcc render dive.uwcc-project.json dive_corrected.mp4 --format h264
python3 -m uwcc render dive.uwcc-project.json dive_clean.mp4 --stabilize --denoise 3
python3 -m uwcc batch dive1.mp4 dive2.mp4 --profile rec709 --output-dir corrected --format prores
```

Render formats are `h264`, `hevc`, `dnxhr`, `prores`, and `ffv1`. To render HDR,
analyze with `--output-profile hlg` or `--output-profile pq`, then use `hevc` or
`ffv1`. Existing files require explicit `--overwrite` in the render command.
Ctrl+C requests cancellation. Input paths may contain spaces and non-ASCII text.
`analyze` and `batch` also accept `--stabilize` and `--denoise 0–10`.
Omitting these options retains the settings saved in the project or preset.

## Optional ONNX model

Install `.[ml]` in an environment with network access, then select a local
manifest in the GUI or pass `--model manifest.json` to `analyze`. UWCC performs
no downloads. A model is optional and no model is bundled.

The interface accepts a single float32 NCHW RGB tensor in encoded sRGB, in the
range 0–1, and returns a single corrected tensor with the same shape. A model
with different normalization, inputs, or output semantics must be converted and
validated before use. Merely renaming a checkpoint is not sufficient.

Put the ONNX file and a manifest beside one another. Example manifest:

```json
{
  "schema_version": 1,
  "model_id": "your-verified-underwater-model",
  "model_version": "1.0.0",
  "file": "model.onnx",
  "onnx_sha256": "replace-with-the-sha256-of-the-onnx-file",
  "code_license": "MIT",
  "weights_license": "MIT",
  "color_space": "srgb",
  "layout": "NCHW",
  "size": 256
}
```

The licenses must describe the actual code and weights; a manifest declaration
does not establish licensing rights. Generate the checksum with `sha256sum
model.onnx`. Raw PyTorch pickle checkpoints are not loaded. Model pixels are
used only to fit a smooth global correction; generated detail never replaces
source video pixels.

## Tests and implementation coverage

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen python3 -m pytest -q
```

FFmpeg with x264, x265, DNxHR, ProRes, and FFV1 encoders must be installed for the
media tests. The GUI test runs offscreen and is skipped if neither Qt binding is
installed. See [implementation status](implementation-status.md) for the
remaining v1 release work and [the bootstrap decision](adr-001-bootstrap.md) for
the dependency choices made in this environment.
