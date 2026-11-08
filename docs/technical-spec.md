# UWCC Technical Specification

Status: Current architecture and v1 verification targets

Implementation baseline: 0.4.3

Last reviewed: 2026-11-08

Sections 1–11 describe the current code unless explicitly marked as a target or
limitation. Section 12 retains the intended v1 verification gates; it does not
claim that those gates have passed. Product requirements and remaining gaps live
in the [product specification](product-spec.md#11-current-coverage-and-open-v1-work).
[ADR 001](adr-001-bootstrap.md) records the original bootstrap decisions and is
historical, rather than an inventory of today's installed dependencies.

## 1. Architectural goals

UWCC separates input interpretation, generated automatic correction, manual
finishing, and output intent. Preview and export share correction code; preview
uses smaller images and an SDR display transform. Local neural color models
provide reference grades that are fitted to global color transforms. Their
reference pixels do not replace full-resolution source detail.

Full-resolution FastDVDnet denoise is a separate, optional export effect.
Stabilization is also export-only. Ordinary processing is local; explicitly
requested model/runtime downloads and cloud advice are separate network paths.

## 2. Technology baseline

| Concern | Current implementation |
| --- | --- |
| Language | CPython 3.12 or later |
| UI | PySide6, with PyQt6 fallback; Qt Widgets and QPainter |
| Numeric processing | NumPy float32; optional compiled C kernels |
| Color transforms | Transfer functions and gamut matrices in `uwcc/color.py`; ACEScg scene-linear path and an optimized encoded Rec.709 LUT path |
| Media I/O | Bounded FFmpeg/ffprobe subprocesses, argument arrays, no shell interpolation |
| Vision | OpenCV for stabilization and optional DNN inference |
| Neural inference | ONNX Runtime; CPU, CUDA, or Windows DirectML where supported |
| Model conversion | Separate user-owned Python environment with PyTorch, ONNX, OpenCV, and ONNX Runtime |
| Packaging | Ubuntu 26.04 Debian packages; PyInstaller Windows bundles and MSIX |
| Tests | pytest; offscreen Qt and real FFmpeg round trips |

PyAV, Qt Quick/QRhi, an embedded OpenColorIO configuration, and AppImage tooling
are not current dependencies. Reviewed display management, dependency locks,
SBOMs, and complete binary-license provenance remain release work. Optional
runtime installation downloads dependencies; it does not modify the application's
own Python environment. Consult `pyproject.toml`, Debian metadata, and the build
scripts for the actual dependency declarations.

## 3. Components and concurrency

- `gui.py` owns the Qt Widgets workspace, worker threads, proxy playback, model
  setup dialog, preview scopes, and undo/recovery interactions. Long operations
  use `QThread` workers; analysis is not isolated in a separate worker process.
- `media.py` probes sources, prepares orientation/range, reads bounded 16-bit RGB
  pipes, and extracts source timestamps through FFmpeg/ffprobe. Frame reads use
  `readinto` into reusable uint16 scratch storage; yielded float frames own their
  buffers. Windows raw-frame pipes have dedicated buffers capped at 1 MiB;
  other platforms keep normal subprocess pipes.
- `color.py`, `analysis.py`, and `enhance.py` share correction evaluation between
  preview and export. `neural.py` and `ml.py` validate and run optional models.
- `render.py` performs full-resolution rendering, ordered parallel color work,
  audio muxing, output validation, and atomic publication.
- `project.py` owns editing state, validated JSON, fingerprints, and presets.
  `batch` CLI/GUI operations use the persistent `BatchQueue` in `jobs.py`.
- `model_setup.py` owns explicit model/runtime preparation; `execution.py`
  persists the chosen runtime and activates it at the next application start.
- `cloud.py` sends explicitly requested proxy-frame advice requests and validates
  bounded parameter proposals; the GUI asks the user to review a proposal.

Native-library crashes are not currently isolated from the GUI process. A global
memory-pressure manager, disk proxy cache, and concurrent-render preference are
not implemented.

## 4. Media, orientation, and color

Supported profile identifiers are `auto`, `rec709`, `slog2`, `slog3`, `hlg`, and
`pq`; exact labels are defined by `PROFILES` in `project.py`. Standard stream tags
and supported embedded Sony gamma/gamut metadata supply detection evidence.
Explicit Sony capture metadata takes precedence over generic tags. Ambiguous
input requires a user choice before analysis/export. See
[profile detection](profile-detection.md).

Decoded RGB uses 16-bit transport into floating-point processing. The scene-linear
working space is ACEScg; SDR analysis can instead produce encoded Rec.709 LUTs.
Manual controls include Bradford temperature/tint adaptation, scene-linear
exposure, luminance tone adjustments, chroma-plane hue/saturation, bounded local
contrast, and output sharpening. This is not an OCIO-managed or calibrated
monitor pipeline. Tone adjustments are not the originally proposed single
validated monotonic spline. Deterministic timestamp-seeded dither is not an
implemented export contract.

The preview always targets SDR Rec.709, including a tone map for log/HDR input.
Output intent is independent and defaults to Rec.709; HDR input does not
automatically imply HDR export. HLG/PQ export requires HEVC or FFV1. DNxHR and
ProRes are SDR master profiles in the current build. HDR monitor output and
comprehensive log/HDR color validation remain open.

Orientation honors source rotation metadata. Additional manual clockwise
rotation is `0`, `90`, `180`, or `270`; `auto` can use the optional CPU RightWayUp
Max INT8 model when metadata does not resolve orientation. Ten sampled frames
produce one conservative decision, persisted in the project. Rotation geometry
is shared by preview, analysis, finishing, and export. See
[orientation evaluation](benchmarks/orientation.md).

## 5. Automatic correction and optional models

Analysis decodes a maximum-width 320-pixel proxy, detects cuts using histogram
and frame differences, samples approximately every half second plus shot
boundaries, and smooths generated parameters forward/backward within each shot.
It reports progress but publishes a complete curve at completion; progressive
partial-curve publication and calibrated confidence thresholds remain targets.

The three automatic modes are adaptive levels/gray-edge white balance (`adaptive`,
the default), local neural reference enhancement (`ai`), and legacy balance.
UDnet V1 is the prepared enhancement model; DICAM is not the selected model.
SDR neural references use regularized polynomial color fitting and temporally
interpolated RGB LUTs. The log/HDR path retains constrained scene-linear matrix
fitting. Learned references require aligned SDR Rec.709 source/reference clips;
the fitted look is saved inline and is specific to that source.

Low red-channel signal and clipping constrain correction and produce warnings.
Confidence is heuristic, not a calibrated measurement of physical color accuracy.
Missing built-in AI can fall back with a visible warning; explicitly selected
invalid model manifests fail instead of being silently substituted.

Models are optional and are not bundled in application packages. Explicit setup
downloads pinned upstream weights, checks SHA-256, and converts locally where
needed. UDnet/FastDVDnet use MIT declarations; RightWayUp declares Apache-2.0.
Code/weight declarations must match actual provenance; a manifest alone does
not establish redistribution rights. A reference manifest contains
`schema_version`, `model_id`, `model_version`, `file`, `onnx_sha256`,
`code_license`, `weights_license`, `color_space`, `layout`, and `size`.
The reference interface is float32 NCHW RGB, encoded sRGB in [0,1], returning
corrected RGB of matching shape. See [model setup](../models/README.md) and
[the user-guide manifest example](user-guide.md#optional-onnx-model).

## 6. Preview and full-resolution finishing

Paused previews use up to 960 pixels; playback uses a 640-pixel proxy. Original
Rec.709 display bypasses a redundant color conversion; log/HDR originals retain
the SDR display transform. Playback follows a monotonic clock and starts optional
Qt Multimedia source audio. Overdue frames can be skipped before correction to
avoid accumulating delay. There is no automatic proxy-resolution adaptation or
claim of audio-clock-authoritative synchronization.

Pointwise preview correction uses up to four persistent CPU workers. Dehaze and
sharpness stay on the full proxy frame to preserve spatial filtering. Preview
decode and timestamp pools are capped at four threads. Scopes are calculated
from displayed proxy images. See [preview performance](preview-performance.md)
for historical measurements and their limits; full-source-rate playback and
visible-window/audio synchronization are not certified.

Export-only stabilization uses OpenCV background motion estimation, a
seabed/lower-half or whole-frame region, and a fixed crop/zoom selected in a first
pass. Each spatial cell's feature detector receives a cropped proxy region plus
a five-pixel neighborhood halo, and points are restored to full-proxy
coordinates. This avoids repeated full-proxy corner-response calculations while
retaining the existing detector parameters and per-cell feature quotas.
Conventional denoise uses FFmpeg `hqdn3d`; optional FastDVDnet uses a
five-frame window, tiles/halos, and repeated endpoint frames. Neural denoise
currently requires Rec.709. A single future-window preparation worker overlaps
decode/conversion with sequential inference. See [AI workflows](ai-workflow.md).

## 7. Persistence contracts

`project.py` is the serializer authority. Current projects use schema 1, not the
original planned v1 document with `manual`, `color`, `input`, or external NPZ
analysis-cache fields. The top-level shape is:

```json
{
  "schema_version": 1,
  "application_version": "0.4.3",
  "source": "relative/or/absolute/source.mp4",
  "source_fingerprint": {
    "size": 0,
    "mtime_ns": 0,
    "first_4m_sha256": "<sha256>",
    "last_4m_sha256": "<sha256>"
  },
  "grade": {
    "values": {"strength": 100},
    "keyframes": {"strength": [{"time": "1/2", "value": 80}]},
    "input_profile": "rec709",
    "output_profile": "rec709",
    "filter_hint": "auto",
    "auto_method": "adaptive"
  },
  "analysis": null,
  "orientation": null
}
```

This shows structure, not a ready-to-load project: fingerprints must match a real
source. Missing grade fields receive validated defaults. Grade also stores
stabilization, denoise engine/strength, rotation, and video encoder/decoder
choices. Keyframe timestamps are rational strings, not numerator/denominator
objects or UUID records. Analysis is an optional validated inline object with
`version: 1`; its sample times are numeric seconds. Manual keyframes interpolate
linearly and hold endpoint values outside their interval.

JSON reads reject duplicate keys, non-finite constants, unsupported schema
versions, oversized files (32 MiB), and invalid values/keyframes. Writes use a
temporary sibling, flush/fsync, and atomic replacement. Unknown extension fields
are not guaranteed to round-trip. Fingerprints use size, mtime, and first/last
4 MiB hashes. Explicit source replacement invalidates analysis/orientation when
the fingerprint differs. Source relinking is available via CLI; a complete GUI
relink flow remains open. Imported local model paths are not automatically
trusted; saved analysis remains usable without the original model.

Presets use `schema_version: 1`, `kind: uwcc-preset`, `name`, and `grade`.
Saving clears keyframes and the selected enhancement manifest; presets do not
contain source paths or automatic analysis. Queue state and recovery data live
in the user data directory. Recovery autosaves follow a two-second edit delay.
Batch pause cancels the current operation; resume restarts that item, rather
than continuing from a frame checkpoint. See [the user guide](user-guide.md).

## 8. Export and acceleration

| Profile | Video | Audio |
| --- | --- | --- |
| `h264` | MP4, libx264, CRF 17, fast, yuv420p | AAC; compatible AAC can be copied |
| `hevc` | MP4, libx265, CRF 18, fast, yuv420p10le, hvc1 | AAC; compatible AAC can be copied |
| `dnxhr` | MOV, DNxHR HQX, yuv422p10le | PCM signed 24-bit |
| `prores` | MOV, prores_ks profile 3, yuv422p10le | PCM signed 24-bit |
| `ffv1` | Matroska, FFV1 level 3, gbrp16le | FLAC |

CPU encoding/decoding are defaults. Optional runtime-tested NVENC H.264/HEVC and
NVDEC are explicit video-codec choices, independent of the AI execution runtime.
GPU decode downloads frames into the CPU pipeline. Fallback behavior depends on
the selected codec policy. See [video acceleration](video-acceleration.md).

Rendering evaluates grades against source timestamps and attempts to retain
variable cadence and audio offset. Before publication, it validates frame count,
duration, audio-stream count, and color tags. Exact final-packet duration on
unusual variable-rate sources and stream-level audio-offset validation remain
open. The FFmpeg 9.0 removal of `-filter_script:v` caused the historical Windows
VFR failures; the current VFR path uses the long-supported inline `-vf` graph
and a separate timestamp command file in the encoder's working directory.
This avoids version-specific filter-script options and path-escaping problems.
The implemented
checks do not establish all of section 12's media gates.

Temporary work is isolated and a validated output is published atomically.
Existing output requires explicit overwrite authorization. Metadata is copied
where representable, location fields are cleared, and rotation tags are reset
after pixel rotation. A configurable private-metadata UI, complete unknown-stream
reporting, and free-space estimation remain targets.

## 9. Resource limits and performance

Color export requests up to four workers, bounded by CPU affinity and a 768 MiB
estimated color-working budget (`UWCC_RENDER_MEMORY_MB`). This is not a hard
process RSS limit and excludes additional neural-window/network allocations.
`UWCC_RENDER_WORKERS` allows 1–16; `UWCC_THREADS` allows 1–32 within affinity.
Codec pools normally use up to eight threads and filters up to four.
Temporal processing precedes the ordered color-worker pool. The optional C
extension accelerates tetrahedral LUT evaluation and fused SDR conversion;
the unreleased export path also fuses encoded Rec.709 LUT interpolation,
strength blending, clamping, and uint16 quantization when manual controls are
neutral. It selects AVX2 by default only after CPU AVX2/XSAVE and OS XMM/YMM
state checks. A separately compiled AVX2 object is excluded from cross-object
optimization; baseline native code stays compatible with non-AVX2 CPUs. The
portable fused scalar path handles unsupported CPUs and vector tails. Non-x86
targets build the scalar implementation. Manual finishing, log/HDR, zero
strength, and missing/non-LUT analysis retain the original float graph.
NumPy is the fallback. Its LUT interpolation uses blocks of at most 65,536
pixels, float32 fractions/results, and direct tetrahedron-corner selection
instead of full-frame sorting and float64 intermediates. Decoder normalization
and output rounding also reuse their temporary arrays. These changes retain
the color graph and encoder settings; measured differences are documented in
[the C0045 report](benchmarks/C0045-export.md).
`UWCC_DENOISE_PREFETCH=0` disables denoise prefetch.

Repository benchmarks describe specific clips/machines, not general guarantees.
Global RAM/disk budgets, deadline guarantees for all cancellation paths, GPU
color shaders, and certified preview throughput remain targets.

## 10. Platforms, packaging, and runtime setup

Ubuntu 26.04 amd64 uses `.deb` packages (`uwcc`, optional `uwcc-model-tools`,
`uwcc-nvidia`) and PPA installation. Windows uses a PyInstaller onedir bundle,
a windowed `uwcc-gui.exe`, a console `uwcc.exe`, and Microsoft Store MSIX
packaging. Windows Store distribution is current, rather than a deferred port.
See [Ubuntu packaging](ubuntu-packaging.md) and
[Windows packaging](windows-packaging.md).

User data uses XDG directories on Linux and `%LOCALAPPDATA%\uwcc` on Windows.
Models default to the `models` subdirectory, with `UWCC_MODEL_DIR` override.
Private converter runtimes live below `runtimes`; frozen Windows setup downloads
an official embeddable Python. CPU/CUDA/DirectML selections are persisted and
need an application restart. DirectML conversion uses CPU PyTorch; CUDA uses
CUDA-enabled PyTorch. Drivers remain user-installed prerequisites. Setup is
intended to work as a standard user, without writing into the installed package.

Since 0.4.3, missing-pip recovery detects embedded runtimes by their
`python*._pth` files, enables `import site`, and uses the same isolated
`get-pip.py` bootstrap as initial setup. Embedded Python does not provide
`ensurepip`; regular virtual environments retain that repair path. Temporary
bootstrap scripts are removed after success, failure, or cancellation.

The 2026-11-08 standard-user Store report shows converter `python.exe` startup
failure `0xC0000022`. The 0.4.2 DLL-search isolation patch still requires
a rebuilt-package retest on the affected account; the root cause is not confirmed.
Do not treat packaging smoke tests as proof of successful standard-user model
bootstrap or hardware-validated DirectML.

## 11. Privacy, security, and release evidence

Ordinary correction, playback, and export run offline and require no UWCC account.
Startup may offer model setup, but a download requires explicit user action.
Model/runtime setup contacts upstream hosts. Optional cloud advice sends up to
three downscaled JPEG proxy frames and a grading request directly to the user's
chosen OpenAI/Anthropic endpoint with an environment-provided API key. Proposals
are range/structure-validated; credentials are excluded from projects/presets.
See [privacy policy](../PRIVACY.md) for the user-facing policy.

Current CI runs on pushes: Ubuntu tests, Debian build/install smoke tests, and
Windows packaging/model/runtime unit tests plus frozen-bundle smoke tests. Tags
publish package artifacts after those jobs. Scheduled GPU/corpus benchmarking,
full Windows test-suite certification, mandatory lint/type checking, SBOM/license
automation, fuzzing, and manual release sign-off are targets, not existing CI
coverage. Existing changelog test results are historical evidence, not fresh
verification of the current working tree.

## 12. v1 verification targets (not a certification of 0.4.3)

### 12.1 Test corpus

The repository shall maintain manifests and redistribution terms for a golden
corpus containing:

- Sony Alpha 6300 and 6500 XAVC S in HD and 4K;
- no-filter, red-filter, and magenta/green-water examples;
- standard gamma, S-Log2, and S-Log3 footage;
- separately sourced HLG and PQ examples;
- daylight, artificial light, mixed light, low red signal, red clipping, marine
  snow, haze, skin, coral, sand, gray cards, and color charts;
- static, panning, rapid motion, gradual depth/lighting change, and hard cuts;
- silent clips, mono/stereo audio, audio offsets, rotation, and variable frame
  rate; and
- deliberately corrupt, truncated, unsupported, and metadata-conflicting files.

Private user footage may inform development but shall not enter CI or a release
artifact without an explicit redistributable grant.

### 12.2 Unit and property tests

Tests shall cover transfer functions, matrix transforms, gamut compression,
LUT identity/interpolation, monotonic tone curves, parameter ranges, keyframe
interpolation, rational timestamp math, dither repeatability, fingerprints,
schema validation/migration, atomic writes, command construction, scene reset,
confidence thresholds, fallback selection, and cancel behavior.

Property tests shall assert finite outputs, neutral-axis preservation where
applicable, monotonic luminance, bounded parameter evaluation, and no path
escape across generated inputs.

### 12.3 Image-quality gates

On the color-chart subset, automatic correction at 100% shall:

- improve median CIEDE2000 by at least 25% against the color-managed source;
- not worsen any clip's median CIEDE2000 by more than 10%;
- add clipped values to no more than 0.1% of pixels per frame, excluding pixels
  already clipped in the source; and
- keep neutral patches within Delta E 3 of the expert-approved reference when
  source channel signal passes the documented minimum.

On non-chart material, a blinded panel of at least three reviewers shall prefer
or rate the Natural result equivalent to the uncorrected source in at least 80%
of clips, with no critical artificial-color or flicker finding. UIQM and UCIQE
shall be reported for regression visibility but shall not independently pass or
fail a release.

### 12.4 Temporal gates

For static tracked regions without real illumination change, the 95th
percentile adjacent-frame mean Lab difference introduced by correction shall be
below Delta E 1.5. No LUT interpolation may cross a confirmed cut. The first
sample after a cut shall be independent of the previous shot. Generated curves
shall reproduce identically on CPU across two runs within `1e-5` per LUT value.

Motion sequences shall be reviewed for pumping, trailing color, and cut flashes.
Any visible one-frame discontinuity attributable to the algorithm is a release
blocker even when the aggregate metric passes.

### 12.5 Media round-trip gates

For every normative export profile:

- output duration differs by no more than one source-frame interval;
- constant-rate frame count is exact;
- first-audio timestamp differs by less than 20 ms;
- orientation, aspect, range, primaries, transfer, and matrix tags match intent;
- first, middle, and final frames decode successfully;
- SDR values remain legal after quantization; and
- HDR output passes an independent FFmpeg probe and reference-player metadata
  inspection.

### 12.6 Workflow tests

End-to-end tests shall exercise open, profile confirmation, analyze, progressive
preview, low-confidence fallback, manual-only correction, keyframes, undo/redo,
save/reopen, crash recovery, source relink, preset round-trip, independent batch
analysis, pause/resume/cancel, retry, output collision, disk full, missing model,
GPU failure with CPU retry, and validated atomic output completion.

### 12.7 CI and release gates

The target CI policy is to run linting, type checks, unit tests, schema fixtures,
CPU golden frames, and short media round trips. Scheduled CI shall run the full
corpus, GPU/CPU comparison, performance benchmarks, fuzzed parsers, and Windows
portability tests. A release candidate additionally requires clean Ubuntu 26.04
installation testing, offline runtime observation, SBOM/license approval,
package malware scanning, and manual visual review sign-off.

Numeric golden comparisons shall state their color space and use per-test
tolerances. Updating a golden result requires a reviewed explanation and before/
after artifacts; accepting all new output wholesale is forbidden.

