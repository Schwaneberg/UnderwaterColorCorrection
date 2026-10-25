# UWCC Technical Specification

Status: Draft for implementation  
Version: 0.1  
Last updated: 2026-10-22

## 1. Architectural goals

The implementation shall optimize for predictable color, temporal stability,
recoverability, and inspectability rather than opaque enhancement. ML proposes
the correction; deterministic, color-managed code applies it at full
resolution. The same correction graph shall drive still preview frames,
playback, scopes, and final rendering.

The architecture shall enforce these boundaries:

1. Media I/O never decides color intent.
2. Input and output color transforms are explicit and independently testable.
3. ML inference never writes full-resolution output frames directly.
4. Automatic analysis and manual finishing remain separate persisted layers.
5. Preview quality may vary, but correction semantics may not.
6. A failure in optional acceleration falls back without changing the project.
7. User media and generated pixels never leave the machine.

## 2. Technology baseline

The reference implementation shall use:

| Concern | Required technology | License constraint |
| --- | --- | --- |
| Language | CPython 3.12 | PSF-compatible |
| UI | PySide6 and Qt Quick/QRhi | LGPL/GPL-compatible, dynamically linked where required |
| Numeric processing | NumPy | BSD-compatible |
| Image analysis | OpenCV | Apache-2.0-compatible |
| Color management | OpenColorIO with an embedded reviewed config | BSD-compatible |
| Media binding | PyAV | BSD-compatible |
| Media backend | FFmpeg built with GPL options | GPL-compatible components only |
| ML inference | ONNX Runtime | MIT-compatible |
| Model exchange | ONNX | Open specification and tooling |
| Packaging | PyInstaller plus AppImage tooling | GPL-compatible |
| Test runner | pytest | MIT-compatible |

PyTorch may be used in the developer-only model conversion and validation
environment, but shall not be part of the runtime package. Runtime version
ranges shall be locked with hashes. The lockfile, FFmpeg configuration string,
OpenColorIO config hash, model hash, and build tool versions shall be captured
in release provenance.

An implementation may replace a listed library only through an architecture
decision record demonstrating equivalent behavior, platform support, licensing,
and test coverage.

## 3. Logical components

### 3.1 Application shell

Owns window lifecycle, commands, undo/redo, recent projects, preferences,
autosave, and recovery. It communicates with long-running services through
typed messages and shall not perform decoding or inference on the UI thread.

### 3.2 Media service

`MediaService` shall:

- probe streams and metadata;
- expose decoded video frames with exact presentation timestamps;
- expose audio to the playback path;
- seek to the nearest required frame and identify approximate seeks;
- decode proxy or full-resolution frames; and
- mux encoded video, audio, and allowed metadata.

PyAV shall be used for interactive decode and timestamp control. Export may use
PyAV or a supervised FFmpeg subprocess, but command construction shall use
argument arrays rather than a shell and shall be captured in diagnostic logs
with paths redacted by default.

### 3.3 Color service

`ColorService` shall own input transforms, the scene-linear working space,
display/output transforms, gamut mapping, and metadata tags. The working space
shall be ACEScg linear floating point. Processing shall use float32; a preview
shader may use float16 intermediates only if golden-image tolerances pass.

OpenColorIO shall perform standardized transforms. Sony log transforms and HDR
transforms shall be validated against published transfer-function test vectors.
The embedded configuration shall be versioned and hashed in every project.

### 3.4 Analysis service

`AnalysisService` shall orchestrate scene detection, sampling, filter
classification, ML inference, transform fitting, confidence evaluation,
temporal stabilization, cache writing, and cancellation.

Analysis shall run in a worker process so a native-library failure or GPU reset
does not terminate the UI. Communication shall contain metadata, shared-memory
frame handles, progress events, and immutable result records. The worker shall
have no network capability configured by the application.

### 3.5 Correction graph

`CorrectionGraph` shall evaluate these stages in order:

1. decode range normalization;
2. input transform to ACEScg;
3. automatic time-varying transform blended by correction strength;
4. temperature and tint;
5. exposure;
6. black point, shadows, highlights, and contrast tone curve;
7. hue and saturation;
8. bounded dehaze;
9. gamut compression;
10. output transform;
11. output-referred edge-limited sharpening; and
12. legal-range conversion and quantization/dither for encode.

Each node shall be deterministic for the same inputs and configuration. Preview
and render nodes shall share parameter evaluation code. GPU shaders shall be
tested against the CPU reference and shall not define independent semantics.

### 3.6 Preview service

`PreviewService` shall coordinate decode, correction evaluation, audio clock,
frame queueing, viewer rendering, scopes, and proxy selection. Qt's QRhi layer
shall target Vulkan or OpenGL on Ubuntu and permit a D3D backend later on
Windows. Unsupported shader features shall fall back to the CPU correction
graph.

The audio clock is authoritative during playback. The preview may drop late
video frames but shall record a dropped-frame counter and lower proxy resolution
after five consecutive seconds below 90% of requested frame rate.

### 3.7 Render queue

`RenderQueue` shall persist jobs before starting them, isolate each output in a
worker process, validate preconditions, write to a `.partial` target, and rename
atomically after validation. Only one software render shall run by default;
users may increase concurrency in preferences. Analysis and rendering shall
honor a shared memory-pressure limit.

## 4. Media and color contract

### 4.1 Required input support

The acceptance suite shall cover:

- MP4/MOV containing H.264/AVC, including Sony XAVC S variants;
- 1920×1080 and 3840×2160 progressive video;
- 23.976, 24, 25, 29.97, 30, 50, 59.94, and 60 fps;
- 8-bit 4:2:0 and 8/10-bit 4:2:2 decode where supported by FFmpeg;
- AAC and uncompressed PCM audio; and
- files with rotation, non-square pixel, full/limited-range, and variable frame
  rate metadata.

Other FFmpeg-decodable sources may open as best effort but are not v1 release
requirements. Variable frame rate shall remain variable unless the user
explicitly selects a constant-frame-rate export.

### 4.2 Input-profile resolution

Resolution order shall be:

1. explicit project override;
2. unambiguous standardized stream/container metadata;
3. a remembered decision for the same verified source fingerprint; and
4. user confirmation.

Filename, camera model, and image statistics may suggest a selection but shall
not silently decide a log or HDR profile. An export job with an unresolved
profile shall fail validation before rendering.

### 4.3 Working and display spaces

Decoded Y'CbCr shall be expanded using its declared matrix, primaries, transfer,
and range. Missing standard-profile metadata shall default to limited-range
Rec.709 only after presenting that assumption in the inspector. Log input never
uses that default.

ACEScg shall be the working space. The display transform shall match the user's
selected display profile when supported; otherwise it shall target sRGB/Rec.709
and disclose the fallback. HDR preview is optional in v1 even though HDR input
and output processing is required. On an SDR display, HLG/PQ shall use the
specified SDR preview tone map without changing render intent.

### 4.4 Quantization

All corrections shall occur before output quantization. Conversion to 8-bit
shall use deterministic triangular-pdf dither. Ten-bit integer output shall use
deterministic ordered or triangular-pdf dither. Dither seed shall derive from
the source fingerprint and output frame presentation timestamp so repeated
renders are byte-stable when the encoder is deterministic.

## 5. Automatic correction design

### 5.1 Model contract

The initial model shall be an ONNX conversion of the openly licensed DICAM
underwater enhancement model. Both repository code and the exact checkpoint
must pass the release license audit; a code license shall not be assumed to
cover weights without evidence. If the chosen checkpoint lacks redistributable
terms, it shall not ship and v1 remains incomplete until an equivalently tested,
redistributable checkpoint is trained or obtained.

The model operates only as a low-resolution reference-grade generator. Its
pixels shall not be scaled and substituted into the output. This prevents the
model from inventing full-resolution detail and makes correction stable across
preview and export resolutions.

Every packaged model shall have a manifest with:

```json
{
  "schema_version": 1,
  "model_id": "org.uwcc.dicam-natural",
  "model_version": "1.0.0",
  "engine_compatibility": ">=1.0,<2.0",
  "onnx_sha256": "<64 lowercase hex characters>",
  "source_url": "https://github.com/hfarhaditolie/DICAM",
  "code_license": "MIT",
  "weights_license": "<verified SPDX identifier>",
  "training_data_summary": "<provenance document identifier>",
  "input": {
    "layout": "NCHW",
    "shape": [1, 3, 512, 512],
    "color_space": "sRGB-linearized-to-model-contract",
    "range": [0.0, 1.0]
  },
  "output": {
    "layout": "NCHW",
    "meaning": "reference corrected RGB",
    "range": [0.0, 1.0]
  }
}
```

The final input normalization shall match the validated source model rather
than the illustrative wording above; changing it requires a model-version
change and renewed golden tests.

### 5.2 Scene detection and sampling

Analysis shall decode a maximum 960-pixel-long-edge proxy and perform:

1. candidate cut detection from color-histogram distance and edge-change ratio;
2. cut confirmation using adjacent-frame comparison;
3. a mandatory sample at each shot's first and last decodable frame;
4. regular samples at 2 Hz within each shot; and
5. extra samples around rapid luminance or chroma changes.

False cut detection is preferable to smoothing across a real cut. Shots shorter
than 0.5 seconds may use one central sample. Cut thresholds shall be stored in
the analysis configuration and covered by the golden corpus.

### 5.3 Filter classification and signal assessment

The classifier shall combine robust channel percentiles, neutral-candidate
chromaticity, highlight chromaticity, red-channel clipping, and model features.
It shall return `none`, `red`, `magenta`, or `unknown`, plus calibrated
confidence. Confidence below 0.70 shall map to `unknown` and cause the UI to keep
the `Auto` label with a confirmation warning.

For each sample, signal assessment shall estimate channel clipping and noise
floor from flat regions. If usable red dynamic range is under four stops or the
red-channel signal-to-noise estimate is below 6 dB in midtones, red recovery
shall be limited and a low-red-signal warning emitted. Already clipped red from
a capture filter shall similarly constrain correction.

### 5.4 Transform fitting

For each sample:

1. run the packaged model on the normalized proxy;
2. pair source pixels with model-reference pixels, excluding clipped pixels,
   specular highlights, high-gradient edges, and the noisiest 5%;
3. fit a 17×17×17 RGB 3D LUT using robust loss;
4. regularize the LUT toward identity, smooth neighboring cells, and require
   monotonic neutral-axis luminance;
5. constrain maximum exposure change to ±3 EV and per-channel gain to 8×;
6. apply conservative saturation and gamut constraints; and
7. compute confidence from fit residual, excluded-pixel ratio, signal quality,
   model-distance indicators, and constraint activation.

The fitted LUT, not the model image, is the automatic transform. At confidence
below 0.40 the sample shall use the deterministic fallback. Confidence from
0.40 through 0.69 shall blend fallback and ML-fitted transforms in proportion
to confidence. Confidence of at least 0.70 shall use the constrained ML fit.

The deterministic fallback shall use robust neutral-candidate selection,
gray-world/white-patch blending, percentile exposure normalization, a bounded
channel-balance matrix, and a filmic monotonic tone curve. It shall share the
same constraints as ML-fitted transforms.

### 5.5 Temporal stabilization

LUT values and scalar fit parameters shall be stabilized within, never across,
confirmed shots. The baseline stabilizer shall:

- reject isolated values more than three median absolute deviations from a
  five-sample neighborhood;
- apply a one-second, timestamp-aware exponential smoothing constant forward
  and backward for offline analysis;
- reduce smoothing around confirmed gradual exposure changes rather than lag
  them; and
- interpolate stabilized sample LUTs tetrahedrally in color and linearly in
  presentation time.

Using forward and backward passes means complete analysis is non-causal. During
progressive preview, provisional causal values shall be marked as provisional
and replaced atomically as offline stabilization completes.

### 5.6 Cache format

Analysis data shall be stored in a compressed `.uwcc-analysis.npz` file with a
small JSON header, shot table, timestamps, confidence, warnings, LUT knots,
model/config hashes, and source fingerprint. Cache content is derived and may
always be deleted. A project that cannot find or validate its cache shall retain
manual edits and return to `Ready, not analyzed`.

## 6. Manual correction math

- Correction strength shall interpolate the automatic LUT toward an identity
  LUT; 100–200% shall extrapolate the LUT displacement with gamut constraints.
- Temperature and tint shall use a chromatic-adaptation transform with a
  perceptually uniform control mapping, not direct RGB multiplication.
- Exposure shall multiply scene-linear values by `2^EV`.
- Tone controls shall compose into one monotonic spline to prevent stage-order
  clipping. The implementation shall reject non-monotonic spline updates.
- Hue and saturation shall operate in a perceptually uniform opponent-color
  representation and preserve neutral pixels.
- Dehaze shall be edge-aware and bounded. It shall not use a learned
  full-resolution generator and shall not change alpha or geometry.
- Sharpening shall use an edge-limited unsharp mask after the output transform,
  suppress sharpening in noise-dominated regions, and never be included in
  scopes unless `Scopes include sharpening` is selected.

Parameter-to-math curves shall be versioned as `correction_graph_version`.
Changing visible results for an existing parameter value requires a new graph
version and migration behavior.

## 7. Persistence contracts

### 7.1 General rules

All JSON shall be UTF-8, use ISO 8601 UTC timestamps, store paths as JSON
strings, reject duplicate object keys, and preserve unknown extension fields
under an `extensions` object. Readers shall reject a newer unsupported major
schema and preserve older projects before migration.

Writes shall use a temporary sibling file, flush file data, and atomically
replace the destination. Autosave and queue state use the same rule.

### 7.2 Source fingerprint

The quick fingerprint shall contain canonical size, modification time, and
SHA-256 hashes of the first and last 4 MiB. A full-file SHA-256 may be computed
on demand. Cache reuse requires all quick-fingerprint fields to match. Relinking
after a mismatch invalidates analysis even when the user confirms the source.

### 7.3 Project schema

The normative top-level shape is:

```json
{
  "schema_version": 1,
  "application_version": "1.0.0",
  "created_at": "2026-10-22T12:00:00Z",
  "modified_at": "2026-10-22T12:00:00Z",
  "source": {
    "path": "relative/or/absolute/source.mp4",
    "fingerprint": {
      "size": 0,
      "mtime_ns": 0,
      "first_4m_sha256": "<sha256>",
      "last_4m_sha256": "<sha256>"
    },
    "video_stream_index": 0,
    "audio_stream_indexes": [1]
  },
  "input": {
    "profile": "sony_slog3_sgamut3cine",
    "range": "limited",
    "filter_policy": "auto",
    "resolved_filter": "red"
  },
  "color": {
    "ocio_config_id": "uwcc-aces-1",
    "ocio_config_sha256": "<sha256>",
    "correction_graph_version": 1
  },
  "analysis": {
    "state": "complete",
    "cache_path": "source.uwcc-analysis.npz",
    "cache_sha256": "<sha256>",
    "model_id": "org.uwcc.dicam-natural",
    "model_version": "1.0.0"
  },
  "manual": {
    "base_values": {},
    "keyframes": []
  },
  "output_intent": {
    "profile": "rec709_gamma24",
    "render_profile": "dnxhr_hqx_master"
  },
  "extensions": {}
}
```

Keyframes shall contain a stable UUID, parameter name, rational source
timestamp, and value. Timestamps shall use `{ "numerator": n,
"denominator": d }`; floating-point seconds are forbidden in persisted edit
decisions.

### 7.4 Preset schema

`.uwcc-preset.json` shall include schema/application compatibility, name,
description, input policy, auto settings, manual base values, optional manual
keyframe values normalized to a 0–1 clip duration, output intent, and extensions.
The default UI saves no keyframes in a preset; including normalized keyframes
requires an explicit `Include animation` choice.

### 7.5 Queue schema

Queue state shall live in the user data directory, not in a project. It shall
contain stable job IDs, source fingerprints, preset snapshot, destination,
render profile snapshot, state, progress, validation result, timestamps, and a
redacted diagnostic reference. On restart, `Analyzing` and `Rendering` jobs
become `Paused` and require an explicit resume.

## 8. Export implementation

### 8.1 Frame and audio timing

Rendering shall iterate source presentation timestamps, not assume a constant
frame interval. Correction parameters shall evaluate at each source PTS. Output
defaults to the source time base and cadence. Audio shall preserve its initial
offset; mux validation shall compare first/last timestamps and total duration.

### 8.2 Normative codec settings

Defaults shall be:

- DNxHR HQX: 10-bit 4:2:2, source dimensions/cadence, PCM signed 24-bit.
- ProRes 422 HQ: `prores_ks` profile 3, 10-bit 4:2:2, PCM signed 24-bit.
- FFV1: version 3, level 3, slice CRC enabled, 10/16-bit RGB or 4:4:4 selected
  to avoid an additional chroma loss, FLAC audio.
- H.264 SDR: libx264 high profile, CRF 17 default, 8-bit 4:2:0, `faststart`, AAC
  320 kbit/s when audio cannot be copied.
- HEVC SDR: libx265 Main10, CRF 18 default, 10-bit 4:2:0, AAC 320 kbit/s.
- HEVC HDR: libx265 Main10, CRF 18 default, 10-bit 4:2:0, Rec.2020 non-constant
  matrix with the selected HLG or PQ transfer and matching container tags.

The UI may expose quality presets but shall always show resulting codec, bit
depth, chroma sampling, color tags, and estimated size. Hardware encoders are
non-normative and must pass the same metadata and timing tests.

### 8.3 Output validation

Before atomic rename, UWCC shall re-probe the `.partial` file and require:

- a decodable video stream with expected codec, dimensions, and color tags;
- expected frame count for constant-rate sources, or matching start/end PTS for
  variable-rate sources;
- duration within one source-frame interval;
- all selected audio streams present with start offset within 20 ms;
- no decode error in first, middle, and last validation frames; and
- a nonzero file size consistent with the encoded streams.

Failed validation marks the job failed and retains the diagnostic probe.

## 9. Concurrency, performance, and resource limits

- The UI thread shall never wait synchronously for probe, decode, inference,
  cache I/O, scopes, or render completion.
- Decode and render workers shall use bounded queues; unbounded frame buffering
  is forbidden.
- The default memory budget shall be the lesser of 50% of physical RAM or
  8 GiB, adjustable down to 2 GiB.
- Proxy cache eviction shall be least-recently-used and honor a default 20 GiB
  disk budget.
- Cancel requests shall be observed within 500 ms during analysis and within
  two seconds or one encoder GOP during rendering, whichever is longer.
- GPU provider order shall be configured explicitly and logged. CPU shall be
  the final provider and the normative result within numeric tolerance.

Reference performance hardware is Ubuntu 26.04, AMD Ryzen 7 5700X, 32 GiB RAM,
and NVIDIA RTX 3060 12 GiB using the installed system driver. After analysis,
the application shall sustain 24 fps at a 1920×1080 preview proxy for supported
4K sources, with p95 manual-control response below 100 ms. CPU-only operation on
the same CPU shall sustain 8 fps at 1280×720 proxy and complete every workflow,
even when it cannot play corrected 4K in real time.

## 10. Platform and packaging

### 10.1 Ubuntu v1

The release artifact shall be an x86-64 AppImage tested on a clean Ubuntu 26.04
installation. It shall bundle the Python runtime, application dependencies,
OpenColorIO configuration, verified model, and a reproducible GPL-compatible
FFmpeg build. It shall not replace system libraries globally or require root.

User directories shall follow the XDG base-directory specification. The app
shall tolerate spaces, non-ASCII characters, removable volumes, and paths longer
than 260 characters.

### 10.2 Windows portability

Core modules shall use `pathlib`, avoid POSIX-only process assumptions, and keep
platform integrations behind interfaces. CI shall run schema, correction-graph,
and path tests on Windows 11. A packaged Windows application, DirectML/WinML
provider, code signing, and installer are deferred and are not v1 gates.

### 10.3 Open-source and provenance gate

The release pipeline shall:

1. generate CycloneDX SBOMs for Python, native, model, and bundled tool assets;
2. verify each asset against a reviewed SPDX allowlist;
3. fail on an unknown license, missing model-weight license, or network-fetched
   unpinned artifact;
4. include source offers/notices required by bundled licenses;
5. include the full canonical GPLv3 text; and
6. publish checksums and reproducible build inputs.

Installed proprietary GPU drivers may accelerate a run through a supported
provider. UWCC shall neither bundle nor require CUDA, proprietary codec SDKs,
or a proprietary runtime. CPU software paths remain the supported baseline.

## 11. Security and privacy

- Media and project parsers shall treat all input as untrusted.
- FFmpeg and native dependencies shall be kept on supported security branches.
- Worker processes shall receive only required file descriptors and paths.
- No shell interpolation shall be used for media commands.
- JSON size, nesting, array length, and numeric ranges shall be bounded before
  allocation.
- Model and analysis files shall be hash-verified before loading.
- Project-relative paths shall not be allowed to escape an explicitly selected
  project directory when resolving adjacent caches.
- Diagnostic bundles require preview and confirmation of included paths and
  metadata; decoded frames are excluded.
- The shipped application shall contain no analytics, crash-upload, update,
  advertisement, authentication, or remote-model endpoint.

## 12. Verification plan

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

Pull requests shall run linting, type checks, unit tests, schema fixtures,
CPU golden frames, and short media round trips. Scheduled CI shall run the full
corpus, GPU/CPU comparison, performance benchmarks, fuzzed parsers, and Windows
portability tests. A release candidate additionally requires clean Ubuntu 26.04
installation testing, offline runtime observation, SBOM/license approval,
package malware scanning, and manual visual review sign-off.

Numeric golden comparisons shall state their color space and use per-test
tolerances. Updating a golden result requires a reviewed explanation and before/
after artifacts; accepting all new output wholesale is forbidden.

