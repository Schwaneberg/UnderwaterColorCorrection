# C0045 export investigation

Measured on 2026-11-08 with the user's `C0045.MP4`. Source media and generated
outputs remain local under `benchmarks/results/C0045-export/` (Git-ignored).

## Setup and scope

- AMD Ryzen 7 5800X, Windows, Python 3.14.10, NumPy 2.5.4, FFmpeg 9.0.1.
- Source: 1920×1080, 50 fps, 20.64 seconds, 1,032 frames, one PCM audio stream.
- Rec.709 interpretation from the existing stream-tag detection. This is a
  performance test, not verification of the camera's recording picture profile.
- Default adaptive levels/white balance, neutral manual controls, SDR H.264
  CRF 17 / preset fast, four color workers, CPU decoding and encoding.
- Stabilization and denoise disabled. These results do not characterize neural
  denoise, custom grades, or hardware video codecs.
- Initial fallback measurements ran without `uwcc._native`. After installing
  Visual Studio Build Tools 2026 (18.11.3), MSVC 14.51.36231, and Windows SDK
  10.0.28000.0, the extension built successfully for the same Python environment.

Analysis was performed once and reused within each before/after comparison.
Export timings include orientation checks, timestamp scanning, decode, color,
encoding, audio muxing, and validation; initial analysis is excluded. The
one-second camera sample was stream-copied into MOV without re-encoding video
or audio; it contains 51 frames over 1.02 seconds.

## Results

| Measurement | Released NumPy path | Optimized NumPy path |
| --- | ---: | ---: |
| Six full-resolution correction calls, average per frame | 762 ms | 290 ms |
| Complete 51-frame sample export | 27.27 s | 16.38 s |
| Complete original 1,032-frame clip export | Not completed | 137.08 s |

The matched sample took **40% less time**, or **1.66× throughput**. An earlier
optimized sample trial took 15.88 seconds. These are individual local runs, not
controlled repeated-trial statistics. Do not extrapolate the sample speedup
into a measured full-clip before/after result: the original full-length baseline
was stopped early after profiling identified the bottleneck.

The full optimized export achieved 7.53 fps (0.15× source playback rate). It
retained all 1,032 frames and one audio stream and passed export duration/color
tag checks. This remains substantially slower than real time on the NumPy path.

## Compiled-kernel follow-up

The same full-clip project and encoder settings were measured with the newly
built `uwcc/_native.cp314-win_amd64.pyd`:

| Measurement | Optimized NumPy | Compiled kernels |
| --- | ---: | ---: |
| Full 1,032-frame export | 137.08 s | 84.66 s |
| Export throughput | 7.53 fps | 12.19 fps |
| Six correction calls, average per frame | 332 ms | 130 ms |

The compiled export took **38% less time** than the optimized NumPy export
(1.62× throughput). The 332 ms profile comes from the full-clip run; the earlier
290 ms sample profile used its separately analyzed sample grade. Both full-clip
runs used the same saved analysis and four color workers. All frames/audio were
retained and export validation passed. These remain individual local trials,
not a statistical throughput guarantee.

This follow-up measures enabling existing C acceleration, rather than a C LUT
algorithm change. Builds that already include compiled kernels should not
expect the NumPy LUT speedup. Restart an already running UWCC process after
building the extension so it can load the compiled module.

## Stabilization measurement

A fresh paired full-clip run used the same saved grade and native extension,
four color workers, CPU H.264 encoding/decoding, and no denoise. Stabilization
used the default seabed/lower-half tracking region. Orientation was resolved
once before both timings, unlike the earlier 84.66-second native measurement
which included orientation detection. OpenCV 5.0.0 reported 16 threads.

| Scenario/stage | Wall time |
| --- | ---: |
| Color-only export | 71.51 s |
| Stabilized export, including analysis | 99.98 s |
| Stabilization plan (included above) | 28.16 s |
| All 1,032 full-resolution warp calls, summed | 4.20 s |

Stabilization added **28.47 seconds**, or **39.8%** to the paired baseline.
The extra proxy analysis pass accounts for almost all of that increase. The
warp calls average about 4.1 ms/frame and overlap with the parallel color and
encoding pipeline; their summed time is not an additional 4.20 seconds to add
on top of the measured export delta. Subtracting the plan leaves 71.82 seconds
for the stabilized render/mux/validation path, close to the color-only run.

The analysis profile recorded 8,248 `goodFeaturesToTrack` calls taking 16.29
seconds, about 58% of the plan's wall time. Motion tracking estimates took
22.39 seconds including those calls; two-way optical flow accounted for 2.09
seconds. The implementation detects features separately in eight spatial
regions for each frame. Avoiding repeated feature-response computation while
retaining spatial coverage is a candidate optimization, not an implemented or
validated change. The profile also contains Python thread-attribution artifacts;
the end-to-end and plan totals above use independent wall-clock timers.

The plan selected a fixed 1.296× zoom to avoid exposed edges. Both outputs
retained 1,032 frames, 20.64 seconds, and one audio stream and passed export
validation. This measurement does not establish perceptual stabilization quality
or that the crop is acceptable for every subject. It is one instrumented pair,
not repeated-trial statistics, and does not test whole-frame tracking or GPUs.
No stabilization algorithm or thread setting was changed for this measurement.

Local outputs and diagnostics:

- `benchmarks/results/C0045-export/native-repeat.mp4`
- `benchmarks/results/C0045-export/native-stabilized.mp4`
- `benchmarks/results/C0045-export/stabilization-report.json`
- `benchmarks/results/C0045-export/stabilization-plan-profile.txt`

## Motion-analysis optimization experiments

Isolated benchmark prototypes compared the existing detector with regional
feature detection on the complete 1,032-frame clip. All variants kept the
640-pixel proxy, eight spatial regions, feature limits/thresholds, two-way optical
flow, and RANSAC settings. They did not alter application code. OpenCV's RNG seed
was reset before each run for comparable RANSAC fits. Runs were sequential and
unprofiled; initial orientation detection was excluded.

| Analysis variant | OpenCV threads | Full plan time | Change from existing |
| --- | ---: | ---: | ---: |
| Existing full-image detector with region masks | 16 | 28.39 s | Baseline |
| Existing detector | 1 | 32.13 s | 13% slower |
| Cropped regional detector plus halo | 16 | 13.22 s | **53% less time** |
| Cropped regional detector plus halo | 1 | 18.53 s | 35% less time |

The strongest candidate is to give each detector call a cropped input region,
rather than the whole proxy with a mask. [OpenCV documents](https://docs.opencv.org/4.12.0/dd/d1a/group__imgproc__feature.html)
that corner quality is computed at every source-image pixel before selection
and nonmaximum suppression. A mask selects eligible points but does not avoid
that full-image response computation. The current eight masked calls repeat
this work eight times per frame.

The prototype retained a five-pixel halo around each crop for the current
7-pixel response block, 3-pixel derivative aperture, and nonmaximum neighborhood,
then translated detected points back into proxy coordinates. This halo must be
revisited if detector parameters change. Cropping still retains a separate
quality threshold and corner quota for each spatial region.

All 88 sampled feature-list comparisons matched exactly, including ordering.
Every full-clip variant produced exactly the same 1,032 motion matrices as the
baseline, zero corner-displacement difference, and the same 1.296295× crop zoom.
This is strong evidence for this clip, not general equivalence across camera
footage or OpenCV versions. Integration adds regional-boundary and border tests
and retains the existing shake/crop tests. End-to-end results following
integration appear below.

Additional opportunities identified by inspection, not separately benchmarked:

- Cache a completed stabilization plan for repeated exports, invalidating it
  when the source fingerprint, orientation, tracking settings, or algorithm
  version changes. Manual color changes should not require repeating geometric
  motion analysis.
- Precompute region bounds/masks and constant scale transforms. Skip the
  identity resize when analysis frames already have the requested proxy size;
  resize accounted for 0.86 seconds in the earlier profile. These are secondary
  to reducing repeated corner-response computation.

Lowering OpenCV thread count alone is not supported by these measurements: it
slowed both detector variants on this machine. No production thread setting
was changed. The prototype restored the original setting before exit.
Local experiment results and matrices are in
`benchmarks/results/C0045-export/motion-alternatives-report.json` and the
corresponding `*-matrices.npy` files.

## Integrated regional detection

Regional detection is now implemented in `uwcc/finishing.py`. Each cell receives
a local mask and a cropped image with the five-pixel halo; detected points are
translated back into proxy coordinates. Feature quotas/thresholds, optical flow,
RANSAC, smoothing, crop calculation, and OpenCV thread settings are unchanged.

| Full-clip measurement | Original | Integrated regional detector |
| --- | ---: | ---: |
| Motion/crop analysis | 28.16 s | 13.25 s |
| Complete stabilized export | 99.98 s | 90.31 s |

Analysis took **53% less time**. End-to-end export took **9.7% less time** in
these individual runs. Other pipeline work varied between runs, so the analysis
time saving should not be treated as a guaranteed subtraction from every export.

All 1,032 production motion matrices matched the original baseline exactly,
including the 1.296295209× zoom. The original and integrated MP4 files were
**byte-identical**, with SHA-256:

```text
373408c54d84f05b343d135f0c406095b4ec8f0764449bd16066a058b6ead226
```

Decoded-frame checksum comparison independently confirmed 1,032 matching video
frames/timestamps. Audio, duration, and export validation also passed. The new
tests exercise random, checkerboard, and flat images; small, portrait, and wide
shapes; noncontiguous inputs; cell/image boundaries; empty cells; scene cuts;
and exact motion/crop/warp parity. Full Windows suite with compiled kernels:
**259 passed, 15 skipped**. Linux execution remains for CI.

Local integrated output and measurements are
`benchmarks/results/C0045-export/regional-stabilized.mp4` and
`benchmarks/results/C0045-export/regional-stabilization-report.json`.

## Export-pipeline bottlenecks

The later [raw-copy and pipe optimization](C0045-raw-copies.md) addresses part
of the transport overhead identified below. It is now integrated; the tables
in this section record the earlier bottleneck investigation.

Stage isolation used the first 200 full-resolution frames with native kernels,
the saved adaptive grade, default codec pools (8 threads / 4 filter threads),
four color workers, and no denoise. The existing full-clip stabilization plan
was reused for the warp tests; motion analysis is excluded from those timings.
Each stage consumed/discarded its output rather than retaining frames in RAM.

| Isolated path | 200-frame time | Throughput |
| --- | ---: | ---: |
| FFmpeg decode + RGB48 conversion to null output | 0.93 s | 214.5 fps |
| FFmpeg decode + RGB48/YUV conversion + H.264, without Python | 4.19 s | 47.7 fps |
| Python frame reader, including float normalization | 5.04 s | 39.7 fps |
| Python frame reader + stabilization warp | 5.83 s | 34.3 fps |
| Python frame reader + native color + quantization, discarded output | 11.71 s | 17.1 fps |
| Python frame reader + warp + native color + quantization, discarded output | 12.51 s | 16.0 fps |

The direct FFmpeg transcode does not apply UWCC's grade and writes to a null
output, so it is not an equivalent rendered deliverable or a measurement of
encoding alone. It nevertheless demonstrates much more throughput headroom
than the current graded pipeline. Codec decoding/conversion itself is fast;
transporting RGB48 through a subprocess pipe and normalizing it into NumPy
adds substantial overhead before grading begins.

One 1080p RGB48 frame is 12.44 MB; the full clip transfers approximately 12.84 GB
from decoder to Python and another 12.84 GB from Python to encoder. These volumes
explain why frame copies, conversion, and IPC merit attention even for a small
compressed source file. They do not by themselves prove a memory-bandwidth limit.

An instrumented full stabilized export took 88.17 seconds and passed export
validation. Independent timers recorded:

| Measurement | Time |
| --- | ---: |
| Motion/crop plan | 12.93 s |
| First rendered frame to audio-mux stage | 69.36 s |
| Aggregate color-worker CPU time | 248.70 CPU-s |
| Aggregate color-worker elapsed time | 272.52 worker-s |
| Waiting for ordered color results on the main thread | 18.85 s |
| Reads from decoder pipes, including the proxy analysis pass | 18.99 s |
| Writes to encoder stdin | 2.99 s |
| Full-resolution warp calls | 4.51 s |
| Audio mux + final validation | 2.38 s |

These timers overlap and must not be added together. Worker CPU time represents
approximately 3.6 CPU cores during frame processing. The small encoder-pipe write
time means encoder backpressure is not the dominant stall in this run. Color
processing is the strongest measured bottleneck, followed by input transport
and float conversion. Warp and mux/validation are smaller costs. The 13-second
motion prepass remains separate overhead, especially for repeated exports.

A separate 200-frame, four-worker CPU breakdown attributed 26.41 CPU-seconds to
the C LUT kernel, 12.41 CPU-seconds to other correction work, and 3.44 CPU-seconds
to quantization/worker overhead: approximately **62.5%, 29.4%, and 8.1%** of
worker CPU time. Native interpolation is still the largest cost; surrounding
blend/clamp/sanitization and conversion passes are also meaningful optimization
targets. The split is approximate and was measured without encoding.

A worker-count sweep on decode + warp + color (no encoder) yielded:

| Color workers | 200-frame time | Throughput |
| --- | ---: | ---: |
| 2 | 19.24 s | 10.4 fps |
| 4 | 12.59 s | 15.9 fps |
| 8 | 11.40 s | 17.5 fps |

Eight workers improved throughput by about 10.4% over four, rather than doubling
it. This isolated test called the worker pool directly; production's default
768 MiB working-memory estimate caps 1080p color processing at four workers.
Requesting eight through normal export would also require a larger memory
budget (approximately 1,536 MiB). No default or memory limit was changed, and a
full eight-worker export has not been benchmarked.

The next targets are a faster/fused native interpolation/blend/quantization
path, fewer frame conversions/copies, and testing color-worker/codec-pool
combinations under a coordinated CPU budget. Increasing encoder/filter pools
alone is not the leading opportunity supported by these measurements. Caching
valid stabilization plans could remove the prepass on repeated exports.
The fused path has since been integrated as described in the
[native-fusion report](C0045-fusion-experiment.md#production-integration).
Other opportunities above remain investigation findings.

Local data: `pipeline-stages-report.json`, `color-worker-sweep.json`, and
`color-breakdown.json` under `benchmarks/results/C0045-export/`.

## Changes and numerical checks

A later [isolated native-fusion experiment](C0045-fusion-experiment.md) measured
37.3% lower full-export time with byte-identical output. Subsequent production
integration selects AVX2 by default where supported, retains a scalar fallback,
and exported the full stabilized clip in 59.90 seconds with identical bytes.

The old NumPy LUT calculation accounted for about 86% of the six-frame color
profile. It created whole-frame float64 intermediates and per-pixel sort/index
arrays. The replacement uses at most 65,536 pixels of scratch work per block,
float32 fractions/results, and direct selection of four tetrahedron corners.
The C LUT implementation is unchanged. Decoder normalization and export rounding
also reuse a temporary array instead of allocating another full-frame result.
Encoder quality, bitrate policy, dimensions, and frame cadence are unchanged.

Six real frames at 0, 0.2, 0.4, 0.6, 0.8, and 1 seconds were compared with the
released color graph before encoding. Maximum absolute RGB error was
`1.1920929e-7`, mean error was at most `1.21e-8`, and 16-bit transport values
differed by at most one code value. The float32 result is not byte-identical to
the former float64 intermediate calculation. The independently encoded H.264
sample outputs measured 42.84 dB average PSNR against each other; lossy output
differences are not a measurement of physical color accuracy.

Reference interpolation tests cover both uniform Rec.709 and log-shaped cubes,
multiple LUT sizes, all tetrahedron axis orderings, neutral ties, gamut edges,
noncontiguous input, and block boundaries. The initial fallback-only Windows
suite reported **237 passed, 23 skipped**. After compiling, all **28 selected
native/color/render tests passed**, and the full suite reported **245 passed,
15 skipped**. Linux timings remain to be measured.

## Reproduce

With the source available locally, a complete benchmark can run even without
the optional C extension:

```powershell
.venv\Scripts\python.exe -m benchmarks.export --clip "D:\Video\Zakynthos_Video\03-2 Ostriff\C0045.MP4" --input-profile rec709 --backend numpy --output-dir benchmarks/results/C0045-repeat --report benchmarks/results/C0045-repeat/report.json
```

Use a fresh output directory for each run. Use `--project` with a saved project
to measure an actual grade instead of the default adaptive analysis. After
installing the MSVC C++ tools and Windows SDK, build with the same Python used
to run UWCC, then repeat with `--backend both`:

```powershell
.venv\Scripts\python.exe setup.py build_ext --inplace
.venv\Scripts\python.exe -c "from uwcc import _native; print(_native.__file__)"
```

The explicit import checks whether the optional build succeeded; an optional
extension build can otherwise finish with only a warning. The fallback
optimization does not establish a speedup for Store builds that already ship
compiled kernels; the separate native follow-up above measures their existing
acceleration on this machine.
