# C0045 fused native-kernel experiments

Measured on 2026-11-08. The initial experiments below ran separately from the
application. Prototype sources, extensions, drivers, and results are under
Git-ignored `benchmarks/results/C0045-export/`. Subsequent production integration
is recorded in its own section below.

## Scope

Same machine and clip as the [pipeline investigation](C0045-export.md): Ryzen 7
5800X, Windows, Python 3.14.10, NumPy 2.5.4, FFmpeg 9.0.1, 1080p/50 fps,
1,032 frames over 20.64 seconds. MSVC compiled a separate
`_fused_lut_experiment` module with `/O2 /fp:strict`, without fast-math or explicit
SIMD. The real `uwcc._native` module was not rebuilt or replaced.

Three paths were compared:

1. Current C LUT interpolation followed by NumPy blend, clamp/sanitization,
   scaling, rounding, and uint16 conversion.
2. Fusion only: retain generic interpolation ordering but combine interpolation,
   strength blending, clamp/sanitization, and RGB48 quantization in one C loop.
3. Fusion plus specialization: additionally specialize the uniform Rec.709 cube,
   select tetrahedron corners directly, and use simple bounds checks in the
   uniform coordinate calculation.

The prototype emits a uint16 output buffer directly, eliminating the intermediate
float32 corrected image and separate full-frame finishing/quantization passes.
It releases the GIL during its pixel loop. This experiment does not isolate
each individual contribution within the combined specialization.

The fast-path scope is Rec.709 input/output, encoded automatic LUT analysis, and
neutral manual finishing controls. Strength remains adjustable/keyframable.
Zero strength uses the existing path because that path currently runs the color
graph rather than the encoded-LUT branch. Manual finishing, log/HDR, missing
analysis, and non-LUT analysis also retain the original path. No denoise was
enabled in these benchmarks. Neural-reference grades were not benchmarked.

## Numerical checks

- 36 synthetic kernel comparisons: cube sizes 2/17/33, strengths 0–200%, neutral
  ties, gamut boundaries, out-of-range values, NaN/infinity, and both prototypes.
- 36 camera-frame comparisons: three full-resolution timestamps, strengths
  0/25/65/100/150/200%, and both guarded prototype variants.
- 32 further checks: rounding boundaries, strength keyframes, and preservation
  of manual/HDR/missing-analysis fallback behavior.

All comparisons had **zero differing uint16 values**. The zero-strength
camera/keyframe checks exercise the existing-path fallback, rather than proving
that an identity shortcut matches the full color graph.

## Timings

Five single-frame repetitions per variant, alternating execution order, measured
correction through RGB48 quantization on the same real 1080p frame:

| Path | Median per-frame time |
| --- | ---: |
| Current C + NumPy stages | 137.94 ms |
| Fusion only | 106.56 ms |
| Fusion plus specialization | **49.25 ms** |

The 200-frame pipeline test included actual decoding, float normalization,
stabilization warping, correction, and quantization. It reused the stabilization
plan and discarded output instead of encoding:

| Path | Elapsed | Throughput |
| --- | ---: | ---: |
| Current | 12.54 s | 16.0 fps |
| Fusion only | 8.48 s | 23.6 fps |
| Fusion plus specialization | **6.36 s** | **31.4 fps** |

Finally, a matched full stabilized export used four color workers, unchanged
codec/filter pools, H.264 CRF 17/preset fast, the same saved grade, and no denoise.
Orientation was resolved once before both timed runs. Both runs included motion
analysis, encoding, audio muxing, and output validation:

| Full export | Elapsed |
| --- | ---: |
| Current application path | 90.84 s |
| Experimental fusion plus specialization | **56.94 s** |

That is **37.3% less time**, or 1.60× end-to-end throughput. The frame-processing
interval, excluding the prepass and final mux/validation, fell from 71.72 to
38.10 seconds. These are individual full-export trials, not repeated-trial
statistics. A fusion-only full export was not measured.

The two complete MP4 files were **byte-identical**, SHA-256:

```text
373408c54d84f05b343d135f0c406095b4ec8f0764449bd16066a058b6ead226
```

Both retained all 1,032 frames and audio and passed duration/color-tag validation.

## SIMD follow-up

Two further isolated modules tested SSE2 across the RGB channels of one pixel
and AVX2 across eight pixels, using gathers for the interleaved input and LUT
corners. Multiplication/addition remained separate (no FMA/fast-math) to preserve
rounding. AVX2 hardware/OS support was checked through NumPy's CPU-feature
information before loading/running the AVX2 module. The experiment was run on
this AVX2-capable Ryzen only; it is not a production CPU-dispatch implementation.

Expanded checks covered 1,026 synthetic comparisons, including every vector-tail
length, cube sizes 2/17/33, strengths 0–200%, ties, edges, and NaN/infinity.
Another 45 real-frame comparisons covered three camera timestamps and positive
strengths through 200%. All uint16 outputs matched the current reference exactly.
Existing Python guards, including zero-strength fallback, were retained.

Seven alternating single-frame repetitions per variant gave:

| Fused kernel | Median time |
| --- | ---: |
| Scalar specialization | 50.55 ms |
| SSE2 | 41.03 ms |
| AVX2 | **19.88 ms** |

AVX2 reduced kernel time by about 61% (2.54× throughput), but the 200-frame
decode/warp/color pipeline was almost unchanged:

| Fused pipeline | Elapsed | Throughput |
| --- | ---: | ---: |
| Scalar | 6.34 s | 31.5 fps |
| SSE2 | 6.23 s | 32.1 fps |
| AVX2 | 6.21 s | 32.2 fps |

The pipeline is now close to the previously measured decode/warp throughput;
faster interpolation has diminishing end-to-end benefit. A fresh matched full
stabilized export pair measured **60.47 seconds scalar versus 57.31 seconds
AVX2**, approximately 5% less time in this pair. The earlier scalar run was
56.94 seconds, so these single-run results do not establish a repeatable 5%
improvement independently of run-to-run variation. A full SSE2 export was not run.

The complete scalar and AVX2 MP4 files were byte-identical, with the same SHA-256
shown above, and passed normal frame/audio/timing/color validation. Application
source hashes were again verified unchanged.

SIMD works correctly in this experiment and substantially accelerates the kernel.
Fusion remains the larger demonstrated export gain; frame transfer/conversion
and pipeline scheduling merit attention next. Production SIMD integration would
need CPU/OS dispatch, a scalar fallback, separate compilation of AVX2-targeted
code, and non-x86 handling. These experimental modules are not shipped or loaded
by UWCC.

Additional local artifacts: `fusion/simd.c`, `fusion/avx2.c`, their build drivers,
`fusion/simd-report.json`, `fusion/simd-pipeline-report.json`,
`simd_experiment.py`, `simd_pipeline.py`, `simd-scalar.mp4`, and `simd-avx2.mp4`.

## Production integration

The fused path is now integrated into `color.correct_rgb48` and the real export
worker. AVX2 is selected by default after CPU AVX2/XSAVE and OS XMM/YMM state
checks. Unsupported CPUs use fused scalar interpolation; without the extension,
NumPy retains the float graph. Active manual controls, log/HDR, zero strength,
and absent/non-LUT analysis retain their existing graph.

AVX2 is a separate object with no cross-object optimization/inlining. Baseline
code has no AVX2 compiler flag, and non-x86 targets build scalar code. Source
archives/Windows snapshots include the new C source and header. Build tests
cover MSVC/GCC flag isolation, ARM64 selection, and AVX2 compiler failure.
Ubuntu/Windows CI now builds native kernels before testing them.

The production module selects `avx2` on this machine. A full stabilized export
through the actual application path took **59.90 seconds**, retaining all 1,032
frames, 20.64 seconds, and audio. Its MP4 was byte-identical to the earlier
90.84-second pre-fusion reference (SHA-256 above). This is a production
confirmation run, not a fresh controlled pair.

Full Windows suite: **294 passed, 15 skipped**. Tests cover default/forced-scalar
parity, vector tails, rounding boundaries, invalid/overlapping buffers,
read-only inputs, fallback guards, animation, and NumPy fallback. A separately
compiled module without the AVX2 object loaded, reported `scalar`, and matched
production output. Source archive contents were checked. Linux native execution
remains for CI; no local WSL environment is installed.

Local production artifacts: `production-avx2.mp4` and
`production-avx2-report.json` under `benchmarks/results/C0045-export/`.

## Assessment and artifacts

Fusion is worthwhile on its own; uniform-cube specialization adds a substantial
additional benefit. The guarded Rec.709 path is now integrated as described
above. Results do not establish equivalence on every platform, compiler, grade,
or model. The SIMD follow-up above shows
kernel-level benefit but limited demonstrated additional export throughput.

During the initial experiments the driver substituted only its own process's
`corrected_frames` function, and application source hashes were verified
unchanged. Production source changes followed during integration.

Local artifacts:

- `fusion/fused.c`, `fusion/setup_experiment.py`: isolated C module/build driver.
- `fusion/parity-report.json`, `fusion/guard-and-rounding-report.json`: checks.
- `fusion/micro-report.json`, `fusion/pipeline-report.json`: timing records.
- `fusion-current.mp4`, `fusion-fast.mp4`: matched full exports.
- `fusion_experiment.py`, `fusion_pipeline.py`: experiment drivers.

All paths above are relative to `benchmarks/results/C0045-export/`. No prototype
module is added to the application package or MSIX build.
