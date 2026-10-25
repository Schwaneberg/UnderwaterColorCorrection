# C0201 AI upgrade and parallel export results

Measured on 2026-10-25, AMD Ryzen 7 5800X (8 cores / 16 threads), Python 3.14.4.
Source: 1920×1080, 100 fps, 24.48 s, 2,448 frames. Camera color interpretation
remains provisional Rec.709. The user's corrected reference has matching geometry
and duration, with no stabilization/denoise.

## Learned look and local AI

The corrected-reference workflow fits time-dependent RGB regressions at 49
timestamps and interpolates their LUTs. Mean training RGB error: 0.0053.
Held-out times 1.25, 7.25, 13.25 and 21.25 s yield errors 0.00605, 0.00527,
0.00401 and 0.00556 (mean 0.00522 on a 0–1 scale). This quantifies resemblance
to the preferred grade, not physical color accuracy.

UDnet V1 runs at proxy resolution during analysis. Its neural color reference
is finished with levels/contrast/vibrance, fitted to smooth color transforms,
and applied to full-resolution frames with the C encoded-domain LUT kernel.
FastDVDnet is installed separately for optional full-resolution temporal
neural denoising. Its actual weights pass a synthetic noise-reduction test;
PyTorch/ONNX conversion maximum error is below 0.000003 for both models.
The full review exports below use conventional denoise at strength 3, not the
much slower full-resolution neural denoiser.

## Parallel processing

The following measurements use the same 201-frame (~2.01 s) source section and
same learned reference grade. Each is one sequential wall-clock observation.
Codec/filter thread limits and color worker counts are varied; no simultaneous
full exports were run for these comparisons.

| Scenario | Codec threads | Color workers | Time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| one-core | 1 | 1 | 23.44 s | 8.58 fps |
| threaded-ffmpeg | 8 | 1 | 18.63 s | 10.79 fps |
| parallel-color | 8 | 3 | 10.28 s | 19.55 fps |

Native FFmpeg threading alone improves the sample from 23.44 s to 18.63 s;
three color workers reduce this to 10.28 s. The threaded-FFmpeg/one-worker and
three-worker exports are byte-for-byte identical:

`4ca282d86cd913702a65316d8df1e2ac553949bab97adf9038f68f508d664f1d`

Color work and output conversion overlap with decode/encode. Stabilization and
temporal denoise are kept sequential, and corrected frames are written in order.
The result queue is bounded and worker count is capped by an estimated memory
budget. Higher-resolution footage automatically uses fewer concurrent workers.

## Fixed crop with no exposed edges

The stabilization analysis pass records all affine corrections, then chooses
the smallest fixed zoom that keeps every inverse-mapped output corner inside
the source with a sampling margin. All frames use that same zoom, including
frames with smaller correction: there is no per-frame zoom pumping. The image
is never filled with mirrored pixels; a constant-fill safeguard is outside the
computed crop bounds. Tests warp uniform valid images under translations and
rotations in landscape, portrait, and wide aspect ratios to check every output
pixel for exposed fill.

The complete UDnet + automatic levels + background stabilization + conventional
denoise export uses **1.470× fixed zoom**. It completed in **173.41 s
(2 min 53 s)**, or **14.12 fps**. All 2,448 frames, 24.48-second duration,
one audio stream, dimensions and requested color signaling passed validation.
The crop necessarily narrows the field of view and scales the retained image
back to 1920×1080.

The prior serial UDnet/stabilization export took 325.97 s with mirrored borders;
the final version includes the extra crop analysis pass and took 173.41 s.
This is an observed workflow comparison, not an isolated crop-speed experiment.

Local output: `benchmarks/results/ai-upgrade/local-ai-cropped-parallel-full.mp4`.
The separately learned user-reference look is available as
`benchmarks/results/ai-upgrade/reference-learned-full.mp4`.

## Checks

85 tests pass, including real FastDVDnet inference/noise reduction, crop bounds,
parallel ordering/backpressure/cancellation, VFR timing/audio preservation,
reference-fit generalization, persistence and mocked cloud-provider contracts.
An additional AI-environment GUI/model-focused run passed 15 tests before the
crop change. Ruff passes. Live OpenAI/Claude calls were not made; those features
require user API credentials and an explicit proxy-frame upload action.

See [AI workflow](../ai-workflow.md) and [user guide](../user-guide.md) for setup,
models, controls and command examples.
