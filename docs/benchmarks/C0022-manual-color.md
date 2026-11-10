# C0022: manual-color export performance

Measured on Windows on 2026-11-10 using the workspace `.venv`, Python 3.14,
the rebuilt native extension, Ryzen 7 5800X and RTX 3060 Ti. This optimization
executes on the CPU; installing CUDA does not move the manual-color graph onto
the GPU.

## Diagnosis

The active Store 0.5.1 export used `C0022.MP4`, 3840×2160 at 25 fps,
stabilization, Rec.709 input/output, black point 36, and CPU H.264 encoding.
Denoise was off. Its recovery project selected the UDnet-derived automatic LUT.
The first diagnostic measured 2.35 seconds for one frame's color stage with
black point 36, versus 0.077 seconds with neutral manual controls. That
comparison changes the look; it diagnosed the dispatch cost, not output parity.

Any non-neutral manual control disabled the existing fused AVX2 LUT-to-RGB48
kernel. The remaining NumPy path allocated large intermediates and ran many
separate per-pixel operations. The outer 768 MiB estimate selected one color
worker at 4K. CUDA libraries were loaded, but enhancement inference is sampled
during analysis rather than executed independently on each output frame.

## Implementation

- Retain the existing AVX2 path for neutral Rec.709 LUT exports.
- Fuse Rec.709 decoding, pointwise manual controls, gamut compression, encoding
  and RGB48 quantization into a portable native kernel.
- Retain both matrix round trips and the reference float32/float64 stages.
  Apply automatic LUT/strength in float32 before manual controls. Do not
  quantize that intermediate or approximate manual controls with a 3D LUT.
- Dispatch blocks of at most 65,536 pixels to a single shared pool of at most
  four workers, respecting CPU affinity and `UWCC_THREADS`. Concurrent frames
  share the same pool. The existing outer frame-worker budget is unchanged.
- Accelerate pointwise manual math in the general float graph too. Preserve
  NumPy fallback when the extension or new methods are absent.
- Keep dehaze and sharpening on their original spatial graph. Small changes
  before a float32 prefix sum or detail threshold can produce larger spatial
  differences; the new fused export route excludes these controls.

## Measurements

The application's recovery project was removed before the implementation
experiment. To make the comparison reproducible, subsequent runs use the first
frame of the same clip and a fixed automatic recipe estimated from every eighth
pixel, with the stated manual settings. This is not the original UDnet fit.
The reference explicitly uses `_manual_numpy` and bypasses the new export
dispatcher. These are individual local measurements, not long-clip guarantees.

| Color stage, one 4K frame | Reference | Optimized | Maximum difference before encoding |
| --- | ---: | ---: | ---: |
| Black point 36 | 2.860 s | 0.428 s | 1 RGB48 code |
| Black point 36, contrast 20, temperature 15, saturation 120 | 2.460 s | 0.387 s | 1 RGB48 code |

Spatial finishing follows the original implementation and is not claimed to
benefit from this optimization. Its sampled output was identical.

For a matched **12-frame** excerpt copied from the same 4K source, full exports
included stabilization and CPU libx264 CRF 17, preset fast, using the default
worker/memory limits:

| Export | Wall time |
| --- | ---: |
| Reference manual graph, black point 36 | 31.87 s |
| Optimized manual graph, black point 36 | 9.02 s |

That is approximately **3.5× faster / 72% less time** for this short sample.
Startup and stabilization are included. A complete original-clip export was
not rerun. H.264 files are not claimed byte-identical: native floating-point
rounding can change a 16-bit input sample, and lossy encoding may propagate it.

## Validation and retained experiments

Windows full suite with rebuilt native kernels: **362 passed, 16 skipped**.
New tests cover control extremes and combinations, strength/keyframe animation,
noncontiguous/read-only inputs, concurrent frames, block boundaries, NumPy-only
fallback, log/HDR compatibility, nonfinite input, and native buffer safety.
Pointwise export comparisons allow at most two 16-bit codes; the real-frame
comparisons above differed by at most one. Spatial fallback comparisons require
identical arrays. Linux execution remains for CI.

Retained local drivers and generated media are in the ignored directory
`benchmarks/results/store-bootstrap/`: `manual_blocks.py`, `manual_native.py`,
`manual_stages.py`, `manual_export.py`, `manual-native-report.json`, and
`manual-export/report.json`. The initial bounded-NumPy experiment reduced
2.01 seconds only to 1.70 seconds, which was insufficient; the final native
worker implementation produced the larger improvement above.
