# NVIDIA video codec benchmark

Measured 2026-10-28 on the RTX 3060 Ti with driver 580.178.04 and Ubuntu FFmpeg
8.0.1. The existing AI-denoise batch was running during these measurements;
results reflect a shared workstation, not an isolated throughput test.

## Complete exports

The harness prepares a one-second, 100-frame 1920×1080 excerpt from the supplied
Hurghada C0001 camera clip. Excerpt preparation re-encodes H.264 at CRF 12 and AAC
to obtain an exact bounded duration and is excluded from timings. All modes then
use that same excerpt, neutral color controls, Rec.709, manual zero rotation,
and no AI denoise or stabilization. Timings include the full UWCC decode, CPU
color pipeline, video encoding, audio/metadata mux, and output validation.

Each mode has one unreported warm-up followed by three timed exports. The
frame counts, duration, audio streams, and output color tags are validated by
the production exporter.

| Encoder / decoder | Median elapsed | Output bytes | SSIM against CPU export |
| --- | ---: | ---: | ---: |
| libx264 / CPU | 7.395 s | 1,607,104 | reference |
| NVENC / CPU | 6.970 s | 5,774,757 | 0.986321 |
| NVENC / NVDEC with download | 7.378 s | 5,774,757 | 0.986321 |

NVENC with CPU decoding reduced median time by about 5.8% in this short complete
export. CPU frame transport, color work, and startup/validation still contribute
substantially; this is not an estimate of the gain for a long AI-denoise batch.
The different compressed output sizes demonstrate that NVENC's CQ 17 and
libx264's CRF 17 are not equivalent settings. SSIM here compares two compressed
outputs, not either output against an uncompressed ground-truth reference.

Both NVIDIA export variants have the same measured size and similarity. The
first decoded RGB camera frame also matches CPU output exactly (mean and
maximum absolute error both zero). During development, an explicit planar
conversion was needed after downloading NV12 to prevent a color-channel
mismatch. Dedicated 8-bit and 10-bit decoder pixel tests cover this path.

## Decode-only check

Decoding 64 original camera frames to full-size RGB48 with FFmpeg frame MD5
checksums, repeated three times, gave median 1.066 s on CPU and 1.693 s with
NVDEC, download, and planar conversion. The final-frame checksums matched.
GPU transfer and initialization overhead outweigh the decode saving in this
short test, supporting CPU decoding as the default. This does not establish
that CPU decoding is faster for all resolutions, codecs, or clip lengths.

## Reproduce complete exports

```bash
python3 -m benchmarks.video_codecs \
  --clip '/path/to/C0001.MP4' --seconds 1 --repeats 3 \
  --report benchmarks/results/video-codecs.json
```

Use the same CPU-only ONNX environment for each mode; FFmpeg's NVIDIA codecs
are independent of ONNX providers. Camera footage, generated clips, and JSON
reports stay local and are ignored by Git.
