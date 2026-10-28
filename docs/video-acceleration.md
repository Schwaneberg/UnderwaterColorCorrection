# GPU video encoding and decoding

UWCC 0.2.0 adds NVIDIA video codec acceleration independently of the ONNX AI
runtime. FFmpeg must include the relevant NVIDIA codec support, and the driver
must support the installed FFmpeg build. No CUDA PyTorch installation is needed
for video codecs. Other GPUs continue to use software video codecs.

## Defaults and controls

The inspector has **Export video encoder** and **Export video decoder** choices.
These are saved with projects and presets and copied to new batch items.
Both encoding and decoding default to CPU, including when launching with
`uwcc-nvidia`. Hardware video acceleration requires an explicit choice.

| Setting | Behavior |
| --- | --- |
| Encoder `cpu` (default) | Use libx264/libx265 for H.264/HEVC. |
| Encoder `nvenc` | Require NVIDIA H.264/HEVC encoding; report an error if unavailable or if another export format is selected. |
| Encoder `auto` (opt-in) | Test an actual NVIDIA encoding session; use NVENC for H.264/HEVC when successful, otherwise use CPU encoding. Other export formats use CPU. |
| Decoder `cpu` (default) | Decode on the CPU. |
| Decoder `auto` | Test NVIDIA decoding for this source and fall back to CPU if unavailable or unsupported. |
| Decoder `cuda` | Require NVIDIA decoding and report an error if unsupported. |

The progress log reports the chosen encoder and decoder, including CPU fallback.
The checks run before streaming and are cached for the application session.
An error after streaming has started fails the export; UWCC does not silently
restart a long export or change codecs halfway through. Output publication
remains atomic, and source files are preserved.

NVENC uses preset `p6`, high-quality tuning, and variable bitrate with a target
quality of 17 for H.264 or 18 for HEVC. HEVC retains 10-bit output and HDR signaling.
These target-quality values are not equivalent to x264/x265 CRF values; output
size and compression differ. DNxHR, ProRes, and FFV1 continue to use CPU encoders.

## Why decoding remains on CPU by default

Color processing, RGB transport, and stabilization still need system-memory
frames. NVDEC output is downloaded as NV12 for 8-bit 4:2:0 sources or P010 for
10-bit 4:2:0 sources, then passed through the existing CPU filters and RGB48
transport. Other source layouts fall back in auto mode. Sampling for preview,
color analysis, and orientation remains on CPU; the decoder setting controls
export and its stabilization prepass.

A three-repeat test on the RTX 3060 Ti decoded 64 original 1080p frames from
C0001 into RGB48 with frame checksums: CPU median 1.066 s, NVDEC plus download
and planar conversion 1.693 s (about 59% longer). This includes process initialization and conversion.
It is a short-clip measurement on this machine, not a universal decoder ranking.
The downloaded frame is explicitly converted to the software decoder's planar
layout before RGB conversion. This avoids a substantial NV12-to-RGB48 color
mismatch found during development. The tested camera frame matches CPU pixels
exactly, and hardware tests compare 8-bit and 10-bit frame sequences.
See [codec benchmarks](benchmarks/video-codecs.md) for complete-export results.

## CLI

```bash
# CPU encoding and decoding (the defaults for new projects):
uwcc-nvidia render project.json corrected.mp4 --video-encoder cpu --video-decoder cpu

# Opt in to NVIDIA encoding with CPU fallback:
uwcc-nvidia render project.json corrected.mp4 --video-encoder auto

# Require GPU decoding and encoding:
uwcc-nvidia render project.json corrected.mp4 --video-encoder nvenc --video-decoder cuda

# Force software codecs:
uwcc render project.json corrected.mp4 --video-encoder cpu --video-decoder cpu

# Each batch item keeps the chosen preferences:
uwcc-nvidia batch clips/*.MP4 --output-dir corrected --ai-denoise --denoise 3 \
  --stabilize --video-encoder cpu --video-decoder cpu
```

`analyze` also accepts the flags so a saved project can carry export preferences.
Use `--format hevc` for 10-bit HEVC output. `uwcc` itself can use NVENC when
explicitly selected; `uwcc-nvidia` additionally enables the separately installed CUDA AI
runtime. See [NVIDIA's FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/)
for the hardware encode/decode mechanisms.

## Verification

Ordinary tests cover auto fallback, explicit failures, cached checks and
cancellation, source layout handling, preference persistence, and GUI undo.
On a configured NVIDIA machine, run the real codec tests:

```bash
UWCC_TEST_VIDEO_GPU=1 QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest tests/test_video_codecs.py -q
```

These exercise actual NVENC/NVDEC sessions, rotated dimensions, audio, HLG/PQ
tags, variable frame timing, and sample aspect ratio.
