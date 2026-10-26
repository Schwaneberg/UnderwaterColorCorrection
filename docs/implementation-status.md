# Implementation status — 0.1.0

This is a runnable development application. The product and technical
specifications remain the v1 target; this file records actual implementation.

## Implemented

- Initial Ubuntu 26.04 Debian source/binary packaging, desktop launcher/icon,
  native Python/Qt/FFmpeg dependencies, build CI and installed-package smoke test.
  User model storage follows XDG data paths; source-checkout models still work.
- Optional `uwcc-model-tools` package for CPU model download/conversion, and
  `uwcc-nvidia` for explicit user-owned CUDA runtime installation and launch.

- Qt desktop workspace with split/original/corrected comparison, draggable split,
  proxy zoom/panning, timeline seeking, and background proxy playback.
- Preview-resolution histogram, waveform, RGB parade, and vectorscope.
- Twelve manual controls, per-control reset, linear keyframes, keyframe jump and
  deletion, and undo/redo.
- Source probing, explicit profile confirmation, capture-filter hints, and
  Rec.709, S-Log2/S-Gamut, S-Log3/S-Gamut3.Cine, HLG, and PQ transforms.
- Float32 ACEScg working values, 16-bit FFmpeg frame transport, a shared CPU
  preview/render graph, SDR tone mapping, and 10-bit HDR export signaling.
- Adaptive classical balance, conservative red-channel limits, scene-cut
  detection, and bidirectional smoothing confined to each detected shot.
- Optional local ONNX reference generator with manifest license/checksum checks,
  RGB tensor validation, CPU fallback, and constrained reference-transform fitting.
- Versioned project/preset JSON, rational keyframe timestamps, atomic saves,
  source fingerprints, and delayed recovery autosaves.
- Batch jobs with independent analysis, collision-safe output names, persistent
  state, pause/resume/retry, and per-item diagnostics. Resuming restarts the
  current item, not the middle of a video.
- Full-resolution H.264, HEVC Main10, DNxHR HQX, ProRes 422 HQ, and FFV1 output.
  Audio is preserved (AAC is copied for delivery; masters use PCM/FLAC).
- Source frame timestamps retained for constant and variable frame rates;
  post-render checks for frame count, duration, audio-stream count, and color tags.
- Atomic final-output publication, cancelable FFmpeg processes, and preservation
  of an existing output when an operation fails.
- Headless probe, analyze, render, batch, and source-relink commands.
- Optional C kernels for LUT interpolation and fused SDR output conversion,
  NumPy fallback, neutral-control bypass, float32 automatic gains, and bounded
  faster H.264/HEVC encoder settings. A reproducible benchmark is included.
- Export-only OpenCV background stabilization with robust motion fitting,
  two-pass fixed crop/zoom that excludes exposed edges, and float RGB warping;
  conventional spatial/temporal denoise or optional tiled,
  five-frame FastDVDnet neural denoise. GUI/CLI, presets, and batches supported.
- Installed, pinned MIT-licensed UDnet and FastDVDnet ONNX models in this local
  checkout; reproducible download/conversion tools with checksums, provenance,
  tensor-only loading, and PyTorch/ONNX parity checks. Python packages do not
  automatically bundle or download weights.
- Adaptive levels, gray-edge white balance, contrast/vibrance finishing, neural
  color references projected into temporal RGB LUTs, and corrected-reference
  learning. SDR fits are tested against the actual C0201 reference video.
- Explicitly opt-in OpenAI/Claude proxy-frame grading advice with bounded,
  validated proposals; provider contracts tested with mocks, no live API run.
- Bounded parallel color processing and multithreaded FFmpeg decode/filter/
  encode/timing scans, preserving temporal ordering and cancellation.

## Remaining before v1

- GUI preparation of upstream model downloads and local conversion,
  broader real-camera/color
  chart validation, neural stabilization, full-frame spatial neural enhancement,
  calibrated confidence, and certified temporal/flicker/quality measurements.
  Actual C0201 comparisons and a user-corrected reference are available; they
  do not establish general color accuracy or parity with AquaColorFix.
- OpenColorIO configuration, calibrated display management, fully reviewed
  HDR/log handling, HDR monitor output, and GPU correction shaders.
- More sophisticated ML LUT fitting, calibrated model/filter confidence,
  optical-flow-aware temporal quality evaluation, and progressive analysis
  previews. Analysis reports progress but publishes its curve only on completion.
- Complete recovery/relink GUI, recent-project list, localization, accessibility
  audit, hardware encoding, runtime resource budgets, and disk-space estimation.
- Repeatable AppImage/Windows installers, dependency locks and SBOM, license
  provenance for all distributed binaries, and clean-machine validation.
- Exact final-packet duration for unusual variable-rate sources and stream-level
  audio-offset validation across all containers. Audio-preview support depends
  on the optional Qt Multimedia module; otherwise preview is silent and says so.
- Photo editing and timeline assembly are intentionally outside the v1 scope.

## Persistence compatibility

0.1 uses `schema_version: 1`, a source path/fingerprint, a `grade` object with
`values` and rational-time `keyframes`, and an optional inline `analysis` object.
Model paths are not automatically trusted when importing projects or presets;
select a model again to re-analyze. Analysis already saved in the project is
fully usable without the original model file.

Do not use the illustrative JSON in the original technical specification as an
input fixture: it is the planned v1 wire format, not the 0.1 serializer.
