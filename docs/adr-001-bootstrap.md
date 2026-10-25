# ADR 001: Runnable offline bootstrap

Date: 2026-10-22  
Status: Accepted for 0.1; does not relax v1 release gates

## Context

The implementation environment has Python 3.14, NumPy, Pillow, PyQt6, pytest,
and FFmpeg 8. Package-network access is unavailable. PySide6, PyAV,
OpenColorIO, OpenCV, ONNX Runtime, and a licensed converted model are not
installed. The original specifications describe the target v1, not the 0.1
implementation delivered in this environment.

## Decisions

- Prefer PySide6 when installed; allow GPL-compatible PyQt6 from Ubuntu packages.
  Use Qt Widgets for the initial workspace. Keep rendering and processing
  independent of the widget layer so a Qt Quick interface remains possible.
- Use bounded FFmpeg subprocess pipes for 16-bit decode and encode, rather than
  require PyAV. Use source presentation timestamps and FFmpeg `sendcmd`/`setpts`
  when the raw input cadence needs to be mapped back to variable source cadence.
- Implement reviewed transfer functions and gamut matrices directly in NumPy.
  The working space remains ACEScg. The OpenColorIO replacement is provisional;
  an embedded OCIO configuration and certified display transforms remain v1 work.
  Sony log decode consumes normalized stored code values, so the media adapter
  restores the legal-range offset/scale after FFmpeg's RGB expansion. Reference
  black/gray values follow [Sony's technical summary](https://pro.sony/s3/cms-static-content/uploadfile/06/1237494271406.pdf).
- Ship a classical adaptive correction engine, explicitly labeled. Provide an
  optional ONNX reference-image interface with checksum and license validation.
  Do not redistribute an unverified checkpoint or silently claim ML inference.
- Fit a regularized global matrix to an optional ML reference and evaluate it
  through a 17-cube LUT. This deliberately restricts the model to a global color
  change. A higher-order fit and confidence calibration require real footage.
- Store compact generated curves inline in a validated schema-1 JSON project.
  Implemented schema examples are in the user guide. The earlier specification
  examples describe v1 and are not files that 0.1 can import.

## Consequences

The application and headless pipeline run with Ubuntu-provided software and
require no runtime networking. The CLI and GUI share correction and export
code. CPU processing is the baseline; 1080p GPU performance, a bundled model,
real-world visual acceptance, AppImage packaging, and a complete dependency
audit are still required before this can be called v1.
