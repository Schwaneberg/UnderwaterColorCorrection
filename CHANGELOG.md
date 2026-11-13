# Changelog

## 0.6.0 — 2026-11-13

### Added

- Load account models asynchronously into an editable cloud model dropdown when
  an API key is available, with automatic refresh on key/provider changes.
- Save provider-specific cloud API keys in local JSON configuration. Keys are
  masked by default, with a Show API key toggle and controls to save or remove
  them. Environment variables remain a fallback.
- Document local plaintext key storage in the privacy policy.

### Fixed

- Capture cloud advice dialog settings before starting the background request,
  preventing access to deleted Qt widgets on Windows.

### Validation

- GUI, cloud, and model discovery suites: 37 tests passed, including key
  persistence, masking, provider switching, authentication, pagination,
  cancellation, and dialog lifetime regressions.

## 0.5.4 — 2026-11-11

### Fixed

- Resolve Chocolatey/Scoop FFmpeg launchers at runtime as well as during
  packaging. Canceling a render now stops the real encoder instead of leaving
  its child holding temporary output files open (`WinError 32`). The 0.5.3
  cleanup synchronization alone did not address these launchers.
- Add runtime launcher-discovery and cancellation regressions to Windows CI.

### Validation

- Windows CI on the fix: 216 tests passed; standalone executable and MSIX
  built; bundled FFmpeg, native kernels, ONNX Runtime, OpenCV, analysis, and
  rendering smoke tests passed.
- Ubuntu and Debian package suites: 369 passed, 24 skipped each. Binary and
  source packages, lintian, and installed-package smoke tests passed.

## 0.5.3 — 2026-11-11

### Fixed

- Serialize Windows export cancellation and encoder cleanup so FFmpeg releases
  its output before temporary files are removed.
- Include `setup.py` in Debian's staged native build tests and update the Debian
  package version to match the application release.

### Validation

- Ubuntu 26.04 binary packages built successfully with the CI dependencies:
  360 tests passed, 24 skipped.
- The cancellation regression test fails with the previous cleanup code and
  passes with the fix. Native kernels rebuilt locally; 131 focused tests passed.
- Verification on the Windows CI runner remains pending.

## 0.5.2 — 2026-11-10

### Fixed

- Launch external Windows converter Python and its absolute path arguments with
  Win32 extended physical paths. This prevents `WinError 206` while pip installs
  deeply nested dependency files under MSIX AppData, without requiring the
  administrator-only long-path policy. Existing runtime/model locations remain
  unchanged. An isolated setuptools 78.1.0 installation reproduced the exact
  failing file and succeeded when retried through the patched launcher.

### Performance

- Fuse pointwise manual color controls into native kernels instead of building
  full-frame NumPy temporaries for each operation. SDR Rec.709 exports with
  black point, temperature/tint, exposure, contrast, shadows/highlights,
  saturation or hue use bounded blocks on a shared pool of at most four workers.
  Automatic LUT blending stays in float32 before manual controls; there is no
  intermediate RGB48 quantization or approximation of controls with a 3D LUT.
- Preserve the existing paths for spatial dehaze/sharpening and the neutral
  AVX2 export kernel. Builds without the updated extension retain NumPy fallback.
  Native floating-point rounding can change a final 16-bit sample slightly;
  numerical parity and spatial fallback tests cover the new dispatch.
- A matched 12-frame 4K C0022 export with stabilization and black point 36
  improved from 31.87 to 9.02 seconds. Real-frame color comparisons differed by
  at most one 16-bit code. See [measurements](docs/benchmarks/C0022-manual-color.md).
  Windows `.venv` full suite with rebuilt native kernels: 367 passed, 16 skipped.

## 0.5.1 — 2026-11-09

### Fixed

- Resolve default Windows user storage to its physical filesystem path before
  launching model-converter Python. MSIX AppData redirection can otherwise leave
  the external interpreter unable to find the downloaded `get-pip.py`, despite
  the Store application seeing it. Runtime and model paths share the resolved
  root. Backend-specific runtime overrides are resolved before launch as well.
- Installed diagnostic MSIX reproduced the logical-path failure and confirmed
  successful pip bootstrap using physical paths in a fresh embedded runtime.
  Full model installation in a rebuilt UWCC package remains to be verified.

### Validation

- Windows `.venv` full suite: 305 passed, 15 skipped. Regression coverage
  includes resolved storage roots and backend-specific converter overrides.

## 0.5.0 — 2026-11-08

### Performance

- Read decoded RGB48 directly into a reusable uint16 scratch buffer with
  `readinto`, eliminating the temporary bytes-to-bytearray copy. Yielded float
  frames retain independent storage. Windows frame pipes use dedicated buffers
  capped at 1 MiB; other platforms retain ordinary subprocess pipes.
- On C0045, the matched stabilized export improved from 63.47 to 50.56 seconds
  with identical MP4 bytes. The actual integrated path completed in 48.24
  seconds; see [raw-copy measurements](docs/benchmarks/C0045-raw-copies.md).
- Add a fused Rec.709 export kernel combining LUT interpolation, correction
  strength, clamping, and RGB48 quantization. AVX2 is selected by default after
  CPU/OS capability checks; unsupported CPUs use the portable fused scalar path.
  Manual finishing, log/HDR, zero strength, and missing/non-LUT analysis retain
  their existing correction graph. Preview behavior is unchanged.
- Compile AVX2 separately from baseline native code, disable cross-object
  optimization for its kernel, and include native sources/headers in source
  distributions. Ubuntu and Windows CI now build and exercise native kernels.
- Detect stabilization features in cropped proxy regions with a neighborhood
  halo, retaining spatial quotas and full-proxy point coordinates. Add parity
  tests for cell/image boundaries, scene cuts, crop zoom, and warped pixels.
  On C0045, motion analysis improved from 28.16 to 13.25 seconds and stabilized
  export from 99.98 to 90.31 seconds; the output MP4 was byte-identical.
- Reduce NumPy LUT interpolation memory traffic with bounded float32 blocks and
  direct tetrahedron-corner selection. Reuse decode-normalization and
  export-rounding buffers. Encoder settings and native LUT kernels are unchanged.
- Allow export benchmarks to run without the optional C extension; an explicit
  `--backend native` still requires a compiled module.
- On a 51-frame C0045 camera sample, NumPy H.264 export improved from 27.27 to
  16.38 seconds (40% less time). The complete optimized 1,032-frame clip exported
  in 137.08 seconds. See [the benchmark report](docs/benchmarks/C0045-export.md)
  for scope, numerical differences, and limitations.
- After installing MSVC and compiling the existing C extension, the same full
  clip exported in 84.66 seconds (38% less time than optimized NumPy). This is
  the benefit of enabling existing kernels, not a change to their LUT algorithm.

### Validation

- Production default AVX2 exported stabilized C0045 in 59.90 seconds with a
  byte-identical MP4 compared with the earlier 90.84-second reference export.
  Forced scalar calls and an independently built scalar-only module matched
  AVX2 output. Linux execution remains for CI.
- Windows `.venv` full suite with compiled kernels: 302 passed, 15 skipped.
  Frame-pipe tests cover large transfers, EOF, startup cleanup, partial reads,
  cancellation, and independent frame ownership.
  LUT reference tests cover uniform/log interpolation, ties, edges, and block
  boundaries. Six real color frames differ by at most one 16-bit code value
  before encoding. Stabilization tests retain legacy feature points, motion,
  crop zoom, and synthetic warped pixels.

## 0.4.3 — 2026-11-08

### Fixed

- Repair missing pip in an existing embedded model-converter Python using
  `get-pip.py` instead of `ensurepip`, which the embedded runtime does not ship.
  This addresses the Store setup error `No module named ensurepip` after an
  incomplete converter bootstrap. Regular virtual environments retain ensurepip.
- Share isolated pip bootstrap between fresh embedded setup and recovery,
  re-enable site-packages, and clean up the bootstrap script on failure/cancel.

### Validation

- Windows `.venv` full suite: 231 passed, 23 skipped. Coverage includes embedded
  recovery, regular ensurepip repair, and cleanup after failure/cancellation.
- Retrying model setup on the affected PC with a rebuilt package remains needed.

## 0.4.2 — 2026-11-08

### Fixed

- Isolate external model-converter Python launches from the frozen Windows
  application's DLL directory and bundled PATH entries. Restore the parent DLL
  search directory after launch, including when process creation fails, and
  correctly handle an empty DLL directory.
- Use the converter runtime directory as the external child's working directory.
  This addresses a possible cause of the standard-user Microsoft Store
  `python.exe` startup failure (`0xC0000022`); affected-machine confirmation is
  still required.
- Include pure Python `uwcc` sources alongside Windows model tools so the
  separate converter runtime can import inference/manifest helpers during CUDA
  FastDVDnet validation, fixing `ModuleNotFoundError: No module named 'uwcc'`.
- Use the long-supported `-vf` option for the short VFR filter graph instead of
  deprecated `-filter_script:v`, removed by FFmpeg 9.0. Keep per-frame timestamp
  commands in a separate file so command length does not grow with the clip.
- Resolve the timestamp command file relative to the encoder's private working
  directory, avoiding filter-language escaping of Windows drive letters and
  destination folders with spaces, apostrophes, or non-ASCII characters.

### Tests

- Mark Linux `uwcc-nvidia` setup/launcher tests as Linux-only, preserving their
  Ubuntu CI coverage while skipping POSIX assumptions on Windows.
- Keep runtime-path tests cross-platform: verify XDG data paths on Linux and
  LocalAppData on Windows, plus explicit runtime overrides without disk writes.
- Add converter launch isolation/restoration and source-only packaging coverage,
  including an isolated-interpreter CUDA-validation prerequisite check.
- Full Windows `.venv` suite with FFmpeg 9.0.1: **228 passed, 23 skipped, no
  failures/errors**. Both VFR timestamp round trips pass, including finishing
  effects and a destination containing spaces, an apostrophe, and non-ASCII text.
- Documentation file links and Git diff whitespace checks passed.

### Documentation

- Refresh product and technical specifications against the current implementation,
  distinguishing implemented behavior from outstanding v1 acceptance targets.
- Document Windows distribution, explicit model/runtime setup, optional cloud
  advice, actual project/preset formats, and known validation limits.
- Retire the duplicated implementation-status checklist in favor of maintained
  specification sections; keep its path as a compatibility pointer.
- Label preview benchmarks and recorded test results as historical evidence.
- Ignore the local Windows Store packaging helper script.

### Remaining validation

- A rebuilt MSIX still needs standard-user model setup verification on the
  affected PC. The reported access-denied root cause is not yet confirmed.
- Actual CUDA graph accuracy validation and DirectML hardware validation remain
  separate from the automated prerequisite and packaging tests.
- Older FFmpeg binaries and Linux execution were not tested locally; the filter
  graph uses established `-vf`/`sendcmd` syntax shared with earlier releases.
  Linux tests remain enabled in the existing Ubuntu CI job.

## 0.4.1 — 2026-11-06

### Improved

- Avoid a redundant color-space round trip for the original Rec.709 preview.
- Process preview color correction across up to four CPU workers. Keep dehaze
  and sharpness on the full frame to preserve spatial filtering.
- Limit preview decoding and timestamp extraction to four threads to reduce CPU
  contention. Export processing and thread settings are unchanged.
- Skip overdue preview frames instead of accumulating delay against the
  playback and audio clock.
- Add reproducible preview benchmarks and performance documentation.

### Validation and limitations

- Playback, color, native-kernel parity, and selected media tests passed.
- An offscreen preview test with a 1080p, 50 fps clip displayed 292 frames over
  eight seconds of video in 8.34 seconds. Full 50 fps playback is not guaranteed;
  manual adjustments and spatial filters can reduce throughput further.
- Visible-window playback and audible synchronization still need manual checks.
- The previously documented Windows variable-frame-rate export failures remain.

## 0.4.0 — 2026-11-06

### Added

- Automatically open AI model setup at startup when no local models are installed,
  with guidance explaining the benefits and how to get started. Setup remains
  optional; users can close the window and install models later.
- Clearly label the full setup option as **Install all models (Full feature set,
  best performance)**, list UDnet, FastDVDnet, and RightWayUp Max INT8, and warn
  that this option requires the biggest download.

### Fixed

- Point the VS Code launch configuration at the existing `.venv-build` environment.

### Validation

- All 41 selected Windows packaging, model setup, execution runtime, provider,
  GPU validation, and startup-condition tests passed.
- The new setup features were manually tested and accepted by the maintainer.

## 0.3.3 — 2026-11-06

### Fixed

- Launch the Windows MSIX Start menu entry and file associations through a
  windowed executable so no terminal window appears. Retain the console
  executable and `uwcc` command alias for command-line use.
- Find the bundled `convert_models.py` when preparing standard models with
  Windows' isolated embedded Python runtime. Apply the same fix to optional
  GPU model preparation, including existing converter runtimes.

### Validation

- All 24 Windows packaging, model preparation, and model setup tests passed.
- Add isolated-interpreter regression tests for model preparation and packaging
  tests for the GUI and console entry points.
- Check the compiled Windows executable subsystem flags in CI.
- Full model downloads, conversion, and installed MSIX launch validation remain
  to be verified with the rebuilt package.

## 0.3.2 — 2026-11-03

### Added

- Privacy policy and documentation for the Microsoft Store submission.
- Additional 71×71 and 300×300 Store icons bundled with MSIX assets.
- CPU, NVIDIA CUDA, and AMD / DirectML (Windows) radio buttons in the AI models
  dialog, with the execution choice remembered for subsequent launches.
- Separate private runtimes for each backend. CUDA installs CUDA-enabled PyTorch
  and ONNX Runtime; DirectML uses GPU ONNX inference and CPU PyTorch conversion.
- Runtime-only setup without rebuilding models, also available through
  `uwcc model-tools runtime --backend cpu|cuda|directml`.
- Regression coverage for runtime installation, saved preferences, provider
  selection, the setup dialog, and CUDA validation prerequisites.

### Fixed

- Recover model converter environments missing pip after an incomplete setup.
- Isolate converter installation and validation from inherited Python settings.
- Run CUDA validation in the selected runtime and provide actionable errors for
  CPU-only installations, forced CPU execution, and CUDA initialization failures.
- Restrict optional CUDA FastDVDnet graph actions to NVIDIA CUDA selection.
- Make the VS Code Windows launch configuration use `.venv` explicitly and
  disable subprocess debugger injection during model setup.
- Synchronize application, Python package, and Debian release versions to 0.3.2.

### Upgrade notes

Open **AI models…**, choose an execution backend, and select **Install or switch
execution runtime only** to configure existing models. Restart UWCC after setup.
GPU setup can download several GB and requires a compatible driver. DirectML is
Windows-only and has not been hardware-validated in this release. ONNX itself
does not have a separate GPU package. See [GPU setup](docs/gpu-setup.md).

### Validation and known limitations

- 51 Windows packaging, execution, model setup, provider, and GUI tests passed.
- The full local Windows suite reported 198 passed, 14 skipped, 3 failed, and
  9 errors. Failures are in unchanged variable-frame-rate export tests and
  Linux-specific NVIDIA launcher tests (POSIX user IDs and XDG paths).
  The same 3 failures and 9 errors were reproduced against the previous commit.
- Windows variable-frame-rate export remains a known issue in the local test
  environment. Linux package builds and release assets are validated by CI.
