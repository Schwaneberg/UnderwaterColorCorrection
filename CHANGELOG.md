# Changelog

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
