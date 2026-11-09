# Windows and Microsoft Store (MSIX) packaging

The Windows packaging pipeline builds a standalone `--onedir` application
bundle (including `ffmpeg.exe`, `ffprobe.exe`, the optional C kernel
`uwcc._native`, and model preparation tools) and stages a Microsoft Store
**MSIX** package (`AppxManifest.xml` and Store visual assets).

Model weights and converted ONNX binaries are never bundled in the package.
Users can prepare models after installation directly from the desktop GUI
(**AI models…** in the toolbar or **Prepare local AI models…** in the
inspector) or from the command line via `uwcc model-tools`.

## Prerequisites

1. **Python 3.12+** (64-bit) with build dependencies and the DirectML runtime:
   ```powershell
   py -3.12 -m venv .venv-build
   .venv-build\Scripts\python.exe -m pip install --upgrade pip
   .venv-build\Scripts\python.exe -m pip install ".[ml-amd,vision,test]" pyinstaller
   ```
   Using `.[ml-amd]` bundles `onnxruntime-directml`, which accelerates ONNX
   inference across DirectX 12 capable AMD, Intel, and NVIDIA GPUs on Windows 10/11
   and falls back to CPU automatically.
2. **FFmpeg & ffprobe** (GPL static or shared Windows x64 binaries):
   Either place `ffmpeg.exe` and `ffprobe.exe` on `PATH` or pass their directory
   via `--ffmpeg-dir` / `UWCC_FFMPEG_DIR`.
3. **Optional C compiler** (Visual Studio Build Tools / Desktop development with C++)
   to compile `uwcc._native.pyd`.
4. **Windows 10/11 SDK** (`makeappx.exe`) to pack the final `.msix` archive.

## Build the standalone bundle and MSIX package

Run `tools/build_windows.py` from the repository root:

```powershell
.venv-build\Scripts\python.exe tools/build_windows.py --ffmpeg-dir C:\path\to\ffmpeg\bin
```

This produces:

- `dist/windows/uwcc/` — standalone runnable application directory with
  `uwcc-gui.exe`, `uwcc.exe`, `bin/ffmpeg.exe`, `bin/ffprobe.exe`, and `tools/`.
- `dist/windows/msix-layout/` — complete MSIX layout containing
  `AppxManifest.xml` and generated tile icons under `Assets/`.
- `dist/windows/uwcc_<version>_x64.msix` — packed MSIX archive (when
  `makeappx.exe` is found on `PATH` or under `C:\Program Files (x86)\Windows Kits\10\bin`).

Use `--layout-only` to stop after staging `dist/windows/msix-layout/` without
invoking `makeappx.exe`.

The Start menu and file associations launch `uwcc-gui.exe`, built with
PyInstaller `--windowed` so no terminal window is created. For the portable
bundle, open `uwcc-gui.exe` to launch the desktop app. The `uwcc.exe` executable
and installed `uwcc` command alias retain a console for command-line output
and automation. Both executables share the bundled runtime and dependencies.

The bundle also includes pure Python `uwcc/` sources beside `tools/`. The separate
model-converter Python needs these sources to import shared inference and
manifest helpers during CUDA graph validation; it cannot read application
modules from PyInstaller's archive. Native extensions/DLLs are not copied into
this source package, so the converter uses its own installed native dependencies.

The build refuses to run unless OpenCV (`cv2`) and ONNX Runtime are installed
in the build environment, so stabilization and local inference are never
silently omitted. Both are bundled with PyInstaller `--collect-all` (including
native DLLs such as `DirectML.dll`). Override with `--allow-missing-vision` /
`--allow-missing-ml`. Pass `--require-native` to fail if the `uwcc._native`
C extension does not compile.

The unreleased native export kernel builds AVX2 in a separate object on x86
targets. The package selects it automatically only when both CPU and OS support
AVX2; other CPUs and ARM64 use the fused scalar fallback. The whole application
is not compiled with AVX2 requirements. After building, verify the selected path
with the same interpreter:

```powershell
.venv\Scripts\python.exe -c "from uwcc import _native; print(_native.rgb48_backend())"
```

The result is `avx2` or `scalar`. `--require-native` ensures native compilation,
while a scalar result remains valid on unsupported CPUs. Build logs report an
AVX2-specific compiler failure if that object cannot be produced.

## Continuous integration

The `windows-package` job in `.github/workflows/tests.yml` runs on every push
and pull request on `windows-latest`. It builds the bundle and `.msix` with
`--require-native`, smoke-tests the frozen `uwcc.exe` (bundled FFmpeg, ONNX
Runtime, OpenCV, analysis with stabilization, and a render), and uploads the
`uwcc-windows-x64` artifact containing the unsigned `.msix` and a portable
`uwcc-windows-x64.zip`.

When a version tag (e.g. `v0.2.0`) is pushed, the `release` job automatically
attaches the `.msix` package, portable `.zip`, and Ubuntu `.deb` packages to a
GitHub Release.

## Publishing to the Microsoft Store

### Standard-user model setup verification

Test the installed MSIX under a standard Windows account with no separate
Python installation. In **AI models**, install a CPU or DirectML runtime and
the full model set, restart UWCC, and confirm inference works. Repeat setup to
exercise reuse of the downloaded runtime.

The external converter Python must not inherit PyInstaller's DLL directory or
bundle entries on `PATH`. Model setup temporarily clears the DLL directory and
restores it after launching the child; the child's working directory is its
own runtime directory. This avoids loading bundled native libraries from the
Store installation into the separate converter runtime.

If `python.exe` fails at startup with `0xC0000022` (access denied), record the
executable path and inspect Windows application/security logs or a Process
Monitor trace for denied executable/DLL access. The status alone does not
identify which file was denied. Model setup uses a per-user runtime and should
not require administrator privileges.

### Interrupted model-runtime setup

The embedded converter Python does not include `ensurepip`. If an interrupted
setup leaves Python without pip, recovery must enable `import site` in its
`python*._pth` file and bootstrap pip with `get-pip.py`, as on first installation.
The setup repair path does this automatically; ordinary virtual environments
continue to use `ensurepip`. This recovery fix is included in 0.4.3. After
installing that version, retry **Prepare models** with automatic runtime
installation enabled. It does not require deleting prepared models or installing
system Python.

The affected Store PC still failed in 0.5.0: external Python reported that
`get-pip.py` did not exist. MSIX can redirect AppData writes into a package's
`LocalCache`, so the app and an external interpreter may see different files at
the same logical path. The current fix creates the default UWCC storage root
and resolves its physical Windows path before selecting runtime and model
directories. Runtime overrides are resolved before interpreter launch too.
The installed diagnostic MSIX reproduced the missing-file failure with logical
paths and successfully installed pip with physical paths in a fresh runtime.
See [the diagnostic report](store-path-probe.md). Full model setup still needs
testing in a rebuilt UWCC package; the diagnostic did not install ML packages.

## Store submission

1. Register a developer account in [Microsoft Partner Center](https://partner.microsoft.com/dashboard)
   and reserve the application name (**Underwater Color Correction**).
2. Copy the assigned **Package/Identity/Name** and **Package/Identity/Publisher**
   values from Partner Center (**Product management → Product identity**).
3. Build the MSIX with those exact identity values:
   ```powershell
   .venv-build\Scripts\python.exe tools/build_windows.py `
     --ffmpeg-dir C:\path\to\ffmpeg\bin `
     --identity-name "YourPartnerCenterIdentityName" `
     --publisher "CN=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" `
     --publisher-display-name "Oliver Schwaneberg"
   ```
4. Upload `dist/windows/uwcc_<version>_x64.msix` to your submission in Partner
   Center. You do **not** need to sign the `.msix` yourself for Store uploads;
   Microsoft Partner Center signs the package automatically upon certification.
5. **Store Listing Details**:
   - **Privacy policy URL**: Provide https://github.com/Schwaneberg/UnderwaterColorCorrection/blob/main/PRIVACY.md.
   - **Restricted capability (`runFullTrust`) justification**: Declare that UWCC is a packaged Win32 application requiring standard file system access and process execution for bundled FFmpeg video encoding/decoding and DirectML GPU acceleration.
   - **Logos**: Upload data/icons/store/Square71x71Logo.png (71x71) and data/icons/store/Square300x300Logo.png (300x300).

