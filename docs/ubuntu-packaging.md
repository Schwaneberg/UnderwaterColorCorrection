# Ubuntu package preparation

The initial package targets Ubuntu 26.04 (Resolute), starting with amd64. It
contains the desktop application, command-line tools, compiled C kernels,
launcher, and icon. Runtime dependencies come from Ubuntu: Python, NumPy,
Pillow, PyQt6, and FFmpeg. OpenCV and Qt Multimedia are recommended for
stabilization and audio preview. The Python wheel's PySide6 requirement is
replaced by an explicit PyQt6 dependency in Debian packaging.

The source builds three packages:

| Package | Purpose |
| --- | --- |
| `uwcc` | Desktop application, native kernels, CLI, and desktop integration |
| `uwcc-model-tools` | Model commands and Ubuntu CPU PyTorch/ONNX conversion dependencies |
| `uwcc-nvidia` | Private CUDA runtime setup command and NVIDIA launcher |

Model weights and converted ONNX binaries are not included. GPU wheels and
CUDA/cuDNN libraries are downloaded only by explicitly running NVIDIA setup.
There are no network operations or pip installations in package installation
hooks. The GUI model setup workflow remains subsequent work.

## Build a local package

On Ubuntu 26.04, install the packaging tools and declared build dependencies:

```bash
sudo apt update
sudo apt install build-essential dpkg-dev debhelper dh-python \
  pybuild-plugin-pyproject python3-all-dev python3-setuptools python3-wheel \
  python3-numpy python3-pil python3-pyqt6 python3-opencv python3-pytest \
  ffmpeg desktop-file-utils lintian
python3 tools/build_deb.py
lintian dist/debian/uwcc_*.changes
```

The helper copies tracked and non-ignored new working-tree files into a
temporary source directory, excluding downloaded sources, weights, and compiled
local binaries. It builds using `debian/control` and `debian/rules`, runs the
test suite offscreen against the built Python package, and puts unsigned
artifacts in `dist/debian/`. It does not change or clean the working checkout,
install packages, or access the network. Local uncommitted edits are included;
build official releases from a reviewed, committed tree.

`--output-dir /path/to/output` selects another artifact directory. Use an
ignored directory or a path outside the source checkout for build outputs.

Install the main package, substituting the version and architecture produced:

```bash
sudo apt install ./dist/debian/uwcc_0.1.0-1~ppa2_amd64.deb
uwcc --help
uwcc models
uwcc
```

It also appears as **Underwater Color Correction** in the application menu.
Opening a supported video through its **Open With** entry launches
`uwcc gui VIDEO`. Classical correction works without AI setup. To remove the
application, use `sudo apt remove uwcc`. User projects, preferences, and models
are kept.

The installed-package smoke test can be run on a test system after installing
the package:

```bash
debian/tests/smoke
```

It checks the installed launcher, icon, Qt binding, and compiled kernels, then
analyzes and renders a generated video from a temporary directory. For release
validation, also build in a clean Ubuntu chroot/container using `sbuild` or
equivalent; a host build does not prove all dependencies were declared.

## User model storage

Install model tools alongside the matching application version:

```bash
sudo apt install ./dist/debian/uwcc_0.1.0-1~ppa2_amd64.deb \
  ./dist/debian/uwcc-model-tools_0.1.0-1~ppa2_all.deb
uwcc-models prepare
uwcc models
```

`uwcc-models prepare` downloads pinned upstream sources and verified weights,
then converts and validates models locally. It runs without sudo using Ubuntu's
CPU PyTorch, ONNX, OpenCV, and ONNX Runtime packages. `--download-only` fetches
files without converting. Model-tool help is available with `uwcc-models --help`.

Installed applications discover models in:

```text
~/.local/share/uwcc/models/udnet/manifest.json
~/.local/share/uwcc/models/udnet/model.onnx
~/.local/share/uwcc/models/fastdvdnet/manifest.json
~/.local/share/uwcc/models/fastdvdnet/model.onnx
```

`XDG_DATA_HOME` replaces `~/.local/share`; `UWCC_MODEL_DIR` overrides the whole
models directory. On Windows the default is under `%LOCALAPPDATA%\uwcc\models`.
Discovery does not create directories or download files. Source checkouts also
recognize the existing repository `models/` directory after user storage;
an explicit `UWCC_MODEL_DIR` disables that fallback.

Keep each generated manifest beside its matching model. `uwcc models` prints
the configured directory and discovered manifests. The
[source-checkout model setup guide](../models/README.md) remains available for
development or other Python environments.

## NVIDIA runtime

Ubuntu's ONNX Runtime package provides CPU inference. `uwcc-nvidia` supplies a
private runtime installer and launcher, rather than claiming that an Ubuntu
CPU runtime enables CUDA. Install the matching packages:

```bash
sudo apt install ./dist/debian/uwcc_0.1.0-1~ppa2_amd64.deb \
  ./dist/debian/uwcc-model-tools_0.1.0-1~ppa2_all.deb \
  ./dist/debian/uwcc-nvidia_0.1.0-1~ppa2_amd64.deb
uwcc-nvidia-setup
uwcc-nvidia
```

Run setup as your desktop user, without sudo. It asks before downloading the
pinned `onnxruntime-gpu[cuda,cudnn]==1.31.0` runtime from PyPI, which may require
several GB. This CUDA 13 runtime requires a compatible GPU and NVIDIA driver;
the command does not install or change the driver. See the official
[CUDA provider compatibility guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

The private environment lives under
`~/.local/share/uwcc/runtimes/nvidia-py3.14-ort1.31.0` on Ubuntu 26.04; it uses
the apt-managed application, Qt binding, and NumPy. `XDG_DATA_HOME` relocates
user data, and `UWCC_NVIDIA_ENV` selects a different environment explicitly.
System Python is not modified. Setup reuses an already validated environment.
`--yes` accepts the download prompt for unattended setup.

Setup and launch both test a small real CUDA inference operation with CPU
fallback disabled. The launcher refuses to use an invalid runtime. Ordinary
`uwcc` remains the system-Python launcher; use `uwcc-nvidia` or the separate
**Underwater Color Correction (NVIDIA)** menu entry for GPU inference.
Weights still need `uwcc-models prepare` as described above.

For the optimized FastDVDnet CUDA graphs, run:

```bash
uwcc-models prepare-gpu
uwcc-nvidia model-tools validate-gpu
```

The first command prepares graphs on CPU; the second validates and registers
them using the GPU interpreter. This step follows standard model preparation.
Each command accepts `--models-dir` for another location. Package removal
preserves user models and the private runtime.

## Prepare a PPA source upload

The package uses Debian source format `3.0 (quilt)`. The build helper generates
the upstream tarball and source package, without local models or environments:

```bash
python3 tools/build_deb.py --source
```

The current changelog version is an initial PPA candidate, not an official
Ubuntu upload. Before publishing, review the maintainer identity, changelog,
source archive, licensing, test results, and supported architectures. Each PPA
upload needs a new version in `debian/changelog`.

After registering a signing key and creating your own PPA, install `devscripts`
and `dput`, then sign and upload the generated source changes:

```bash
debsign -k YOUR_KEY_ID dist/debian/uwcc_0.1.0-1~ppa2_source.changes
dput ppa:YOUR_LAUNCHPAD_NAME/YOUR_PPA dist/debian/uwcc_0.1.0-1~ppa2_source.changes
```

These are manual publishing steps; the build helper and CI do not upload.
Launchpad builds the `.deb` from source. A PPA does not confer inclusion in
Ubuntu Universe. See [Launchpad's build/upload instructions](https://ubuntu.com/docs/launchpad/user/how-to/packaging/build-deb-packages-in-launchpad/)
and the [Debian pybuild manual](https://manpages.debian.org/testing/dh-python/pybuild.1.en.html).
