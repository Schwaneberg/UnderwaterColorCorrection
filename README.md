# Underwater Color Correction

Underwater Color Correction (working name: **UWCC**) is a desktop
application for correcting underwater video with a simple, repeatable workflow.
It is aimed first at footage from Sony Alpha 6300 and 6500 cameras, including
clips recorded with red or magenta filters, but its media pipeline is intended
to accept ordinary standards-compliant video from other cameras as well.

This repository contains a runnable Python desktop application and headless
video-processing commands, together with the target v1 specifications. The
current development version is 0.1.0; it includes classical adaptive correction
and an optional local ONNX model interface. No trained model is bundled.

## Install on Ubuntu

Ubuntu 26.04 (amd64) packages are provided through the project's
[UWCC PPA](https://launchpad.net/~oliver-schwaneberg-z/+archive/ubuntu/uwcc).
Register the PPA, refresh the package index, and install the application:

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:oliver-schwaneberg-z/uwcc
sudo apt update
sudo apt install uwcc
uwcc
```

Optional packages provide model download/conversion tools and the NVIDIA
runtime setup and launcher:

```bash
sudo apt install uwcc-model-tools uwcc-nvidia
```

Model weights and the NVIDIA runtime require separate setup after installation;
follow the [model setup instructions](models/README.md) and
[Ubuntu package guide](docs/ubuntu-packaging.md). Package availability and build
progress are listed on the [PPA package page](https://launchpad.net/~oliver-schwaneberg-z/+archive/ubuntu/uwcc/+packages).

## Run from source

Ubuntu 26.04 `.deb` packaging is also available for local builds. See the
[Ubuntu packaging guide](docs/ubuntu-packaging.md) for build/install commands,
desktop integration, user model storage, and preparation of a PPA source upload.

With Python 3.12+, FFmpeg, NumPy, Pillow, and PySide6 or PyQt6 installed:

```bash
python3 -m uwcc
```

See the [user guide](docs/user-guide.md) for Ubuntu installation, the visual
workflow, batch jobs, optional models, CLI examples, and tests. Read the
[implementation status](docs/implementation-status.md) for the capabilities
and remaining v1 work; the specifications below describe the target release.

For faster exports, build the optional C kernels with
`python3 setup.py build_ext --inplace` (requires setuptools, Python development
headers, and a C compiler). Package installation also attempts this build;
NumPy provides the fallback. Video exports now offer stabilization and adjustable
spatial/temporal denoise in the inspector and CLI.

The [AI workflow guide](docs/ai-workflow.md) covers installed UDnet enhancement,
FastDVDnet temporal denoise, automatic levels/white balance, learning the look
of a corrected reference, background stabilization, and opt-in OpenAI/Claude
grading advice. Model weights are downloaded from their original upstream hosts
and converted locally; this repository contains manifests and license notices,
without redistributing model binaries. Follow the [model setup](models/README.md)
to prepare them.

Optional AI dependencies are available as `.[ml]` (CPU), `.[ml-nvidia]`
(NVIDIA CUDA on Linux/Windows), and `.[ml-amd]` (AMD via DirectML on Windows).
Install one ML extra per environment. See [GPU setup](docs/gpu-setup.md) for
installation commands, driver requirements, verification, and platform limits.

## Product direction

UWCC will provide:

- one-click, local ML-assisted correction with conservative, natural defaults;
- a small set of understandable finishing controls;
- temporally stable automatic adaptation and manual keyframes;
- Rec.709, Sony log, HLG, and PQ color-management workflows;
- reusable presets and batch processing;
- high-quality editing masters and H.264/H.265 delivery exports; and
- offline operation without accounts, telemetry, or cloud processing.

Ubuntu 26.04 is the required v1 platform. The design must remain portable to
Windows 11, but a Windows package is not a v1 release gate.

## Specifications

- [Product specification](docs/product-spec.md) — users, workflows, interface,
  functional requirements, error behavior, and acceptance criteria.
- [Technical specification](docs/technical-spec.md) — architecture, color and
  ML pipelines, persisted formats, dependencies, exports, packaging, and tests.

In the specifications, **shall** marks a release requirement, **should** marks a
strong recommendation, and **may** marks an optional behavior.

## Licensing policy

The application is intended to be released under
GPL-3.0-or-later. Application code, build tools, libraries, model definitions,
model weights, and bundled codec implementations must be open source and
redistributable under GPL-compatible terms. A user's separately installed GPU
driver or acceleration runtime may be used, but is never a required bundled
dependency.

See [LICENSE](LICENSE) for the complete GPL version 3 text. A release must also
include dependency notices, model provenance, and a software bill of materials.

