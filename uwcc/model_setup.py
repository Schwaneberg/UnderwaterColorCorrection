"""Installed entry point for explicit, user-owned model preparation."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import venv

from .errors import Cancelled, UWCCError
from .media import NO_WINDOW
from .project import model_directory, user_directory

SCRIPTS = {
    "prepare": "prepare_models.py",
    "prepare-gpu": "prepare_gpu_denoise.py",
    "validate-gpu": "validate_gpu_denoise.py",
    "prepare-orientation": "prepare_orientation_model.py",
}

SETUP_ACTIONS = {
    "all": "Prepare all standard models (UDnet, FastDVDnet, RightWayUp)",
    "prepare": "Prepare enhancement & denoise models (UDnet + FastDVDnet)",
    "prepare-orientation": "Download orientation model (RightWayUp Max INT8)",
    "download-only": "Download UDnet & FastDVDnet weights only (no conversion)",
    "prepare-gpu": "Prepare optional CUDA FastDVDnet graphs",
    "validate-gpu": "Validate optional CUDA FastDVDnet graphs",
}


def tool_directory():
    installed = Path("/usr/share/uwcc/model-tools")
    if installed.is_dir():
        return installed
    if getattr(sys, "frozen", False):
        for root in (Path(sys.executable).resolve().parent, Path(getattr(sys, "_MEIPASS", ""))):
            if (root / "tools").is_dir():
                return root / "tools"
    return Path(__file__).resolve().parent.parent / "tools"


def converter_environment_path():
    override = os.environ.get("UWCC_CONVERTER_ENV")
    if override:
        return Path(override).expanduser()
    abi = f"{sys.version_info.major}.{sys.version_info.minor}"
    return user_directory(create=False) / "runtimes" / f"converter-py{abi}"


def _venv_python(env_path: Path) -> Path:
    if os.name == "nt":
        # Embeddable Python distributions keep python.exe at the root.
        embedded = env_path / "python.exe"
        return embedded if embedded.is_file() else env_path / "Scripts/python.exe"
    return env_path / "bin/python"


CONVERTER_CHECK = "import torch, onnx, cv2, onnxruntime, numpy"


def _has_converter_modules():
    return all(importlib.util.find_spec(name) for name in ("torch", "onnx", "cv2", "onnxruntime"))


def converter_ready():
    if not getattr(sys, "frozen", False) and _has_converter_modules():
        return True
    return _venv_python(converter_environment_path()).is_file()


def _ready_manifest(manifest_path: str | Path | None) -> str | None:
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.is_file():
        return None
    try:
        from .project import read_json
        data = read_json(path)
        onnx_file = path.parent / data.get("file", "model.onnx")
        return str(path) if onnx_file.is_file() else None
    except Exception:
        return None


def model_status(models_dir=None):
    from .neural import builtin_model
    directory = Path(models_dir).expanduser() if models_dir else model_directory()
    models = {}
    for task, folder in (("enhance", "udnet"), ("denoise", "fastdvdnet"), ("orientation", "rightwayup")):
        if models_dir is not None:
            candidate = directory / folder / "manifest.json"
            models[task] = _ready_manifest(candidate)
        else:
            models[task] = _ready_manifest(builtin_model(task))
    has_ort = bool(importlib.util.find_spec("onnxruntime"))
    providers = []
    if has_ort:
        try:
            import onnxruntime as ort
            providers = list(ort.get_available_providers())
        except Exception:
            pass
    return {
        "models_dir": str(directory),
        "models": models,
        "onnxruntime": has_ort,
        "providers": providers,
        "opencv": bool(importlib.util.find_spec("cv2")),
        "converter_ready": converter_ready(),
    }


def _run_cancellable(args, cancel=None, env=None):
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            **NO_WINDOW,
        )
        try:
            while True:
                try:
                    code = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if cancel and cancel.is_set():
                        raise Cancelled("Operation canceled.")
            output.seek(0)
            text = output.read()[-12000:].decode("utf-8", "replace").strip()
            if code:
                raise UWCCError(text or f"Command failed with exit code {code}: {' '.join(args)}")
            return text
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _download(url, destination: Path, cancel=None):
    from urllib.request import urlopen
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            while block := response.read(1024 * 1024):
                if cancel and cancel.is_set():
                    raise Cancelled("Operation canceled.")
                output.write(block)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _bootstrap_embedded_python(destination: Path, cancel=None, progress=None) -> Path:
    """Install the official python.org embeddable distribution plus pip (Windows).

    Used when UWCC runs as a frozen/MSIX app on a PC without a Python installation.
    """
    import platform
    import zipfile
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "amd64"
    archive = destination.parent / f"python-{version}-embed-{arch}.zip"
    if progress:
        progress(0.05, f"Downloading Python {version} runtime for the model converter…")
    _download(f"https://www.python.org/ftp/python/{version}/python-{version}-embed-{arch}.zip", archive, cancel)
    shutil.rmtree(destination, ignore_errors=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink(missing_ok=True)
    # Enable site-packages so pip-installed modules are importable.
    for pth in destination.glob("python*._pth"):
        lines = pth.read_text(encoding="utf-8").splitlines()
        lines = ["import site" if line.strip() == "#import site" else line for line in lines]
        if "import site" not in lines:
            lines.append("import site")
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    python = destination / "python.exe"
    if progress:
        progress(0.1, "Installing pip into the model converter runtime…")
    get_pip = destination / "get-pip.py"
    _download("https://bootstrap.pypa.io/get-pip.py", get_pip, cancel)
    _run_cancellable([str(python), str(get_pip), "--no-warn-script-location"], cancel=cancel)
    get_pip.unlink(missing_ok=True)
    return python


def _create_converter_runtime(destination: Path, cancel=None, progress=None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(destination, ignore_errors=True)
    if not getattr(sys, "frozen", False):
        if progress:
            progress(0.05, "Creating private model converter environment…")
        venv.EnvBuilder(with_pip=True).create(destination)
        return _venv_python(destination)
    if os.name == "nt":
        return _bootstrap_embedded_python(destination, cancel, progress)
    host_python = shutil.which("python3") or shutil.which("python")
    if not host_python:
        raise UWCCError("Install Python 3 to set up the local PyTorch model converter.")
    if progress:
        progress(0.05, "Creating private model converter environment…")
    _run_cancellable([host_python, "-m", "venv", str(destination)], cancel=cancel)
    return _venv_python(destination)


def ensure_converter_python(bootstrap=True, cancel=None, progress=None):
    """Return a Python interpreter with PyTorch, ONNX, OpenCV and ONNX Runtime.

    Installs a private, user-owned converter runtime on first use (about 1 GB,
    CPU-only PyTorch). The application's own environment is never modified.
    """
    if not getattr(sys, "frozen", False) and _has_converter_modules():
        return sys.executable
    destination = converter_environment_path()
    python = _venv_python(destination)
    if python.is_file():
        check = subprocess.run([str(python), "-c", CONVERTER_CHECK], capture_output=True, text=True, **NO_WINDOW)
        if check.returncode == 0:
            return str(python)
    if not bootstrap:
        raise UWCCError("PyTorch and ONNX are required to convert UDnet/FastDVDnet weights.")
    if not python.is_file():
        python = _create_converter_runtime(destination, cancel, progress)
    if not python.is_file():
        raise UWCCError(f"Converter runtime python was not created at {python}")
    if progress:
        progress(0.15, "Installing CPU PyTorch (large download, this can take several minutes)…")
    _run_cancellable(
        [str(python), "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
         "--index-url", "https://download.pytorch.org/whl/cpu", "torch"],
        cancel=cancel,
    )
    if progress:
        progress(0.3, "Installing ONNX, OpenCV and ONNX Runtime…")
    _run_cancellable(
        [str(python), "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
         "--index-url", "https://pypi.org/simple",
         "onnx", "onnxscript", "opencv-python-headless", "onnxruntime", "numpy"],
        cancel=cancel,
    )
    check = subprocess.run([str(python), "-c", CONVERTER_CHECK], capture_output=True, text=True, **NO_WINDOW)
    if check.returncode:
        raise UWCCError(f"Model converter runtime validation failed:\n{check.stderr[-4000:]}")
    return str(python)


def run_frozen_script(script: Path, script_args: list[str]) -> int:
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(script), *script_args]
        if str(script.parent) not in sys.path:
            sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (1 if exc.code else 0)
    finally:
        sys.argv = old_argv
        sys.path = old_path


def run_setup(action="all", models_dir=None, bootstrap_converter=True, cancel=None, progress=None):
    if action not in SETUP_ACTIONS:
        raise UWCCError(f"Unknown model setup action: {action}")
    directory = (Path(models_dir) if models_dir else model_directory()).expanduser()
    if action == "all":
        steps = [("prepare", False), ("prepare-orientation", False)]
    elif action == "download-only":
        steps = [("prepare", True)]
    else:
        steps = [(action, False)]
    tools = tool_directory()
    outputs = []
    for index, (command, download_only) in enumerate(steps):
        if cancel and cancel.is_set():
            raise Cancelled("Operation canceled.")
        script = tools / SCRIPTS[command]
        if not script.is_file():
            raise UWCCError(f"Model setup script is missing: {script}")
        base_fraction = index / len(steps)
        needs_converter = command in ("prepare", "prepare-gpu") and not download_only
        if needs_converter:
            python = ensure_converter_python(
                bootstrap=bootstrap_converter,
                cancel=cancel,
                progress=(lambda f, msg: progress(base_fraction + f * 0.4 / len(steps), msg)) if progress else None,
            )
            cmd = [python, str(script), "--models-dir", str(directory)]
        elif getattr(sys, "frozen", False):
            cmd = [sys.executable, "model-tools", command, "--models-dir", str(directory)]
        else:
            cmd = [sys.executable, str(script), "--models-dir", str(directory)]
        if download_only:
            cmd.append("--download-only")
        if progress:
            label = "Downloading weights" if download_only else f"Running {command}"
            progress(base_fraction + (0.4 if needs_converter else 0.1) / len(steps), f"{label}…")
        text = _run_cancellable(cmd, cancel=cancel)
        if text:
            outputs.append(text)
    if progress:
        progress(1.0, "Model preparation complete.")
    return "\n".join(outputs) if outputs else f"Model setup ({action}) complete in {directory}."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(SCRIPTS))
    parser.add_argument("--models-dir", type=Path, default=model_directory())
    parser.add_argument("--download-only", action="store_true", help="Download without converting (prepare only)")
    args = parser.parse_args(argv)
    if args.download_only and args.command != "prepare":
        parser.error("--download-only is only available with prepare")
    script = tool_directory() / SCRIPTS[args.command]
    if not script.is_file():
        parser.exit(1, "Install uwcc-model-tools to use model preparation.\n")
    target_dir = str(args.models_dir.expanduser())
    if getattr(sys, "frozen", False):
        needs_converter = args.command in ("prepare", "prepare-gpu") and not args.download_only
        if not needs_converter:
            script_args = ["--models-dir", target_dir]
            if args.download_only:
                script_args.append("--download-only")
            try:
                return run_frozen_script(script, script_args)
            except KeyboardInterrupt:
                return 130
            except Exception as exc:
                print(f"UWCC model setup failed: {exc}", file=sys.stderr)
                return 1
        try:
            python = ensure_converter_python(bootstrap=True)
        except UWCCError as exc:
            parser.exit(1, f"{exc}\n")
        command = [python, str(script), "--models-dir", target_dir]
    else:
        command = [sys.executable, str(script), "--models-dir", target_dir]
    if args.download_only:
        command.append("--download-only")
    try:
        return subprocess.run(command).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
