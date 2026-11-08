"""Installed entry point for explicit, user-owned model preparation."""
import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
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
    "runtime": "Install or switch execution runtime only",
    "all": "Install all models (Full feature set, best performance)",
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
_DLL_SEARCH_LOCK = threading.RLock()


@contextmanager
def _system_dll_search():
    """Keep PyInstaller's DLL directory out of external Python children."""
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_directory = kernel32.GetDllDirectoryW
    get_directory.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p]
    get_directory.restype = ctypes.c_uint32
    set_directory = kernel32.SetDllDirectoryW
    set_directory.argtypes = [ctypes.c_wchar_p]
    set_directory.restype = ctypes.c_int
    with _DLL_SEARCH_LOCK:
        ctypes.set_last_error(0)
        size = get_directory(0, None)
        if not size and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        previous = ctypes.create_unicode_buffer(size + 1)
        if size:
            ctypes.set_last_error(0)
            # An empty DLL directory returns zero even when the first query
            # requested one character for its terminating NUL.
            if not get_directory(len(previous), previous) and ctypes.get_last_error():
                raise ctypes.WinError(ctypes.get_last_error())
        if not set_directory(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            if not set_directory(previous.value or None):
                raise ctypes.WinError(ctypes.get_last_error())


def _converter_process(operation, args, **kwargs):
    if not (os.name == "nt" and getattr(sys, "frozen", False)) or Path(args[0]) == Path(sys.executable):
        return operation(args, **kwargs)
    # Qt/PyInstaller may also add bundle directories to PATH. External Python
    # must load its own native libraries, including inside a Store installation.
    env = dict(kwargs.get("env") if kwargs.get("env") is not None else os.environ)
    roots = [Path(sys.executable).resolve().parent]
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(sys._MEIPASS).resolve())
    env["PATH"] = os.pathsep.join(
        entry for entry in env.get("PATH", "").split(os.pathsep)
        if entry and not any(Path(entry).resolve().is_relative_to(root) for root in roots)
    )
    kwargs["env"] = env
    kwargs.setdefault("cwd", str(Path(args[0]).resolve().parent))
    with _system_dll_search():
        return operation(args, **kwargs)


def _has_converter_modules():
    return all(importlib.util.find_spec(name) for name in ("torch", "onnx", "cv2", "onnxruntime"))


def converter_ready(backend=None):
    if backend is None and not getattr(sys, "frozen", False) and _has_converter_modules():
        return True
    destination = converter_environment_path()
    if backend:
        destination = destination.with_name(destination.name + f"-{backend}")
    return _venv_python(destination).is_file()


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
        process = _converter_process(subprocess.Popen,
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


def _install_embedded_pip(python: Path, cancel=None, progress=None):
    """Bootstrap/repair pip without ensurepip, absent from embedded Python."""
    destination = python.parent
    # Re-enable site-packages even after an interrupted first-time setup.
    for pth in destination.glob("python*._pth"):
        lines = pth.read_text(encoding="utf-8").splitlines()
        lines = ["import site" if line.strip() == "#import site" else line for line in lines]
        if "import site" not in lines:
            lines.append("import site")
        pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if progress:
        progress(0.1, "Installing pip into the model converter runtime…")
    get_pip = destination / "get-pip.py"
    try:
        _download("https://bootstrap.pypa.io/get-pip.py", get_pip, cancel)
        _run_cancellable(
            [str(python), "-I", str(get_pip), "--isolated", "--disable-pip-version-check",
             "--no-warn-script-location"], cancel=cancel,
        )
    finally:
        get_pip.unlink(missing_ok=True)


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
    python = destination / "python.exe"
    _install_embedded_pip(python, cancel, progress)
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


def ensure_converter_python(bootstrap=True, cancel=None, progress=None, backend=None):
    """Return a Python interpreter with PyTorch, ONNX, OpenCV and ONNX Runtime.

    Installs a private, user-owned runtime on first use. An explicit backend
    selects matching inference packages and CPU or CUDA PyTorch. The application's
    own environment is never modified.
    """
    if backend not in (None, "cpu", "cuda", "directml"):
        raise UWCCError(f"Unknown execution backend: {backend}")
    if backend == "directml" and sys.platform != "win32":
        raise UWCCError("DirectML requires Windows. Choose CPU or NVIDIA CUDA.")
    if backend is None and not getattr(sys, "frozen", False) and _has_converter_modules():
        return sys.executable
    destination = converter_environment_path()
    if backend:
        destination = destination.with_name(destination.name + f"-{backend}")
    python = _venv_python(destination)
    validation = CONVERTER_CHECK
    if backend:
        provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "directml": "DmlExecutionProvider"}[backend]
        validation += f"; assert '{provider}' in onnxruntime.get_available_providers()"
        validation += "; assert " + ("torch.version.cuda is not None" if backend == "cuda" else "torch.version.cuda is None")
    if python.is_file():
        check = _converter_process(subprocess.run, [str(python), "-I", "-c", validation], capture_output=True, text=True, **NO_WINDOW)
        if check.returncode == 0:
            return str(python)
    if not bootstrap:
        raise UWCCError("The selected PyTorch/ONNX runtime is missing or incomplete. Enable automatic installation.")
    if not python.is_file():
        python = _create_converter_runtime(destination, cancel, progress)
    if not python.is_file():
        raise UWCCError(f"Converter runtime python was not created at {python}")
    # A previous interrupted/debugged bootstrap can leave Python without pip.
    # Repair it before installing packages into this private environment.
    pip_check = _converter_process(subprocess.run,
        [str(python), "-I", "-m", "pip", "--version"],
        capture_output=True, text=True, **NO_WINDOW,
    )
    if pip_check.returncode:
        if any(python.parent.glob("python*._pth")):
            _install_embedded_pip(python, cancel, progress)
        else:
            _run_cancellable(
                [str(python), "-I", "-m", "ensurepip", "--upgrade"], cancel=cancel,
            )
    if progress:
        progress(0.15, f"Installing {'CUDA' if backend == 'cuda' else 'CPU'} PyTorch (large download, this can take several minutes)…")
    torch_index = "cu130" if backend == "cuda" else "cpu"
    _run_cancellable(
        [str(python), "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
         "--index-url", f"https://download.pytorch.org/whl/{torch_index}", "--upgrade", "--force-reinstall", "torch"],
        cancel=cancel,
    )
    if progress:
        progress(0.3, "Installing ONNX, OpenCV and ONNX Runtime…")
    ort_package = {"cuda": "onnxruntime-gpu[cuda,cudnn]>=1.21", "directml": "onnxruntime-directml>=1.17"}.get(backend, "onnxruntime")
    if backend:
        # Interrupted installs must never leave overlapping ORT distributions.
        _run_cancellable(
            [str(python), "-I", "-m", "pip", "--isolated", "uninstall", "-y",
             "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"], cancel=cancel,
        )
    _run_cancellable(
        [str(python), "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
         "--index-url", "https://pypi.org/simple",
         "onnx", "onnxscript", "opencv-python-headless", ort_package, "numpy"],
        cancel=cancel,
    )
    check = _converter_process(subprocess.run, [str(python), "-I", "-c", validation], capture_output=True, text=True, **NO_WINDOW)
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


def run_setup(action="all", models_dir=None, bootstrap_converter=True, cancel=None, progress=None, backend=None):
    if cancel and cancel.is_set():
        raise Cancelled("Operation canceled.")
    if action not in SETUP_ACTIONS:
        raise UWCCError(f"Unknown model setup action: {action}")
    if backend is not None and backend not in ("cpu", "cuda", "directml"):
        raise UWCCError(f"Unknown execution backend: {backend}")
    if backend is not None and action in ("prepare-gpu", "validate-gpu") and backend != "cuda":
        raise UWCCError("Optional CUDA graphs require NVIDIA CUDA. CPU and DirectML use the standard models.")
    directory = (Path(models_dir) if models_dir else model_directory()).expanduser()
    if action == "runtime":
        if backend is None:
            raise UWCCError("Choose an execution backend to install its runtime.")
        steps = []
    elif action == "all":
        steps = [("prepare", False), ("prepare-orientation", False)]
    elif action == "download-only":
        steps = [("prepare", True)]
    else:
        steps = [(action, False)]
    tools = tool_directory()
    outputs = []
    selected_python = None
    if backend is not None:
        selected_python = ensure_converter_python(
            bootstrap=bootstrap_converter, cancel=cancel,
            progress=(lambda f, msg: progress(f * 0.4, msg)) if progress else None,
            backend=backend,
        )
    script_env = None
    if backend:
        script_env = os.environ.copy()
        script_env["UWCC_INFERENCE_DEVICE"] = backend
    for index, (command, download_only) in enumerate(steps):
        if cancel and cancel.is_set():
            raise Cancelled("Operation canceled.")
        script = tools / SCRIPTS[command]
        if not script.is_file():
            raise UWCCError(f"Model setup script is missing: {script}")
        base_fraction = (0.4 + 0.6 * index / len(steps)) if selected_python else index / len(steps)
        needs_converter = command in ("prepare", "prepare-gpu") and not download_only
        if selected_python:
            cmd = [selected_python, str(script), "--models-dir", str(directory)]
        elif needs_converter:
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
        text = _run_cancellable(cmd, cancel=cancel, env=script_env)
        if text:
            outputs.append(text)
    if selected_python:
        if cancel and cancel.is_set():
            raise Cancelled("Operation canceled.")
        from .execution import save_runtime
        site_packages = _run_cancellable(
            [selected_python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], cancel=cancel,
        ).strip()
        if not Path(site_packages).is_dir():
            raise UWCCError("Selected runtime site-packages directory was not created.")
        save_runtime(backend, site_packages)
        outputs.append("Execution runtime saved. Restart UWCC to use it.")
    if progress:
        progress(1.0, "Model preparation complete.")
    return "\n".join(outputs) if outputs else f"Model setup ({action}) complete in {directory}."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=list(SCRIPTS) + ["runtime"])
    parser.add_argument("--backend", choices=["cpu", "cuda", "directml"], help="Install and select an execution runtime")
    parser.add_argument("--models-dir", type=Path, default=model_directory())
    parser.add_argument("--download-only", action="store_true", help="Download without converting (prepare only)")
    args = parser.parse_args(argv)
    if args.backend is None:
        from .execution import saved_runtime
        saved = saved_runtime()
        if saved:
            args.backend = saved["backend"]
    if args.backend is not None or args.command == "runtime":
        if args.backend is None:
            parser.error("runtime requires --backend")
        try:
            action = "download-only" if args.download_only and args.command == "prepare" else args.command
            if args.download_only and args.command != "prepare":
                parser.error("--download-only is only available with prepare")
            print(run_setup(action, args.models_dir, backend=args.backend))
            return 0
        except UWCCError as exc:
            parser.exit(1, f"{exc}\n")
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
