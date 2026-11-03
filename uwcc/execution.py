"""Persist the explicitly selected local AI runtime and activate it at startup."""
import json
import os
from pathlib import Path
import sys

from .project import atomic_json, user_directory

BACKENDS = {"cpu": "CPU", "cuda": "NVIDIA CUDA", "directml": "AMD / DirectML (Windows)"}


def settings_path():
    return user_directory(create=False) / "ai-execution.json"


def saved_runtime():
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        abi = f"{sys.version_info.major}.{sys.version_info.minor}"
        if (data.get("backend") in BACKENDS and data.get("python_abi") == abi
                and Path(data["site_packages"]).is_dir()):
            return data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def selected_backend():
    saved = saved_runtime()
    if saved:
        return saved["backend"]
    if os.environ.get("UWCC_INFERENCE_DEVICE") in BACKENDS:
        return os.environ["UWCC_INFERENCE_DEVICE"]
    # Preserve an existing installation's effective selection before setup.
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda"
        if "DmlExecutionProvider" in providers:
            return "directml"
    except ImportError:
        pass
    return "cpu"


def save_runtime(backend, site_packages):
    atomic_json(settings_path(), {"backend": backend, "site_packages": str(site_packages),
                                 "python_abi": f"{sys.version_info.major}.{sys.version_info.minor}"})


def activate_runtime():
    """Call before importing inference libraries; native runtimes need a restart."""
    saved = saved_runtime()
    if saved:
        path = saved["site_packages"]
        if path not in sys.path:
            sys.path.insert(0, path)


def inference_device():
    # An explicit environment override remains available for headless workflows.
    override = os.environ.get("UWCC_INFERENCE_DEVICE")
    if override:
        return override
    saved = saved_runtime()
    return saved["backend"] if saved else "auto"
