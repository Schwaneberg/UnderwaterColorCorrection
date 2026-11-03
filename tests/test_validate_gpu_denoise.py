"""CUDA prerequisites must fail clearly before model validation changes files."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from uwcc.errors import UWCCError


@pytest.fixture
def validator():
    path = Path(__file__).resolve().parents[1] / "tools/validate_gpu_denoise.py"
    spec = importlib.util.spec_from_file_location("validate_gpu_denoise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_runtime_reports_install_commands(validator, monkeypatch, tmp_path):
    monkeypatch.delenv("UWCC_INFERENCE_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
    ))
    with pytest.raises(UWCCError, match="no CUDA execution provider") as error:
        validator.validate(tmp_path)
    assert sys.executable in str(error.value)
    assert 'pip uninstall -y onnxruntime onnxruntime-directml' in str(error.value)
    assert 'onnxruntime-gpu[cuda,cudnn]' in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_cpu_override_explains_how_to_enable_gpu(validator, monkeypatch, tmp_path):
    monkeypatch.setenv("UWCC_INFERENCE_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ))
    with pytest.raises(UWCCError, match="UWCC_INFERENCE_DEVICE=auto"):
        validator.validate(tmp_path)
