"""Execution selection, installation isolation, and provider routing."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from uwcc import execution, model_setup
from uwcc.errors import UWCCError


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "settings_path", lambda: tmp_path / "execution.json")
    monkeypatch.delenv("UWCC_INFERENCE_DEVICE", raising=False)
    return tmp_path


def test_runtime_selection_survives_restart_and_environment_override(settings, monkeypatch):
    packages = settings / "site-packages"
    packages.mkdir()
    execution.save_runtime("directml", packages)
    assert execution.selected_backend() == "directml"
    assert execution.inference_device() == "directml"
    monkeypatch.setattr(sys, "path", sys.path[:])
    execution.activate_runtime()
    assert sys.path[0] == str(packages)
    execution.activate_runtime()
    assert sys.path.count(str(packages)) == 1
    monkeypatch.setenv("UWCC_INFERENCE_DEVICE", "cpu")
    assert execution.inference_device() == "cpu"


@pytest.mark.parametrize("backend,ort,index", [
    ("cpu", "onnxruntime", "cpu"),
    ("cuda", "onnxruntime-gpu[cuda,cudnn]>=1.21", "cu130"),
    pytest.param("directml", "onnxruntime-directml>=1.17", "cpu",
                 marks=pytest.mark.skipif(sys.platform != "win32", reason="Windows runtime")),
])
def test_private_runtime_installs_matching_packages(settings, monkeypatch, backend, ort, index):
    root = settings / "converter"
    monkeypatch.setattr(model_setup, "converter_environment_path", lambda: root)
    monkeypatch.setattr(model_setup, "_has_converter_modules", lambda: True)
    created = []

    def create(destination, *args):
        created.append(destination)
        python = model_setup._venv_python(destination)
        python.parent.mkdir(parents=True)
        python.touch()
        return python

    monkeypatch.setattr(model_setup, "_create_converter_runtime", create)
    checks = []
    monkeypatch.setattr(model_setup.subprocess, "run",
                        lambda args, **kw: checks.append(args) or SimpleNamespace(returncode=0, stderr=""))
    installs = []
    monkeypatch.setattr(model_setup, "_run_cancellable", lambda args, **kw: installs.append(args) or "")
    python = model_setup.ensure_converter_python(backend=backend)
    assert created == [root.with_name(f"converter-{backend}")]
    assert python != sys.executable
    assert any("torch" in args and f"https://download.pytorch.org/whl/{index}" in args for args in installs)
    assert any(ort in args and "onnxscript" in args for args in installs)
    cuda_check = "torch.version.cuda is not None" if backend == "cuda" else "torch.version.cuda is None"
    assert cuda_check in checks[-1][-1]


def test_runtime_only_setup_saves_after_success(settings, monkeypatch):
    packages = settings / "site-packages"
    packages.mkdir()
    calls = []
    monkeypatch.setattr(model_setup, "ensure_converter_python",
                        lambda **kw: calls.append(kw["backend"]) or "private-python")
    monkeypatch.setattr(model_setup, "_run_cancellable", lambda *args, **kw: str(packages))
    output = model_setup.run_setup("runtime", backend="cuda")
    assert calls == ["cuda"]
    assert execution.saved_runtime()["backend"] == "cuda"
    assert "Restart UWCC" in output

    def fail(**kw):
        raise UWCCError("Installation canceled")

    monkeypatch.setattr(model_setup, "ensure_converter_python", fail)
    with pytest.raises(UWCCError):
        model_setup.run_setup("runtime", backend="cpu")
    assert execution.saved_runtime()["backend"] == "cuda"


def test_cli_uses_saved_runtime_for_model_validation(settings, monkeypatch):
    packages = settings / "site-packages"
    packages.mkdir()
    execution.save_runtime("cuda", packages)
    calls = []
    monkeypatch.setattr(model_setup, "run_setup", lambda *args, **kw: calls.append((args, kw)) or "ok")
    assert model_setup.main(["validate-gpu", "--models-dir", str(settings)]) == 0
    assert calls == [(("validate-gpu", settings), {"backend": "cuda"})]


@pytest.mark.parametrize("backend", ["cpu", "directml"])
def test_cuda_graph_actions_require_cuda(settings, backend):
    with pytest.raises(UWCCError, match="require NVIDIA CUDA"):
        model_setup.run_setup("validate-gpu", backend=backend)


@pytest.mark.parametrize("device,expected", [
    ("cpu", ["CPUExecutionProvider"]),
    ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    ("directml", ["DmlExecutionProvider", "CPUExecutionProvider"]),
])
def test_explicit_device_excludes_other_gpu_vendor(settings, monkeypatch, device, expected):
    from uwcc.neural import Network
    monkeypatch.setenv("UWCC_INFERENCE_DEVICE", device)
    calls = []
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        get_available_providers=lambda: ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=lambda *a, providers, **kw: calls.append(providers) or SimpleNamespace(
            get_providers=lambda: providers, get_inputs=lambda: []),
    ))
    Network({}, "unused.onnx")
    assert calls == [expected]
