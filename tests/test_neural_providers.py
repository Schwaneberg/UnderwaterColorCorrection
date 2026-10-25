"""Exercise provider initialization contracts without requiring GPU hardware."""
import sys
from types import SimpleNamespace

import pytest

from uwcc.neural import Network


@pytest.mark.parametrize("provider", ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])
@pytest.mark.parametrize("force_cpu", [False, True])
def test_provider_session_options_and_cpu_override(monkeypatch, provider, force_cpu):
    monkeypatch.setenv("UWCC_INFERENCE_DEVICE", "cpu" if force_cpu else "auto")
    calls = []

    def session(path, options, providers, provider_options=None):
        calls.append(providers)
        if "DmlExecutionProvider" in providers:
            # DirectML rejects ORT's default memory-pattern optimization.
            assert options.enable_mem_pattern is False
            assert options.execution_mode == "sequential"
        else:
            assert options.enable_mem_pattern is True
        return SimpleNamespace(get_providers=lambda: providers, get_inputs=lambda: [])

    available = list(dict.fromkeys([provider, "CPUExecutionProvider"]))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(enable_mem_pattern=True),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        get_available_providers=lambda: available,
        InferenceSession=session,
    ))
    # OpenCV-backed model manifests should also select an available GPU runtime.
    data = {"runtime": "opencv"} if provider != "CPUExecutionProvider" and not force_cpu else {}
    network = Network(data, "model.onnx")
    expected = ["CPUExecutionProvider"] if force_cpu else available
    assert calls == [expected]
    assert network.uses_cuda == (provider == "CUDAExecutionProvider" and not force_cpu)
    expected_label = "CPU" if force_cpu else {
        "CUDAExecutionProvider": "CUDA GPU",
        "DmlExecutionProvider": "DirectML GPU",
        "CPUExecutionProvider": "CPU",
    }[provider]
    assert network.execution_label == f"{expected_label} · ONNX Runtime"
