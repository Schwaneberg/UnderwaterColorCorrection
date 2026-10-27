import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


def test_orientation_download_verifies_cache_and_preserves_previous_file(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/"tools/prepare_orientation_model.py"
    spec = importlib.util.spec_from_file_location("prepare_orientation_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path/"model.onnx"
    payload = b"verified upstream model"
    checksum = hashlib.sha256(payload).hexdigest()
    calls = []
    def fetch(*args, **kwargs):
        calls.append(args[0])
        return io.BytesIO(payload)
    monkeypatch.setattr(module, "urlopen", fetch)
    module.download("model.onnx", destination, checksum)
    module.download("model.onnx", destination, checksum)
    assert len(calls) == 1 and destination.read_bytes() == payload
    with pytest.raises(RuntimeError, match="Checksum"):
        module.download("model.onnx", destination, "0"*64)
    assert destination.read_bytes() == payload
    assert not destination.with_name("model.onnx.partial").exists()
