"""Ensure packaging snapshots exclude local models and development artifacts."""
import importlib.util
from pathlib import Path


def test_source_snapshot_excludes_model_binaries_even_if_tracked(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/"tools/build_deb.py"
    spec = importlib.util.spec_from_file_location("build_deb", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ["pyproject.toml", "uwcc/gui.py", "models/udnet/manifest.json",
             "models/udnet/model.onnx", "models/.sources/UDnet.py", "weights.pth",
             "uwcc/_native.so", ".aws/credentials", ".venv-models/pyvenv.cfg"]
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "\0".join(names).encode())
    assert set(module.source_files(tmp_path)) == {
        Path("pyproject.toml"), Path("uwcc/gui.py"), Path("models/udnet/manifest.json")}
