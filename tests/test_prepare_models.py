"""Upstream download checks use in-memory responses, never the network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import shutil
import subprocess
import textwrap

import pytest


@pytest.fixture
def prepare():
    path = Path(__file__).resolve().parents[1]/"tools/prepare_models.py"
    spec = importlib.util.spec_from_file_location("prepare_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_verifies_and_reuses_cached_weights(prepare, tmp_path, monkeypatch):
    payload = b"upstream weights"
    checksum = hashlib.sha256(payload).hexdigest()
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(prepare, "urlopen", fetch)
    destination = tmp_path/"cache/weights.pth"
    prepare.download("https://upstream.example/weights", destination, checksum)
    assert destination.read_bytes() == payload
    prepare.download("https://upstream.example/weights", destination, checksum)
    assert len(calls) == 1


def test_bad_download_does_not_overwrite_cached_file(prepare, tmp_path, monkeypatch):
    destination = tmp_path/"weights.pth"
    destination.write_bytes(b"old cache")
    monkeypatch.setattr(prepare, "urlopen", lambda *a, **k: io.BytesIO(b"bad weights"))
    with pytest.raises(RuntimeError, match="checksum"):
        prepare.download("https://upstream.example/weights", destination, "0"*64)
    assert destination.read_bytes() == b"old cache"


def test_download_only_uses_pinned_sources_without_converter(prepare, tmp_path, monkeypatch):
    calls = []

    def download(url, path, checksum=None):
        calls.append((url, checksum))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")

    monkeypatch.setattr(prepare, "download", download)
    # Download-only setup must work before NumPy/PyTorch/ONNX are installed.
    monkeypatch.setitem(sys.modules, "convert_models", None)
    prepare.main(["--download-only", "--models-dir", str(tmp_path)])
    for name, metadata in prepare.SOURCES.items():
        revision = json.loads((tmp_path/".sources"/name/"revision.json").read_text())
        assert revision == {k: metadata[k] for k in ("repo", "sha")}
        for file in metadata["files"]:
            assert (f"https://raw.githubusercontent.com/{metadata['repo']}/{metadata['sha']}/{file}", None) in calls
    for filename, url, checksum in prepare.WEIGHTS.values():
        assert (url, checksum) in calls
        assert (tmp_path/".sources"/filename).is_file()
    assert not list(tmp_path.rglob("*.onnx"))


def test_full_preparation_converts_after_downloads(prepare, tmp_path, monkeypatch):
    events = []

    def download(url, path, checksum=None):
        events.append("download")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")

    def export(kind, source, output):
        assert all((source/file).is_file() for file, _, _ in prepare.WEIGHTS.values())
        assert output == tmp_path/kind
        events.append(kind)

    from types import SimpleNamespace
    monkeypatch.setattr(prepare, "download", download)
    monkeypatch.setitem(sys.modules, "convert_models", SimpleNamespace(export=export))
    prepare.main(["--models-dir", str(tmp_path)])
    assert events[-2:] == list(prepare.WEIGHTS)


@pytest.mark.parametrize("script", ["prepare_models.py", "prepare_gpu_denoise.py"])
def test_preparation_finds_sibling_converter_in_isolated_python(tmp_path, script):
    tools = tmp_path / "installed app/tools"
    tools.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "tools" / script
    shutil.copy2(source, tools / script)
    # Keep this regression independent of downloads and conversion dependencies.
    (tools / "convert_models.py").write_text(
        "def export(kind, source, output):\n"
        "    output.mkdir(parents=True)\n"
        "    (output / 'converted').write_text(kind)\n"
        "def load_module(name, path):\n"
        "    pass\n",
        encoding="utf-8",
    )
    models = tmp_path / "user models"
    code = textwrap.dedent("""
        import runpy, sys, types
        from pathlib import Path
        script, models = sys.argv[1:]
        assert str(Path(script).parent) not in sys.path
        sys.modules['numpy'] = types.ModuleType('numpy')
        scope = runpy.run_path(script)
        if Path(script).name == 'prepare_models.py':
            def download(url, path, checksum=None):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'cached weights')
            scope['main'].__globals__['download'] = download
            scope['main'](['--models-dir', models])
        else:
            assert callable(scope['load_module'])
    """)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tools / script), str(models)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if script == "prepare_models.py":
        for kind in ("udnet", "fastdvdnet"):
            assert (models / kind / "converted").read_text() == kind
