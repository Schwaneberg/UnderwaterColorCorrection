"""Tests for bundled FFmpeg discovery and Windows MSIX packaging helpers."""
import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from uwcc import media


@pytest.fixture
def build_windows():
    script = Path(__file__).resolve().parents[1] / "tools/build_windows.py"
    spec = importlib.util.spec_from_file_location("build_windows", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_executable_prefers_bundled_ffmpeg_directory(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bundled_bin"
    bin_dir.mkdir()
    suffix = ".exe" if os.name == "nt" else ""
    ffmpeg_bin = bin_dir / f"ffmpeg{suffix}"
    ffprobe_bin = bin_dir / f"ffprobe{suffix}"
    ffmpeg_bin.write_bytes(b"fake-ffmpeg")
    ffprobe_bin.write_bytes(b"fake-ffprobe")
    if os.name != "nt":
        ffmpeg_bin.chmod(0o755)
        ffprobe_bin.chmod(0o755)
    monkeypatch.setenv("UWCC_FFMPEG_DIR", str(bin_dir))
    assert Path(media.executable("ffmpeg")).resolve() == ffmpeg_bin.resolve()
    assert Path(media.executable("ffprobe")).resolve() == ffprobe_bin.resolve()


def test_windows_source_snapshot_excludes_binaries_and_models(build_windows, tmp_path, monkeypatch):
    names = [
        "pyproject.toml",
        "uwcc/gui.py",
        "models/udnet/manifest.json",
        "models/udnet/model.onnx",
        "models/.sources/UDnet.py",
        "weights.pth",
        "uwcc/_native.pyd",
        "dist/windows/uwcc.msix",
        ".venv-win/pyvenv.cfg",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(build_windows.subprocess, "check_output", lambda *a, **k: "\0".join(names).encode())
    assert set(build_windows.source_files(tmp_path)) == {
        Path("pyproject.toml"),
        Path("uwcc/gui.py"),
        Path("models/udnet/manifest.json"),
    }


def test_msix_version_normalization(build_windows):
    assert build_windows.msix_version("0.2.0") == "0.2.0.0"
    assert build_windows.msix_version("1.4") == "1.4.0.0"
    assert build_windows.msix_version("2.3.4.5") == "2.3.4.5"


def test_bundle_ffmpeg_copies_binaries_and_companion_files(build_windows, tmp_path):
    src_dir = tmp_path / "ffmpeg_src"
    src_dir.mkdir()
    suffix = ".exe" if os.name == "nt" else ""
    for name in (f"ffmpeg{suffix}", f"ffprobe{suffix}", "avcodec-61.dll", "LICENSE.txt"):
        (src_dir / name).write_bytes(b"content")
        if os.name != "nt" and not name.endswith((".dll", ".txt")):
            (src_dir / name).chmod(0o755)
    dest_bin = tmp_path / "bundle/bin"
    ffmpeg_out, ffprobe_out = build_windows.bundle_ffmpeg(dest_bin, ffmpeg_dir=src_dir)
    assert ffmpeg_out == dest_bin / "ffmpeg.exe"
    assert ffprobe_out == dest_bin / "ffprobe.exe"
    assert (dest_bin / "avcodec-61.dll").is_file()
    assert (dest_bin / "LICENSE.txt").is_file()


def test_msix_layout_generates_valid_manifest_and_assets(build_windows, tmp_path):
    from PIL import Image
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "bin").mkdir(parents=True)
    (bundle_dir / "uwcc.exe").write_bytes(b"exe")
    (bundle_dir / "bin/ffmpeg.exe").write_bytes(b"ffmpeg")
    (bundle_dir / "bin/ffprobe.exe").write_bytes(b"ffprobe")

    icon_path = tmp_path / "test_icon.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(icon_path)

    layout_dir = tmp_path / "msix-layout"
    build_windows.stage_msix_layout(
        bundle_dir,
        layout_dir,
        icon_path,
        identity_name="OliverSchwaneberg.UnderwaterColorCorrection",
        publisher="CN=12345678-ABCD",
        version="0.2.0",
        display_name="Underwater Color Correction",
        publisher_display_name="Oliver Schwaneberg",
        architecture="x64",
    )
    manifest_path = layout_dir / "AppxManifest.xml"
    assert manifest_path.is_file()
    tree = ET.parse(manifest_path)
    ns = {
        "m": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
        "rescap": "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
    }
    identity = tree.find("m:Identity", ns)
    assert identity is not None
    assert identity.attrib["Name"] == "OliverSchwaneberg.UnderwaterColorCorrection"
    assert identity.attrib["Publisher"] == "CN=12345678-ABCD"
    assert identity.attrib["Version"] == "0.2.0.0"
    capability = tree.find("m:Capabilities/rescap:Capability", ns)
    assert capability is not None and capability.attrib["Name"] == "runFullTrust"
    for asset in ("StoreLogo.png", "Square44x44Logo.png", "Square150x150Logo.png", "Wide310x150Logo.png", "uwcc.ico"):
        assert (layout_dir / "Assets" / asset).is_file()


def test_chocolatey_and_scoop_shims_resolve_to_real_binaries(build_windows, tmp_path):
    choco_shim = tmp_path / "chocolatey/bin/ffmpeg.exe"
    choco_real = tmp_path / "chocolatey/lib/ffmpeg/tools/ffmpeg/bin/ffmpeg.exe"
    scoop_shim = tmp_path / "scoop/shims/ffprobe.exe"
    scoop_real = tmp_path / "scoop/apps/ffmpeg/current/bin/ffprobe.exe"
    for path, size in ((choco_shim, 10), (choco_real, 1000), (scoop_shim, 10), (scoop_real, 1000)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    assert build_windows.resolve_shim(choco_shim) == choco_real.resolve()
    assert build_windows.resolve_shim(scoop_shim) == scoop_real.resolve()
    plain = tmp_path / "ffmpeg/bin/ffmpeg.exe"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(b"real")
    assert build_windows.resolve_shim(plain) == plain


def test_verify_runtime_dependencies_requires_cv2_and_onnxruntime_by_default(build_windows, monkeypatch):
    monkeypatch.setattr(build_windows.importlib.util, "find_spec", lambda name: object() if name == "cv2" else None)
    with pytest.raises(RuntimeError, match="ONNX Runtime"):
        build_windows.verify_runtime_dependencies()
    assert build_windows.verify_runtime_dependencies(allow_missing_ml=True) == ["cv2"]
    monkeypatch.setattr(build_windows.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="OpenCV"):
        build_windows.verify_runtime_dependencies(allow_missing_ml=True)
    assert build_windows.verify_runtime_dependencies(allow_missing_vision=True, allow_missing_ml=True) == []

