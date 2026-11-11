"""Runtime FFmpeg discovery must bypass launchers that outlive cancellation."""
from pathlib import Path
import shutil
import threading

import pytest

from uwcc import media
from uwcc.errors import Cancelled
from uwcc.project import Grade, Project
from uwcc.render import render


@pytest.mark.parametrize("manager", ["chocolatey", "scoop"])
@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("explicit", [False, True])
def test_runtime_resolves_package_manager_launchers(tmp_path, monkeypatch, manager, name, explicit):
    root = tmp_path / manager
    shim = root / ("bin" if manager == "chocolatey" else "shims") / f"{name}.exe"
    real = root / ("lib/ffmpeg/tools/ffmpeg/bin" if manager == "chocolatey"
                   else "apps/ffmpeg/current/bin") / shim.name
    for path, data in ((shim, b"launcher"), (real, b"real executable" * 10)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    unrelated = root / ("lib/another-package/tools" if manager == "chocolatey"
                        else "apps/another-package/current") / shim.name
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"unrelated executable" * 10)
    monkeypatch.setattr(media, "bundled_ffmpeg_dirs", lambda: [shim.parent] if explicit else [])
    monkeypatch.setattr(media.shutil, "which", lambda *a, **k: str(shim))
    assert Path(media.executable(name)) == real.resolve()


def test_cancel_with_chocolatey_ffmpeg_preserves_output(clip, tmp_path, monkeypatch):
    binary = media.executable("ffmpeg")
    shim = tmp_path / "chocolatey/bin/ffmpeg.exe"
    real = tmp_path / "chocolatey/lib/ffmpeg/tools/ffmpeg/bin/ffmpeg.exe"
    shim.parent.mkdir(parents=True)
    real.parent.mkdir(parents=True)
    shim.write_bytes(b"This launcher must never be executed")
    shutil.copy2(binary, real)
    which = shutil.which
    monkeypatch.setattr(media, "bundled_ffmpeg_dirs", lambda: [])
    monkeypatch.setattr(media.shutil, "which", lambda name, **kw:
                        str(shim) if name == "ffmpeg" else which(name, **kw))
    cancel = threading.Event()

    def progress(fraction, message):
        if message.startswith("Rendering frame"):
            cancel.set()

    target = tmp_path / "existing.mp4"
    target.write_bytes(b"existing output")
    project = Project(str(clip), Grade(input_profile="rec709"))
    with pytest.raises(Cancelled):
        render(project, target, cancel=cancel, progress=progress, overwrite=True)
    assert target.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".uwcc-render-*"))
