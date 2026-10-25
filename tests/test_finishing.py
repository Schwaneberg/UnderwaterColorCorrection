import subprocess
import threading

import numpy as np
import pytest

from uwcc.errors import Cancelled, UWCCError
from uwcc.media import FrameReader, frame_timing, probe
from uwcc.project import Grade, Project, fingerprint, load_preset, save_preset
from uwcc.render import render


def write_clip(path, frames):
    height, width = frames.shape[1:3]
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "12", "-i", "pipe:0",
        "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(path),
    ], input=frames.tobytes(), check=True)


def read_all(info, grade):
    with FrameReader(info, finishing=grade) as reader:
        return np.stack(list(reader))


def test_stabilization_reduces_shake(tmp_path):
    rng = np.random.default_rng(42)
    base = rng.integers(10, 245, (45, 80, 1), dtype=np.uint8)
    base = np.repeat(np.repeat(np.repeat(base, 4, axis=0), 4, axis=1), 3, axis=2)
    frames = np.stack([np.roll(base, 5 if i % 2 else -5, axis=1) for i in range(24)])
    path = tmp_path / "shake.mkv"
    write_clip(path, frames)
    info = probe(path)
    before = read_all(info, Grade())[:, 24:-24, 24:-24]
    after = read_all(info, Grade(stabilize=True))[:, 24:-24, 24:-24]
    assert before.shape == after.shape
    # The scene is stationary; inter-frame differences measure residual shake.
    assert np.mean(abs(np.diff(after, axis=0))) < .7 * np.mean(abs(np.diff(before, axis=0)))


def test_denoise_reduces_noise(tmp_path):
    rng = np.random.default_rng(8)
    frames = np.clip(rng.normal(128, 5, (18, 120, 160, 3)), 0, 255).astype(np.uint8)
    path = tmp_path / "noise.mkv"
    write_clip(path, frames)
    info = probe(path)
    before = read_all(info, Grade())
    after = read_all(info, Grade(denoise=6))
    assert after.shape == before.shape
    assert np.std(after[3:]) < .8 * np.std(before[3:])
    assert abs(after.mean() - before.mean()) < .01


def test_finishing_export_keeps_frames_and_audio(clip, tmp_path):
    grade = Grade(input_profile="rec709", stabilize=True, denoise=3)
    project = Project(str(clip), grade, fingerprint(clip))
    target = tmp_path / "finished.mp4"
    render(project, target)
    info = probe(target)
    assert frame_timing(info)[0] == 18
    assert info.audio_count == 1
    assert (info.width, info.height) == (320, 180)


def test_finishing_cancel_preserves_existing_output(clip, tmp_path):
    project = Project(str(clip), Grade(input_profile="rec709", stabilize=True, denoise=3), fingerprint(clip))
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"existing output")
    cancel = threading.Event()
    def progress(fraction, message):
        if message.startswith("Rendering frame"):
            cancel.set()
    with pytest.raises(Cancelled):
        render(project, target, cancel=cancel, progress=progress, overwrite=True)
    assert target.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".uwcc-render-*"))


def test_finishing_persistence(clip, tmp_path):
    grade = Grade(input_profile="rec709", stabilize=True, denoise=3.5)
    project = Project(str(clip), grade, fingerprint(clip))
    project.save(tmp_path / "project.json")
    restored = Project.load(tmp_path / "project.json").grade
    assert restored.stabilize and restored.denoise == 3.5
    save_preset(tmp_path / "preset.json", grade)
    restored = load_preset(tmp_path / "preset.json")
    assert restored.stabilize and restored.denoise == 3.5
    assert Grade.from_dict({}).denoise == 0
    assert not Grade.from_dict({}).stabilize


@pytest.mark.parametrize("data", [{"denoise": -1}, {"denoise": 11}, {"denoise": float("nan")},
                                  {"denoise": True}, {"stabilize": "yes"}])
def test_invalid_finishing_settings(data):
    with pytest.raises(UWCCError):
        Grade.from_dict(data)
