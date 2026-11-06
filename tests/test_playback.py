from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from uwcc import gui
from uwcc.project import Grade, Project


@pytest.mark.parametrize("profile", ["rec709", "slog3", "hlg", "pq"])
def test_parallel_preview_matches_full_frame(profile):
    raw = np.random.default_rng(8).random((80, 96, 3), dtype=np.float32)
    grade = Grade(input_profile=profile)
    grade.values.update(temperature=15, saturation=110, exposure=.3)
    grade.set_key("contrast", 0, 5)
    grade.set_key("contrast", 2, 20)
    expected = gui.correct(raw, grade, 1, preview=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = gui.preview_correct(raw, grade, 1, None, executor, 4)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("control", ["dehaze", "sharpness"])
def test_spatial_preview_uses_full_frame(control):
    raw = np.random.default_rng(8).random((80, 96, 3), dtype=np.float32)
    grade = Grade(input_profile="rec709")
    grade.values[control] = 10
    executor = SimpleNamespace(map=lambda *args: pytest.fail("Spatial filter split into stripes"))
    result = gui.preview_correct(raw, grade, 1, None, executor, 4)
    np.testing.assert_array_equal(result, gui.correct(raw, grade, 1, preview=True))


def test_rec709_original_avoids_color_roundtrip(monkeypatch):
    raw = np.random.default_rng(8).random((12, 16, 3), dtype=np.float32)
    expected = gui.encode(gui.decode(raw, "rec709"), "rec709")
    monkeypatch.setattr(gui, "decode", lambda *args: pytest.fail("Redundant decode"))
    assert gui.preview_original(raw, "rec709") is raw
    np.testing.assert_allclose(raw, expected, atol=.001)


@pytest.mark.parametrize("profile", ["slog2", "slog3", "hlg", "pq"])
def test_log_hdr_original_preserves_display_transform(profile):
    raw = np.random.default_rng(8).random((12, 16, 3), dtype=np.float32)
    expected = gui.encode(gui.decode(raw, profile), "rec709", True)
    np.testing.assert_array_equal(gui.preview_original(raw, profile), expected)


@pytest.mark.parametrize("correction_seconds", [.02, .25])
def test_playback_skips_expensive_correction_only_for_late_frames(monkeypatch, correction_seconds):
    raw = np.zeros((2, 2, 3), dtype=np.float32)
    clock = [0.0]
    corrected = []
    shown = []

    class Reader:
        def __init__(self, *args, **kwargs):
            assert kwargs["thread_limit"] == 4

        def __enter__(self):
            return iter([raw] * 10)

        def __exit__(self, *args):
            pass

    def correct(frame, grade, timestamp, *args, **kwargs):
        corrected.append(timestamp)
        clock[0] += correction_seconds
        return frame

    monkeypatch.setattr(gui, "FrameReader", Reader)
    monkeypatch.setattr(gui, "frame_timestamps", lambda *args, **kwargs: (i/10 for i in range(10)))
    monkeypatch.setattr(gui, "prepare_frame", lambda frame, *args: frame)
    monkeypatch.setattr(gui, "correct", correct)
    monkeypatch.setattr(gui, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    worker = gui.Playback(SimpleNamespace(fps=10), Project("test", Grade(input_profile="rec709")), 0)
    class Cancel:
        def is_set(self):
            return False

        def wait(self, delay):
            clock[0] += delay
            return False

    worker.cancel = Cancel()
    worker.frame.connect(lambda original, rgb, timestamp: shown.append(timestamp))
    worker.run()
    assert shown == corrected
    assert shown[0] == 0
    assert shown[-1] >= .8
    if correction_seconds > .1:
        assert len(shown) < 10
    else:
        assert shown == [i/10 for i in range(10)]
