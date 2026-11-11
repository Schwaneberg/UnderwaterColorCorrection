import threading
from types import SimpleNamespace

import numpy as np
import pytest

from uwcc.errors import Cancelled, UWCCError
from uwcc.project import Grade, Project
from uwcc.render import color_workers, corrected_frames


def test_cancel_waits_for_watcher_before_encoder_cleanup(clip, tmp_path, monkeypatch):
    import uwcc.render as module
    original = module.frame_process
    killing = threading.Event()
    release = threading.Event()

    class Encoder:
        def __init__(self, *args, **kwargs):
            self.process = original(*args, **kwargs)
            self.stdin = self.process.stdin

        def poll(self):
            assert not killing.is_set(), "Cleanup raced the cancellation watcher"
            return self.process.poll()

        def kill(self):
            killing.set()
            self.process.kill()
            assert release.wait(3)

        def wait(self):
            code = self.process.wait()
            killing.clear()
            return code

    monkeypatch.setattr(module, "frame_process", Encoder)
    cancel = threading.Event()

    def progress(fraction, message):
        if message.startswith("Rendering frame"):
            cancel.set()
            assert killing.wait(3)
            # Keep the watcher inside kill while the render loop unwinds.
            timer = threading.Timer(.2, release.set)
            timer.start()

    target = tmp_path / "existing.mp4"
    target.write_bytes(b"existing output")
    project = Project(str(clip), Grade(input_profile="rec709"))
    with pytest.raises(Cancelled):
        module.render(project, target, cancel=cancel, progress=progress, overwrite=True)
    assert target.read_bytes() == b"existing output"
    assert not list(tmp_path.glob(".uwcc-render-*"))


def test_parallel_color_is_bounded_ordered_and_concurrent(monkeypatch):
    import uwcc.render as module
    barrier = threading.Barrier(2)
    threads = set()
    def fake(frame, grade, seconds, analysis):
        if seconds < 2:
            threads.add(threading.get_ident())
            barrier.wait(timeout=3)
        return np.rint(frame*65535).astype("<u2")
    monkeypatch.setattr(module, "correct_rgb48", fake)
    read = []
    def frames():
        for i in range(12):
            read.append(i)
            yield np.full((2, 2, 3), i/20, np.float32)
    project = Project("unused", Grade(input_profile="rec709"))
    iterator = corrected_frames(frames(), SimpleNamespace(color_range="pc"), project,
                                list(range(12)), threading.Event(), 2)
    first = next(iterator)
    assert 2 <= len(read) <= 4  # at most two pending frames per worker
    results = [first, *iterator]
    assert [index for index, _ in results] == list(range(12))
    assert len(threads) == 2
    for index, pixels in results:
        assert np.all(pixels == np.rint(np.float32(index/20)*65535))


def test_worker_count_obeys_memory_budget(monkeypatch):
    monkeypatch.setenv("UWCC_THREADS", "8")
    monkeypatch.setenv("UWCC_RENDER_MEMORY_MB", "768")
    assert color_workers(SimpleNamespace(width=7680, height=4320), 8) == 1
    with pytest.raises(UWCCError):
        color_workers(SimpleNamespace(width=1920, height=1080), 0)


def test_parallel_color_cancellation_stops_before_reading():
    cancel = threading.Event()
    cancel.set()
    project = Project("unused", Grade(input_profile="rec709"))
    with pytest.raises(Cancelled):
        list(corrected_frames([np.zeros((2, 2, 3), np.float32)], SimpleNamespace(color_range="pc"),
                              project, [0], cancel, 2))


def test_ready_color_result_is_not_held_until_queue_fills(monkeypatch):
    from concurrent.futures import Future
    import uwcc.render as module
    class ImmediateExecutor:
        def __init__(self, **_):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def submit(self, fn, *args):
            result = Future()
            result.set_result(fn(*args))
            return result
    monkeypatch.setattr(module, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(module, "correct_rgb48", lambda frame, *_: np.rint(frame*65535).astype("<u2"))
    read = []
    def frames():
        for i in range(10):
            read.append(i)
            yield np.full((2, 2, 3), .2, np.float32)
    project = Project("unused", Grade(input_profile="rec709"))
    iterator = corrected_frames(frames(), SimpleNamespace(color_range="pc"), project,
                                list(range(10)), threading.Event(), 3)
    assert next(iterator)[0] == 0
    assert read == [0]
    iterator.close()
