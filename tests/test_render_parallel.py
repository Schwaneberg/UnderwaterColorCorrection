import threading
from types import SimpleNamespace

import numpy as np
import pytest

from uwcc.errors import Cancelled, UWCCError
from uwcc.project import Grade, Project
from uwcc.render import color_workers, corrected_frames


def test_parallel_color_is_bounded_ordered_and_concurrent(monkeypatch):
    import uwcc.render as module
    barrier = threading.Barrier(2)
    threads = set()
    def fake(frame, grade, seconds, analysis):
        if seconds < 2:
            threads.add(threading.get_ident())
            barrier.wait(timeout=3)
        return frame
    monkeypatch.setattr(module, "correct", fake)
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
    monkeypatch.setattr(module, "correct", lambda frame, *_: frame)
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
