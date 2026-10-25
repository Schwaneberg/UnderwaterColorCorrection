import threading

import numpy as np
import pytest

from uwcc.errors import Cancelled
from uwcc.finishing import BackgroundStabilizer, TemporalDenoiser
from uwcc.neural import builtin_model


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_neural_stream_keeps_frame_count_and_endpoint_order(count):
    model = TemporalDenoiser.__new__(TemporalDenoiser)
    model.predict = lambda frames: frames[2]
    frames = [np.full((8, 8, 3), .1+i*.05, np.float32) for i in range(count)]
    result = list(model.stream(iter(frames)))
    assert len(result) == count
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(frames, result))


def test_stabilization_preserves_float_precision():
    pytest.importorskip("cv2")
    model = BackgroundStabilizer(30)
    frame = np.full((80, 120, 3), .1234567, np.float32)
    model.process(frame)
    result = model.process(frame)
    assert result.dtype == np.float32
    # Bilinear crop interpolation can round by an ulp; an 8-bit round trip
    # would change this value by several thousandths instead.
    assert np.max(abs(result-frame)) < 1e-7


def test_real_fastdvdnet_reduces_noise_and_cancels():
    pytest.importorskip("cv2")
    manifest = builtin_model("denoise")
    if not manifest:
        pytest.skip("Optional FastDVDnet weights are not installed")
    rng = np.random.default_rng(8)
    frames = [np.clip(.5+rng.normal(0, 7.5/255, (96, 96, 3)), 0, 1).astype(np.float32) for _ in range(5)]
    model = TemporalDenoiser(manifest, 3)
    result = model.predict(frames)
    assert np.mean((result-.5)**2) < .5*np.mean((frames[2]-.5)**2)
    model.cancel = threading.Event()
    model.cancel.set()
    with pytest.raises(Cancelled):
        model.predict(frames)


def test_denoise_reports_backend_and_tiles_before_inference(monkeypatch):
    from itertools import count
    from types import SimpleNamespace
    import uwcc.finishing as finishing
    events = []
    class FakeNetwork:
        execution_label = "CPU · test backend"
        def __init__(self, *_):
            pass
        def run(self, values):
            assert events and events[0].startswith("AI denoise · CPU · test backend")
            return values["frames"][:, 6:9]
    monkeypatch.setattr(finishing, "artifact", lambda *_: ({"frames": 5, "size": 40}, "unused"))
    monkeypatch.setattr(finishing, "Network", FakeNetwork)
    ticks = count(step=.6)
    monkeypatch.setattr(finishing, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    model = TemporalDenoiser("unused", 2, progress=events.append)
    frame = np.full((16, 16, 3), .4, np.float32)
    result = model.predict([frame]*5)
    assert np.allclose(result, frame)
    assert "tile 0/4" in events[0]
    assert "tile 4/4" in events[-1]
    assert all("frame 1" in message for message in events)


@pytest.mark.parametrize("fail_gpu", [False, True])
def test_gpu_full_frame_shape_and_bounded_cpu_fallback(tmp_path, monkeypatch, fail_gpu):
    import uwcc.finishing as finishing
    from uwcc.errors import UWCCError
    messages, calls = [], []
    static = {"frames": 5, "size": 128, "gpu_variant_manifest": "gpu-float16.json"}
    variant = {"frames": 5, "size": 512, "gpu_variant": True, "gpu_validated": True, "precision": "float16"}
    def artifact(path, task):
        return (variant if str(path).endswith("gpu-float16.json") else static), "unused.onnx"
    class FakeNetwork:
        def __init__(self, data, path, cpu_only=False, **_):
            self.data = data
            self.uses_cuda = not cpu_only
            self.execution_label = "CUDA GPU test" if self.uses_cuda else "CPU test"
            calls.append((data["size"], cpu_only))
        def run(self, values):
            if self.data.get("gpu_variant") and fail_gpu:
                raise UWCCError("GPU unavailable")
            return values["frames"][:, 6:9]
    monkeypatch.setattr(finishing, "artifact", artifact)
    monkeypatch.setattr(finishing, "Network", FakeNetwork)
    model = TemporalDenoiser(tmp_path/"manifest.json", 2, progress=messages.append)
    assert model.gpu_variant
    frame = np.full((17, 19, 3), .4, np.float32)
    actual = model.predict([frame]*5)
    assert actual.shape == frame.shape
    assert np.allclose(actual, frame)
    assert model.gpu_variant != fail_gpu
    if fail_gpu:
        assert calls[-1] == (128, True)
        assert any("falling back to CPU tiles" in message for message in messages)
    model.cancel = threading.Event()
    model.cancel.set()
    before = len(calls)
    with pytest.raises(Cancelled):
        model.predict([frame]*5)
    assert len(calls) == before
