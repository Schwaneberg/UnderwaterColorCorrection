import threading

import numpy as np
import pytest

from uwcc.errors import Cancelled
from uwcc.finishing import BackgroundStabilizer, TemporalDenoiser
from uwcc.neural import builtin_model


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("prefetch", [False, True])
def test_neural_stream_keeps_frame_count_and_endpoint_order(count, prefetch):
    model = TemporalDenoiser.__new__(TemporalDenoiser)
    model.cancel = None
    model.gpu_variant = False
    model.predict = lambda frames: frames[2]
    frames = [np.full((8, 8, 3), .1+i*.05, np.float32) for i in range(count)]
    result = list(model.stream(iter(frames), prefetch=prefetch))
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
    if not manifest or not (Path(manifest).parent / "model.onnx").is_file():
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


def pipeline_model(gpu=True):
    model = TemporalDenoiser.__new__(TemporalDenoiser)
    model.cancel = threading.Event()
    model.gpu_variant = gpu
    model.size, model.halo, model.sigma = 40, 16, .01
    model.progress = lambda *_: None
    model.report = lambda *_: None
    model.cpu_artifact = ({"size": 40}, "unused")
    class Network:
        def run(self, values):
            return values["frames"][:, 6:9].copy()
    model.network = Network()
    return model


def test_gpu_tensor_preparation_overlaps_inference_and_stays_one_frame_ahead():
    model = pipeline_model()
    next_prepared = threading.Event()
    preparation_threads, inference_threads, decoded = [], [], []
    prepare = model.prepare_full_frame
    def prepare_window(frames):
        preparation_threads.append(threading.get_ident())
        result = prepare(frames)
        if len(preparation_threads) == 2:
            next_prepared.set()
        return result
    model.prepare_full_frame = prepare_window
    run = model.network.run
    def infer(values):
        inference_threads.append(threading.get_ident())
        if len(inference_threads) == 1:
            assert next_prepared.wait(3), "Next tensor was not prepared during inference"
        return run(values)
    model.network.run = infer
    def frames():
        for i in range(10):
            decoded.append(i)
            yield np.full((9, 13, 3), .1+i*.03, np.float32)
    stream = model.stream(frames(), prefetch=True)
    first = next(stream)
    assert decoded == [0, 1, 2, 3]  # current window + one next window, no extra queue
    results = [first, *stream]
    assert len(results) == 10
    assert all(np.allclose(frame, .1+i*.03, atol=1e-6) for i, frame in enumerate(results))
    assert set(inference_threads) == {threading.get_ident()}
    assert len(set(preparation_threads)) == 1
    assert preparation_threads[0] != threading.get_ident()


@pytest.mark.parametrize("prefetch", [False, True])
def test_stream_gpu_fallback_preserves_prefetched_frames(monkeypatch, prefetch):
    import uwcc.finishing as finishing
    from uwcc.errors import UWCCError
    model = pipeline_model()
    run = model.network.run
    calls = []
    def infer(values):
        calls.append("gpu")
        if len(calls) == 2:
            raise UWCCError("GPU failed")
        return run(values)
    model.network.run = infer
    class CPU:
        def __init__(self, *args, **kwargs):
            assert kwargs["cpu_only"]
        def run(self, values):
            calls.append("cpu")
            return values["frames"][:, 6:9].copy()
    monkeypatch.setattr(finishing, "Network", CPU)
    frames = [np.full((9, 13, 3), .1+i*.03, np.float32) for i in range(7)]
    results = list(model.stream(iter(frames), prefetch=prefetch))
    assert len(results) == len(frames)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(results, frames))
    assert not model.gpu_variant
    assert calls[:2] == ["gpu", "gpu"]
    assert set(calls[2:]) == {"cpu"}


@pytest.mark.parametrize("stop", ["cancel", "close", "inference-error"])
def test_pipeline_shutdown_unblocks_source_and_joins_worker(stop):
    from uwcc.errors import UWCCError
    model = pipeline_model()
    blocked, release, closed = threading.Event(), threading.Event(), threading.Event()
    def frames():
        try:
            for _ in range(3):
                yield np.full((8, 8, 3), .3, np.float32)
            blocked.set()
            assert release.wait(3), "Pipeline did not interrupt its source read"
            raise OSError("Decoder stopped")
        finally:
            closed.set()
    run = model.network.run
    def infer(values):
        assert blocked.wait(3)
        if stop == "inference-error":
            # Avoid GPU recovery: test cleanup of a fatal inference error.
            model.gpu_variant = False
            raise UWCCError("Fatal inference error")
        return run(values)
    model.network.run = infer
    stream = model.stream(frames(), prefetch=True, on_stop=release.set)
    if stop == "inference-error":
        with pytest.raises(UWCCError, match="Fatal inference"):
            next(stream)
    else:
        next(stream)
        if stop == "cancel":
            model.cancel.set()
            with pytest.raises(Cancelled):
                next(stream)
        else:
            stream.close()
            assert not model.cancel.is_set()
    assert closed.is_set()
    assert not any(t.name.startswith("uwcc-denoise-prepare") for t in threading.enumerate())


def test_pipeline_preserves_source_error_after_prior_frames():
    model = pipeline_model(gpu=False)
    model.predict = lambda frames: frames[2]
    def frames():
        for _ in range(4):
            yield np.full((8, 8, 3), .3, np.float32)
        raise OSError("Decode failed")
    stream = model.stream(frames(), prefetch=True)
    assert np.allclose(next(stream), .3)
    assert np.allclose(next(stream), .3)
    with pytest.raises(OSError, match="Decode failed"):
        next(stream)


def test_reader_early_close_stops_pipeline_and_decoder(clip, monkeypatch):
    import uwcc.finishing as finishing
    from uwcc.media import FrameReader, probe
    from uwcc.project import Grade
    monkeypatch.setattr(finishing, "TemporalDenoiser", lambda *args: pipeline_model())
    grade = Grade(input_profile="rec709", denoise=3, denoise_engine="ai", denoise_manifest="unused")
    with FrameReader(probe(clip), finishing=grade) as reader:
        stream = iter(reader)
        assert iter(reader) is stream
        assert next(stream).shape == (180, 320, 3)
    assert reader.process.poll() is not None
    assert not reader.watcher.is_alive()
    assert not any(t.name.startswith("uwcc-denoise-prepare") for t in threading.enumerate())
    reader.close()  # Closing an already-stopped reader is harmless.


def test_full_frame_packing_preserves_all_temporal_channels_and_padding():
    model = pipeline_model()
    rng = np.random.default_rng(9)
    frames = [rng.random((9, 13, 3), dtype=np.float32) for _ in range(5)]
    padded = [np.pad(frame, ((0, 3), (0, 3), (0, 0)), mode="reflect") for frame in frames]
    reference = np.ascontiguousarray(np.concatenate([f.transpose(2, 0, 1) for f in padded], axis=0)[None], np.float32)
    values = model.prepare_full_frame(frames)
    assert values["frames"].flags.c_contiguous
    assert np.array_equal(values["frames"], reference)
    assert values["noise"].shape == (1, 1, 12, 16)
    assert np.all(values["noise"] == np.float32(model.sigma))


def test_canceled_pipeline_does_not_read_source():
    model = pipeline_model()
    model.cancel.set()
    def frames():
        pytest.fail("Canceled denoise read a frame")
        yield
    with pytest.raises(Cancelled):
        list(model.stream(frames()))
