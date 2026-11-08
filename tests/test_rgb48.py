"""Fused export parity, default dispatch, portable fallback, and buffer safety."""
from types import SimpleNamespace

import numpy as np
import pytest

from uwcc import color
from uwcc.project import Grade


@pytest.fixture
def native():
    module = pytest.importorskip("uwcc._native")
    if not hasattr(module, "lut_rgb48"):
        pytest.skip("Rebuild the native extension for RGB48 tests")
    return module


@pytest.mark.parametrize("size", [2, 17, 33])
@pytest.mark.parametrize("strength", [0, .25, .65, 1, 1.5, 2])
def test_fused_default_and_scalar_match_original_float_kernel(native, size, strength):
    rng = np.random.default_rng(83)
    cube = rng.uniform(-.2, 1.2, (size, size, size, 3)).astype(np.float32)
    # Every eight-pixel vector tail, plus a larger odd-sized block.
    for count in list(range(1, 18)) + [65, 2007]:
        frame = rng.uniform(-.2, 1.2, (1, count, 3)).astype(np.float32)
        frame[0, 0] = [np.nan, np.inf, -np.inf]
        if count > 3:
            frame[0, 1:4] = [[0, 0, 0], [1, 1, 1], [.5, .5, .5]]
        auto = np.empty_like(frame)
        native.apply_lut(frame, cube, auto, size, True)
        with np.errstate(invalid="ignore"):
            reference = np.nan_to_num(np.clip(frame+(auto-frame)*np.float32(strength), 0, 1))
            expected = np.rint(reference*65535).astype(np.uint16)
        for use_avx2 in (True, False):
            actual = np.empty_like(expected)
            native.lut_rgb48(frame, cube, actual, size, strength, use_avx2)
            np.testing.assert_array_equal(actual, expected)
    assert native.rgb48_backend() in ("avx2", "scalar")
    # Forcing scalar on one invocation does not change the default dispatch.
    default = native.rgb48_backend()
    native.lut_rgb48(frame, cube, actual, size, strength)
    assert native.rgb48_backend() == default


def test_fused_rounding_boundaries(native):
    frame = np.zeros((1, 9, 3), np.float32)  # AVX2 block and scalar tail.
    for code in (.5, 1.5, 2.5, 127.5, 255.5, 4095.5, 32767.5, 65534.5):
        value = np.float32(code)/np.float32(65535)
        cube = np.full((2, 2, 2, 3), value, np.float32)
        for use_avx2 in (True, False):
            out = np.empty_like(frame, dtype=np.uint16)
            native.lut_rgb48(frame, cube, out, 2, 1, use_avx2)
            assert np.all(out == np.rint(value*np.float32(65535)))


def test_fused_readonly_input_and_invalid_frame_shape(native):
    frame = np.zeros((1, 9, 3), np.float32)
    cube = np.zeros((2, 2, 2, 3), np.float32)
    frame.flags.writeable = cube.flags.writeable = False
    out = np.empty(frame.shape, np.uint16)
    native.lut_rgb48(frame, cube, out, 2, 1)
    assert not out.any()
    with pytest.raises(ValueError, match="three channels"):
        color.correct_rgb48(np.zeros((9, 11, 4), np.float32), Grade(input_profile="rec709"), 0, analysis())


def test_fused_rejects_invalid_and_overlapping_buffers(native):
    frame = np.zeros((1, 9, 3), np.float32)
    cube = np.zeros((2, 2, 2, 3), np.float32)
    output = np.empty(frame.shape, np.uint16)
    for strength in (-.01, 2.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            native.lut_rgb48(frame, cube, output, 2, strength)
    for src, lut, dst, n in [
        (frame.astype(np.float64), cube, output, 2),
        (frame, cube, output.astype(np.uint8), 2),
        (frame, cube, output, 3),
        (frame, cube, output, 1),
        (frame, cube, output[:, :1], 2),
        (frame, cube, np.ndarray(frame.shape, dtype=np.uint16, buffer=frame, offset=2), 2),
        (frame[:, :2], cube, np.ndarray((1, 2, 3), dtype=np.uint16, buffer=cube, offset=2), 2),
        (np.ndarray(frame.shape, dtype=np.float32, buffer=bytearray(frame.nbytes+1), offset=1), cube, output, 2),
    ]:
        with pytest.raises((ValueError, BufferError)):
            native.lut_rgb48(src, lut, dst, n, 1)
    output.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        native.lut_rgb48(frame, cube, output, 2, 1)


def analysis():
    return {"version": 1, "samples": [{"time": 0, "scene": 0, "gain": [1, 1, 1], "exposure": 0,
            "recipe": {"gain": [1.2, 1, .8], "black": .01, "white": .8,
                       "gamma": .9, "vibrance": .1, "contrast": 1.06}}]}


@pytest.mark.parametrize("case", ["neutral", "manual", "zero", "hdr", "log", "missing", "legacy", "animated"])
def test_export_guards_and_animation_preserve_existing_graph(native, monkeypatch, case):
    frame = np.random.default_rng(4).uniform(.1, .9, (9, 11, 3)).astype(np.float32)
    grade, curve = Grade(input_profile="rec709"), analysis()
    if case == "manual":
        grade.values["contrast"] = 15
    elif case == "zero":
        grade.values["strength"] = 0
    elif case == "hdr":
        grade.output_profile = "hlg"
    elif case == "log":
        grade.input_profile = "slog3"
    elif case == "missing":
        curve = None
    elif case == "legacy":
        del curve["samples"][0]["recipe"]
    elif case == "animated":
        grade.set_key("strength", 0, 0)
        grade.set_key("strength", 1, 200)
    calls = []

    def fused(*args):
        calls.append(args)
        return native.lut_rgb48(*args)

    monkeypatch.setattr(color, "_native", SimpleNamespace(
        apply_lut=native.apply_lut, encode709=native.encode709, lut_rgb48=fused))
    for t in ((0, .25, .5, .75, 1) if case == "animated" else (.4,)):
        reference = color.correct(frame, grade, t, curve)
        expected = np.rint(reference*65535).astype("<u2")
        actual = color.correct_rgb48(frame[:, ::-1], grade, t, curve)
        np.testing.assert_array_equal(actual, expected[:, ::-1])
        assert actual.dtype == np.dtype("<u2")
    assert len(calls) == (4 if case == "animated" else 1 if case == "neutral" else 0)


def test_export_without_native_keeps_numpy_path(monkeypatch):
    monkeypatch.setattr(color, "_native", None)
    frame = np.random.default_rng(4).random((9, 11, 3), dtype=np.float32)
    grade, curve = Grade(input_profile="rec709"), analysis()
    expected = np.rint(color.correct(frame, grade, .4, curve)*65535).astype("<u2")
    np.testing.assert_array_equal(color.correct_rgb48(frame, grade, .4, curve), expected)
