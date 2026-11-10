"""Native controls versus the preserved graph, not versus another fast path."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from uwcc import color
from uwcc.project import Grade


@pytest.fixture
def native():
    module = pytest.importorskip("uwcc._native")
    if not hasattr(module, "manual_rgb48"):
        pytest.skip("Rebuild the native extension for manual-color tests")
    return module


def curve():
    return {"samples": [{"time": 0, "scene": 0, "gain": [1, 1, 1], "exposure": 0,
                         "recipe": {"gain": [1.2, 1, .8], "black": .01,
                                    "white": .8, "gamma": .9, "vibrance": .1, "contrast": 1.06}}]}


def reference(frame, grade, analysis, seconds, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(color, "manual", color._manual_numpy)
        return np.rint(color.correct(frame, grade, seconds, analysis)*65535).astype("<u2")


CONTROLS = [
    {"black_point": 36}, {"black_point": -100}, {"black_point": 100},
    {"temperature": -100}, {"temperature": 100}, {"tint": -100}, {"tint": 100},
    {"exposure": -4}, {"exposure": 4}, {"contrast": -100}, {"contrast": 100},
    {"shadows": -100}, {"shadows": 100}, {"highlights": -100}, {"highlights": 100},
    {"saturation": 0}, {"saturation": 200}, {"hue": -180}, {"hue": 137},
    {"temperature": 53, "tint": -43, "exposure": .7, "contrast": 27,
     "shadows": 20, "highlights": -50, "black_point": 36, "saturation": 125, "hue": 37},
]


@pytest.mark.parametrize("controls", CONTROLS)
@pytest.mark.parametrize("analysis", [None, curve()])
def test_manual_export_matches_reference(native, monkeypatch, controls, analysis):
    frame = np.random.default_rng(123).random((49, 57, 3), dtype=np.float32)
    frame[0, :5] = [[0, 0, 0], [1, 1, 1], [.081, .081, .081],
                     [.001, .002, .003], [.5, .5, .5]]
    grade = Grade(input_profile="rec709")
    grade.values.update(controls)
    expected = reference(frame, grade, analysis, .3, monkeypatch)
    actual = color.correct_rgb48(frame, grade, .3, analysis)
    difference = np.abs(actual.astype(int)-expected.astype(int))
    assert difference.max() <= 2
    assert difference.mean() < .02
    assert actual.dtype == np.dtype("<u2")


def test_block_boundaries_noncontiguous_and_concurrent_frames(native, monkeypatch):
    monkeypatch.setattr(color, "_MANUAL_BLOCK_PIXELS", 37)
    frame = np.random.default_rng(91).random((63, 71, 3), dtype=np.float32)[:, ::-1]
    frame.flags.writeable = False
    grade = Grade(input_profile="rec709")
    grade.values["black_point"] = 36
    expected = reference(frame, grade, curve(), 0, monkeypatch)
    with ThreadPoolExecutor(max_workers=3) as pool:
        images = list(pool.map(lambda _: color.correct_rgb48(frame, grade, 0, curve()), range(3)))
    for image in images:
        np.testing.assert_array_equal(image, images[0])
        assert np.abs(image.astype(int)-expected.astype(int)).max() <= 2


def test_animated_controls_and_strength(native, monkeypatch):
    frame = np.random.default_rng(6).random((31, 29, 3), dtype=np.float32)
    grade = Grade(input_profile="rec709")
    grade.set_key("black_point", 0, -50)
    grade.set_key("black_point", 1, 50)
    grade.set_key("strength", 0, 0)
    grade.set_key("strength", 1, 200)
    for seconds in (0, .25, .5, .75, 1):
        expected = reference(frame, grade, curve(), seconds, monkeypatch)
        actual = color.correct_rgb48(frame, grade, seconds, curve())
        assert np.abs(actual.astype(int)-expected.astype(int)).max() <= 2


@pytest.mark.parametrize("profile,output", [("rec709", "rec709"), ("slog3", "rec709"),
                                           ("hlg", "pq"), ("pq", "hlg")])
@pytest.mark.parametrize("spatial", [{"dehaze": 30}, {"sharpness": 30}])
def test_spatial_finishing_keeps_original_graph(native, monkeypatch, profile, output, spatial):
    frame = np.random.default_rng(18).uniform(.05, .8, (47, 59, 3)).astype(np.float32)
    grade = Grade(input_profile=profile, output_profile=output)
    grade.values.update(black_point=36, **spatial)
    expected = reference(frame, grade, curve(), 0, monkeypatch)
    np.testing.assert_array_equal(color.correct_rgb48(frame, grade, 0, curve()), expected)


def test_numpy_only_keeps_original_output(monkeypatch):
    monkeypatch.setattr(color, "_native", None)
    frame = np.random.default_rng(21).random((17, 23, 3), dtype=np.float32)
    grade = Grade(input_profile="rec709")
    grade.values["black_point"] = 36
    expected = reference(frame, grade, curve(), 0, monkeypatch)
    np.testing.assert_array_equal(color.correct_rgb48(frame, grade, 0, curve()), expected)


@pytest.mark.parametrize("fused", [False, True])
def test_native_manual_buffer_validation(native, fused):
    frame = np.zeros((1, 13, 3), np.float32)
    grade = Grade(input_profile="rec709")
    grade.values["black_point"] = 36
    matrices, parameters = color._manual_parameters(grade.at(0), fused=fused)
    output = np.zeros(frame.shape, dtype=np.uint16 if fused else np.float64)
    run = native.manual_rgb48 if fused else native.manual_color
    for source, matrix, params, dest in (
        (frame.astype(np.float64), matrices, parameters, output),
        (frame, matrices[:-1], parameters, output),
        (frame, matrices, parameters.astype(np.float32), output),
        (frame, matrices, parameters[:-1], output),
        (frame, matrices, parameters, output[:, :1]),
        (frame, matrices, parameters, output.astype(np.float32)),
        (frame, matrices, parameters*np.nan, output),
        (frame, matrices*np.inf, parameters, output),
        (np.ndarray(frame.shape, dtype=np.float32, buffer=bytearray(frame.nbytes+1), offset=1),
         matrices, parameters, output),
    ):
        with pytest.raises((ValueError, BufferError)):
            run(source, matrix, params, dest)
    storage = np.zeros(256, dtype=np.float64)
    with pytest.raises(ValueError, match="overlap"):
        source = storage.view(np.float32)[:frame.size]
        dest = storage.view(output.dtype)[:frame.size]
        run(source, matrices, parameters, dest)
    frame.flags.writeable = False
    run(frame, matrices, parameters, output)
    output.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        run(frame, matrices, parameters, output)


@pytest.mark.parametrize("profile,output", [("slog2", "rec709"), ("slog3", "rec709"),
                                           ("hlg", "pq"), ("pq", "hlg")])
def test_log_hdr_manual_reference(native, monkeypatch, profile, output):
    frame = np.random.default_rng(8).uniform(.02, .9, (41, 53, 3)).astype(np.float32)
    grade = Grade(input_profile=profile, output_profile=output)
    grade.values.update(black_point=36, saturation=120, hue=17)
    expected = reference(frame, grade, None, 0, monkeypatch)
    actual = color.correct_rgb48(frame, grade, 0, None)
    assert np.abs(actual.astype(int)-expected.astype(int)).max() <= 2


def test_nonfinite_and_empty_input(native, monkeypatch):
    grade = Grade(input_profile="rec709")
    grade.values["black_point"] = 36
    frame = np.array([[[np.nan, .1, .5], [np.inf, .3, .6], [-np.inf, .2, .4]]], np.float32)
    with np.errstate(invalid="ignore"):
        expected = reference(frame, grade, None, 0, monkeypatch)
        np.testing.assert_array_equal(color.correct_rgb48(frame, grade, 0, None), expected)
    assert color.correct_rgb48(np.empty((0, 3), np.float32), grade).shape == (0, 3)
    with pytest.raises(ValueError, match="three channels"):
        color.correct_rgb48(np.zeros((3, 4), np.float32), grade)
