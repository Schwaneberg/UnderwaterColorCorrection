import numpy as np
import pytest

from uwcc.color import (apply_lut, bt709_decode, bt709_encode, correct, decode, encode,
                        hlg_decode, hlg_encode, identity_lut, pq_decode, pq_encode)
from uwcc.project import Grade


def test_transfer_round_trips():
    x = np.linspace(0, 1, 1024, dtype=np.float32)
    assert np.max(np.abs(bt709_encode(bt709_decode(x))-x)) < .001
    assert np.max(np.abs(pq_encode(pq_decode(x))-x)) < .0001
    assert np.max(np.abs(hlg_encode(hlg_decode(x))-x)) < .0001


@pytest.mark.parametrize("profile", ["rec709", "hlg", "pq"])
def test_gamut_and_transfer_round_trip(profile):
    random = np.random.default_rng(0).uniform(.05, .7, (20, 20, 3)).astype(np.float32)
    assert np.allclose(encode(decode(random, profile), profile), random, atol=.001)


def test_identity_controls_and_lut():
    random = np.random.default_rng(4).uniform(.05, .8, (10, 12, 3)).astype(np.float32)
    result = correct(random, Grade(input_profile="rec709"))
    assert np.max(np.abs(result-random)) < .001
    # Log-domain piecewise linear interpolation has a small expected shaper error.
    assert np.max(np.abs(apply_lut(random, identity_lut())-random)) < .012


def test_controls_change_pixels_and_remain_finite():
    from uwcc.project import CONTROLS
    random = np.random.default_rng(1).uniform(.1, .7, (40, 40, 3)).astype(np.float32)
    baseline = correct(random, Grade(input_profile="rec709"))
    for name, (_, low, high, default, _) in CONTROLS.items():
        if name == "strength":
            continue
        grade = Grade(input_profile="rec709")
        grade.values[name] = high
        output = correct(random, grade)
        assert np.isfinite(output).all(), name
        assert output.min() >= 0 and output.max() <= 1, name
        assert not np.allclose(output, baseline), name


def test_log_gray_reference_values():
    # Sony S-Log3 18% scene gray is code value 420/1023.
    gray = np.ones((2, 2, 3), dtype=np.float32)*420/1023
    linear = decode(gray, "slog3")
    assert np.allclose(linear, .18, atol=.0001)
    slog2 = np.full((2, 2, 3), .339532524633774, dtype=np.float32)
    assert np.allclose(decode(slog2, "slog2"), .18, atol=.0001)


def test_sony_video_range_expansion_is_not_applied_twice():
    from types import SimpleNamespace
    from uwcc.media import prepare_frame
    # Expanded RGB from a neutral limited-range S-Log3 patch at code 420.
    expanded = np.full((2, 2, 3), (420-64)/876, dtype=np.float32)
    restored = prepare_frame(expanded, SimpleNamespace(color_range="tv"), "slog3")
    assert np.allclose(decode(restored, "slog3"), .18, atol=.0001)


def test_identity_lut_grid_nodes():
    lut = identity_lut()
    assert np.allclose(apply_lut(lut, lut), lut, atol=2e-5)


def test_analysis_gains_keep_frame_graph_float32():
    from uwcc.color import gain_matrix
    matrix = gain_matrix(np.array([1.2, 1, .8], dtype=np.float64), .2)
    assert matrix.dtype == np.float32
    rgb = np.ones((8, 8, 3), dtype=np.float32)
    assert (rgb @ matrix).dtype == np.float32
