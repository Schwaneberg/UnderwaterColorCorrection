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


def _reference_tetrahedral(rgb, lut, uniform):
    """Original full-array interpolation, independent of the block evaluator."""
    n = lut.shape[0]
    coords = np.clip(rgb if uniform else np.log2(1+np.maximum(rgb, 0))/np.log2(17), 0, 1)*(n-1)
    indices = np.minimum(coords.astype(np.int32), n-2)
    knots = np.linspace(0, 1, n) if uniform else 2**(np.linspace(0, 1, n)*np.log2(17))-1
    fractions = np.clip((rgb-knots[indices])/(knots[indices+1]-knots[indices]), 0, 1)
    order = np.argsort(-fractions, axis=-1, kind="stable")
    weights = np.take_along_axis(fractions, order, axis=-1)
    result = lut[indices[..., 0], indices[..., 1], indices[..., 2]]*(1-weights[..., 0, None])
    cursor = indices.copy()
    for axis in range(3):
        cursor += np.eye(3, dtype=np.int32)[order[..., axis]]
        weight = weights[..., axis]-(weights[..., axis+1] if axis < 2 else 0)
        result += lut[cursor[..., 0], cursor[..., 1], cursor[..., 2]]*weight[..., None]
    return result


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("size", [2, 17, 33])
def test_numpy_lut_blocks_preserve_reference_at_ties_edges_and_boundaries(monkeypatch, uniform, size):
    from uwcc import color
    monkeypatch.setattr(color, "_native", None)
    rng = np.random.default_rng(41)
    # More than one block; noncontiguous float64 input, out-of-range channels,
    # all six axis orderings, and equal fractions on the neutral axis.
    rgb = rng.uniform(-.2, 1.2 if uniform else 18, (257, 257, 3))
    rgb[0] = np.linspace(0, 1 if uniform else 16, 257)[:, None]
    rgb[1, :6] = np.array([[.2, .5, .8], [.2, .8, .5], [.5, .2, .8],
                           [.5, .8, .2], [.8, .2, .5], [.8, .5, .2]])
    rgb = rgb[:, ::-1]
    lut = rng.uniform(-.1, 2, (size, size, size, 3)).astype(np.float32)
    reference = _reference_tetrahedral(rgb, lut, uniform)
    output = apply_lut(rgb, lut, uniform)
    assert output.dtype == np.float32
    assert output.shape == rgb.shape
    assert np.allclose(output, reference, atol=2e-5, rtol=0)
