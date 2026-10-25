"""Parity with the reference graph, including gamut edges and tetrahedron ties."""
import numpy as np
import pytest

import uwcc.color as color


@pytest.mark.parametrize("size", [2, 17, 33])
def test_native_lut_matches_reference(monkeypatch, size):
    native = pytest.importorskip("uwcc._native")
    rng = np.random.default_rng(22)
    rgb = rng.uniform(-.2, 18, (30, 40, 3)).astype(np.float32)
    rgb[0] = np.linspace(0, 16, 40)[:, None]  # equal fractions; stable tie order
    lut = rng.uniform(-.1, 2, (size, size, size, 3)).astype(np.float32)
    monkeypatch.setattr(color, "_native", None)
    reference = color.apply_lut(rgb, lut)
    monkeypatch.setattr(color, "_native", native)
    # Exercise noncontiguous input as well as native float64 conversion.
    result = color.apply_lut(rgb[:, ::-1].astype(np.float64), lut)
    assert result.dtype == np.float32
    assert np.allclose(result, reference[:, ::-1], atol=2e-5)


@pytest.mark.parametrize("tone", [False, True])
def test_native_sdr_matches_reference(monkeypatch, tone):
    native = pytest.importorskip("uwcc._native")
    rgb = np.random.default_rng(10).uniform(-1, 20, (30, 40, 3)).astype(np.float32)
    monkeypatch.setattr(color, "_native", None)
    reference = color.encode(rgb, "rec709", tone)
    monkeypatch.setattr(color, "_native", native)
    result = color.encode(rgb, "rec709", tone)
    assert np.allclose(result, reference, atol=2e-5)


def test_native_rejects_invalid_buffers():
    native = pytest.importorskip("uwcc._native")
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        native.encode709(rgb.astype(np.float64), rgb, False)
    with pytest.raises(ValueError):
        native.apply_lut(rgb, rgb, np.empty_like(rgb), 17)
    with pytest.raises(ValueError):
        native.encode709(rgb, np.zeros(3, dtype=np.float32), False)


@pytest.mark.parametrize("size", [17, 33])
def test_uniform_encoded_lut_matches_numpy(monkeypatch, size):
    native = pytest.importorskip("uwcc._native")
    rng = np.random.default_rng(9)
    rgb = rng.uniform(-.1, 1.1, (20, 30, 3)).astype(np.float32)
    rgb[0] = np.linspace(0, 1, 30)[:, None]
    lut = rng.random((size, size, size, 3), dtype=np.float32)
    monkeypatch.setattr(color, "_native", None)
    reference = color.apply_lut(rgb, lut, uniform=True)
    monkeypatch.setattr(color, "_native", native)
    assert np.allclose(color.apply_lut(rgb, lut, uniform=True), reference, atol=2e-5)
