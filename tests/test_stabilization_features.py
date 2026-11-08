"""Regional corner detection must retain full-image masked tracking behavior."""
import numpy as np
import pytest

from uwcc import finishing


def masked_features(cv, gray, x0, y0, x1, y1):
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = 255
    return cv.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=.008,
                                 minDistance=5, mask=mask, blockSize=7)


@pytest.mark.parametrize("shape", [(24, 32), (180, 320), (320, 180), (16, 256)])
@pytest.mark.parametrize("texture", ["random", "checker", "flat"])
def test_regional_features_match_masks_at_cell_and_image_boundaries(shape, texture):
    cv = pytest.importorskip("cv2")
    height, width = shape
    if texture == "random":
        gray = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    elif texture == "checker":
        y, x = np.indices(shape)
        gray = (((x//4+y//4) % 2)*255).astype(np.uint8)
    else:
        gray = np.full(shape, 120, np.uint8)
    gray = gray[:, ::-1]  # Also exercise noncontiguous input.
    regions = [(0, 0, width, height), (0, 0, width//2, height//2),
               (width//2, height//2, width, height),
               (1, 1, width-1, height-1), (2, 2, 5, 5),
               (width//4, height//4, width//2, height//2), (3, 3, 3, 5)]
    for bounds in regions:
        expected = masked_features(cv, gray, *bounds)
        actual = finishing._regional_features(cv, gray, *bounds)
        if expected is None:
            assert actual is None
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("region", ["seabed", "whole"])
def test_regional_tracking_preserves_motion_crop_and_warp_through_a_cut(monkeypatch, region):
    cv = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    base = rng.uniform(.1, .8, (96, 160, 3)).astype(np.float32)
    second = rng.uniform(.5, 1, base.shape).astype(np.float32)
    frames = [np.roll(base if i < 6 else second, (i % 3-1)*3, axis=1) for i in range(12)]

    def run():
        cv.setRNGSeed(8)
        tracker = finishing.BackgroundStabilizer(25, region)
        matrices = [finishing.safe_center(tracker.estimate(frame), 160, 96) for frame in frames]
        zoom = max(finishing.required_zoom(matrix, 160, 96) for matrix in matrices)
        return matrices, zoom

    actual, zoom = run()
    monkeypatch.setattr(finishing, "_regional_features", masked_features)
    expected, expected_zoom = run()
    np.testing.assert_array_equal(actual, expected)
    assert zoom == expected_zoom
    for frame, matrix, reference in zip(frames, actual, expected):
        np.testing.assert_array_equal(finishing.warp_cropped(frame, matrix, zoom),
                                      finishing.warp_cropped(frame, reference, expected_zoom))
