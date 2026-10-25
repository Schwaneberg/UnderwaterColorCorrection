import math

import numpy as np
import pytest

from uwcc.finishing import crop_matrix, required_zoom, safe_center, warp_cropped, build_stabilization_plan
from uwcc.media import probe
from uwcc.project import Grade


@pytest.mark.parametrize("shape", [(90, 160), (160, 90), (64, 512)])
def test_crop_never_reads_outside_source_for_rotated_translated_frames(shape):
    pytest.importorskip("cv2")
    height, width = shape
    rng = np.random.default_rng(42)
    transforms = []
    for _ in range(25):
        angle = rng.uniform(-.4, .4)
        matrix = np.eye(3)
        matrix[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        matrix[:2, 2] = rng.uniform(-.15, .15, 2)*[width, height]
        transforms.append(safe_center(matrix, width, height))
    zoom = max(required_zoom(matrix, width, height) for matrix in transforms)
    # A constant valid image makes even a partly sampled black border detectable.
    frame = np.full((height, width, 3), .75, np.float32)
    for matrix in transforms:
        actual = warp_cropped(frame, matrix, zoom)
        assert np.max(abs(actual-.75)) < 1e-6
        inverse = np.linalg.inv(crop_matrix(matrix, width, height, zoom))
        for x, y in [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]:
            source = inverse @ [x, y, 1]
            assert 0 <= source[0] <= width-1
            assert 0 <= source[1] <= height-1


def test_plan_uses_one_fixed_zoom_for_entire_video(clip):
    pytest.importorskip("cv2")
    info = probe(clip)
    plan = build_stabilization_plan(info, Grade(input_profile="rec709", stabilize=True))
    assert len(plan.matrices) == 18
    assert plan.zoom >= 1
    assert all(required_zoom(matrix, info.width, info.height) <= plan.zoom for matrix in plan.matrices)
