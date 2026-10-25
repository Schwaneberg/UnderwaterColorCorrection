import subprocess

import numpy as np
import pytest

from uwcc.analysis import learn_reference, validate_analysis
from uwcc.color import apply_lut, correct
from uwcc.enhance import estimate_recipe, fit_colors, cube_at, validate_recipe
from uwcc.errors import UWCCError
from uwcc.media import probe, read_frame
from uwcc.project import Grade


def test_learned_color_fit_generalizes_to_unseen_pixels():
    rng = np.random.default_rng(14)
    train = rng.uniform(0, .8, (60, 60, 3)).astype(np.float32)
    target = np.clip(train*[1.1, .95, 1.05]+.03, 0, 1)
    coefficients, error = fit_colors(train, target)
    assert error < .002
    data = {"reference_fit": coefficients}
    unseen = rng.uniform(0, .8, (20, 20, 3)).astype(np.float32)
    output = apply_lut(unseen, cube_at(data, 0), uniform=True)
    assert np.mean(abs(output-np.clip(unseen*[1.1, .95, 1.05]+.03, 0, 1))) < .002


def test_automatic_levels_expand_flat_tonal_range_without_invalid_pixels():
    raw = np.broadcast_to(np.linspace(.2, .55, 60)[:, None, None], (60, 60, 3)).copy().astype(np.float32)
    recipe = estimate_recipe(raw)
    validate_recipe(recipe)
    analysis = {"samples": [{"time": 0, "scene": 0, "gain": [1, 1, 1], "exposure": 0, "recipe": recipe}]}
    grade = Grade(input_profile="rec709")
    output = correct(raw, grade, analysis=analysis)
    assert np.isfinite(output).all()
    assert output.max()-output.min() > raw.max()-raw.min()
    assert np.max(abs(output[..., 0]-output[..., 1])) < .001
    grade.values["strength"] = 0
    assert np.allclose(correct(raw, grade, analysis=analysis), raw, atol=.001)


def test_encoded_fit_respects_manual_controls_and_shot_boundaries():
    rng = np.random.default_rng(4)
    raw = rng.uniform(.1, .5, (20, 20, 3)).astype(np.float32)
    left, _ = fit_colors(raw, raw+.1)
    right, _ = fit_colors(raw, raw*.5)
    analysis = {"samples": [{"time": t, "scene": scene, "gain": [1, 1, 1], "exposure": 0, "encoded_fit": fit}
                             for t, scene, fit in [(0, 0, left), (1, 1, right)]]}
    assert np.allclose(cube_at(analysis, .99), cube_at(analysis, 0))
    grade = Grade(input_profile="rec709")
    baseline = correct(raw, grade, analysis=analysis)
    grade.values["exposure"] = .5
    assert correct(raw, grade, analysis=analysis).mean() > baseline.mean()


def test_reference_video_learning_and_validation(clip, tmp_path):
    target = tmp_path/"graded.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "eq=brightness=0.05:contrast=1.1",
                    "-an", "-c:v", "ffv1", str(target)], check=True)
    info = probe(clip)
    grade = Grade(input_profile="rec709")
    analysis = learn_reference(info, grade, target)
    validate_analysis(analysis)
    raw = read_frame(info, .25, 160)
    expected = read_frame(probe(target), .25, 160)
    actual = correct(raw, grade, .25, analysis)
    assert np.mean(abs(actual-expected)) < .03
    grade.input_profile = "slog3"
    with pytest.raises(UWCCError, match="Rec.709"):
        learn_reference(info, grade, target)


def test_invalid_new_analysis_fields_rejected():
    data = {"version": 1, "samples": [{"time": 0, "gain": [1, 1, 1], "exposure": 0, "encoded_fit": [[float('nan')]*3]*20}]}
    with pytest.raises(UWCCError):
        validate_analysis(data)
    with pytest.raises(UWCCError):
        validate_recipe({"gain": [1, 1, 1]})
