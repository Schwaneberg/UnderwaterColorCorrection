import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from uwcc import orientation
from uwcc.errors import Cancelled, UWCCError
from uwcc.media import probe, read_frame
from uwcc.project import Grade, Project, fingerprint, save_preset, load_preset


def sample(correction, probability=.99):
    probabilities = {angle: (1-probability)/3 for angle in (0, 90, 180, 270)}
    probabilities[correction] = probability
    return {"probabilities": probabilities}


def test_consensus_accepts_agreement_and_rejects_ambiguous_clips():
    accepted = orientation.consensus([sample(90)]*9+[sample(270, .7)])
    assert accepted["status"] == "detected" and accepted["correction_cw"] == 90
    assert accepted["agreement"] == 9
    for samples in ([sample(90)]*7+[sample(270)]*3, [sample(90, .8)]*10,
                    [sample(90)]*8+[sample(270)]*2, [sample(90)]*4):
        decision = orientation.consensus(samples)
        assert decision["status"] == "uncertain"
        assert decision["correction_cw"] == 0


def test_quarter_probability_wraparound_and_correction_direction():
    distribution = np.zeros((1, 360), np.float32)
    distribution[0, 359], distribution[0, 1] = .5, .5
    assert orientation.quarter_probabilities(distribution)[0] == 1
    distribution[:] = 0
    distribution[0, 90] = 1
    assert orientation.quarter_probabilities(distribution)[270] == 1
    with pytest.raises(UWCCError):
        orientation.quarter_probabilities(np.ones((1, 360)))


def test_input_letterbox_retains_aspect_and_normalizes_padding():
    image = Image.new("RGB", (640, 360), (255, 0, 0))
    tensor = orientation.input_tensor(image)
    assert tensor.shape == (1, 3, 280, 280) and tensor.flags.c_contiguous
    assert np.all(tensor[0, 0, 100:180, :] > 2)
    assert np.max(np.abs(tensor[:, :, 0, :])) < .02


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_manual_rotation_changes_decoded_geometry_and_pixel_positions(clip, angle):
    info = probe(clip)
    raw = read_frame(info, .5)
    rotated = orientation.oriented_info(info, angle)
    actual = read_frame(rotated, .5)
    expected = np.rot90(raw, -angle//90)
    assert actual.shape == expected.shape
    # Transposing before YUV->RGB conversion may shift chroma interpolation at
    # edges. Interior blocks still identify exactly the same visual regions.
    assert np.mean(np.abs(actual-expected)) < .01
    restored = orientation.oriented_info(rotated, 0)
    assert (restored.width, restored.height, restored.sar) == (info.width, info.height, info.sar)


def test_non_square_pixel_aspect_is_inverted_once(clip):
    info = replace(probe(clip), sar="2:1")
    rotated = orientation.oriented_info(info, 90)
    assert rotated.sar == "1:2"
    assert orientation.oriented_info(rotated, 270).sar == "1:2"
    assert orientation.oriented_info(rotated, 0).sar == "2:1"


def test_manual_and_cached_orientations_do_not_run_detector(clip, monkeypatch):
    info = probe(clip)
    monkeypatch.setattr(orientation, "detect", lambda *_: pytest.fail("Unexpected inference"))
    project = Project(str(clip), Grade(rotation="90"))
    assert orientation.project_info(project, info).width == info.height
    project.grade.rotation = "auto"
    project.orientation = orientation.consensus([sample(270)]*10)
    assert orientation.project_info(project, info).correction_rotation == 270


def test_metadata_orientation_and_missing_model_leave_no_extra_rotation(clip, monkeypatch):
    info = probe(clip)
    assert orientation.detect(info)["status"] == "unavailable"
    monkeypatch.setattr(orientation, "builtin_model", lambda _: pytest.fail("Metadata should take precedence"))
    assert orientation.detect(replace(info, rotation=90))["status"] == "metadata"


def test_saved_orientation_and_manual_preset_round_trip(clip, tmp_path):
    project = Project(str(clip), Grade(rotation="90"), fingerprint(clip))
    project.orientation = orientation.consensus([sample(270)]*10)
    path = tmp_path/"project.json"
    project.save(path)
    assert Project.load(path).orientation == project.orientation
    preset = tmp_path/"preset.json"
    save_preset(preset, project.grade)
    assert load_preset(preset).rotation == "90"
    legacy = json.loads(path.read_text())
    legacy.pop("orientation")
    path.write_text(json.dumps(legacy))
    assert Project.load(path).orientation["status"] == "legacy"
    bad = project.to_dict()
    bad["orientation"] = {"correction_cw": 45, "status": "detected", "message": "bad"}
    path.write_text(json.dumps(bad))
    with pytest.raises(UWCCError, match="orientation"):
        Project.load(path)


def test_saved_uncertain_decision_cannot_apply_rotation():
    with pytest.raises(UWCCError):
        orientation.validate_decision({"correction_cw": 90, "status": "uncertain", "message": "bad"})
    with pytest.raises(UWCCError):
        Grade.from_dict({"rotation": "45"})


@pytest.mark.parametrize("manual", [True, False])
def test_rotated_export_preserves_dimensions_frames_audio_and_source(clip, tmp_path, manual):
    import threading
    from uwcc.media import frame_timing
    from uwcc.render import render
    before = fingerprint(clip)
    project = Project(str(clip), Grade(input_profile="rec709", rotation="90" if manual else "auto"), before)
    if not manual:
        project.orientation = orientation.consensus([sample(270)]*10)
    output = tmp_path/"portrait.mp4"
    render(project, output)
    info = probe(output)
    assert (info.width, info.height) == (180, 320)
    assert info.rotation == 0
    assert frame_timing(info, threading.Event())[0] == 18
    assert info.audio_count == 1
    assert fingerprint(clip) == before
