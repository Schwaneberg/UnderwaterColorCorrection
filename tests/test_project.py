from fractions import Fraction

import pytest

from uwcc.errors import UWCCError
from uwcc.project import Grade, Project, fingerprint, load_preset, read_json, save_preset


def test_keyframes_interpolate_and_hold():
    grade = Grade()
    grade.set_key("exposure", Fraction(1, 3), -1)
    grade.set_key("exposure", Fraction(4, 3), 1)
    assert grade.at(0)["exposure"] == -1
    assert grade.at(Fraction(5, 6))["exposure"] == pytest.approx(0)
    assert grade.at(3)["exposure"] == 1
    grade.set_key("exposure", Fraction(1, 3), -.5)
    assert len(grade.keyframes["exposure"]) == 2
    assert Grade.from_dict(grade.to_dict()).to_dict() == grade.to_dict()


def test_project_and_preset_round_trip(tmp_path):
    source = tmp_path / "video.bin"
    source.write_bytes(b"video")
    project = Project(str(source), Grade(input_profile="rec709"), fingerprint(source))
    project.grade.set_key("saturation", 1, 120)
    target = tmp_path / "project.json"
    project.save(target)
    assert Project.load(target).grade.to_dict() == project.grade.to_dict()
    preset = tmp_path / "preset.json"
    save_preset(preset, project.grade)
    assert not load_preset(preset).keyframes
    source.write_bytes(b"changed")
    with pytest.raises(UWCCError, match="changed"):
        Project.load(target)
    assert Project.load(target, replacement=source).analysis is None


@pytest.mark.parametrize("data", [{"values": {"exposure": 9}}, {"values": {"hue": float("nan")}},
                                  {"input_profile": "guess"}, {"keyframes": {"bad": []}}])
def test_invalid_grade_rejected(data):
    with pytest.raises(UWCCError):
        Grade.from_dict(data)


def test_duplicate_json_key_rejected(tmp_path):
    target = tmp_path / "bad.json"
    target.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(UWCCError, match="Duplicate"):
        read_json(target)


def test_project_cannot_overwrite_source(clip):
    from uwcc.project import Project
    project = Project(str(clip))
    before = clip.read_bytes()
    with pytest.raises(UWCCError, match="source video"):
        project.save(clip)
    assert clip.read_bytes() == before
