import struct

import pytest

from uwcc.__main__ import main, parser
from uwcc.profiles import capture_metadata, detect_profile


def box(kind, content):
    return struct.pack(">I4s", len(content) + 8, kind) + content


def sony_xml(gamma, gamut):
    return (f'<NonRealTimeMeta xmlns="urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.20">'
            f'<AcquisitionRecord><Group name="CameraUnitMetadataSet">'
            f'<Item name="CaptureGammaEquation" value="{gamma}"/>'
            f'<Item name="CaptureColorPrimaries" value="{gamut}"/>'
            '</Group></AcquisitionRecord></NonRealTimeMeta>').encode()


@pytest.mark.parametrize("transfer,primaries,expected", [
    ("bt709", "bt709", "rec709"), ("iec61966-2-4", "bt709", "rec709"),
    ("arib-std-b67", "bt2020", "hlg"), ("smpte2084", "bt2020", "pq"),
    ("arib-std-b67", "bt709", "auto"), ("smpte2084", "unknown", "auto"),
    ("unknown", "bt709", "auto"), ("bt709", "bt2020", "auto"),
])
def test_declared_stream_pairs(tmp_path, transfer, primaries, expected):
    path = tmp_path / "video.mp4"
    path.write_bytes(box(b"ftyp", b"isom"))
    assert detect_profile(path, transfer, primaries)[0] == expected


@pytest.mark.parametrize("gamma,gamut,expected", [
    ("s-log2", "s-gamut", "slog2"),
    ("s-log3-cine", "s-gamut3-cine", "slog3"),
    ("s-log3", "s-gamut3-cine", "slog3"),
    ("s-log3", "s-gamut3", "auto"),
    ("s-cinetone", "rec709", "auto"),
    ("rec2100-hlg", "rec2020", "hlg"),
    ("rec2100-pq", "rec2020", "pq"),
    ("rec709-xvycc", "rec709", "rec709"),
    ("s-log2", "", "auto"),
])
def test_capture_declarations_override_generic_tags(tmp_path, gamma, gamut, expected):
    path = tmp_path / "video.mp4"
    path.write_bytes(box(b"meta", bytes(4) + box(b"xml ", bytes(4) + sony_xml(gamma, gamut))))
    profile, evidence = detect_profile(path, "bt709", "bt709")
    assert profile == expected
    assert "Sony capture metadata" in evidence


def test_reader_seeks_over_large_video_and_supports_extended_box(tmp_path):
    path = tmp_path / "large.mp4"
    with path.open("wb") as source:
        source.write(struct.pack(">I4sQ", 1, b"mdat", 1 << 32))
        source.seek(1 << 32)
        content = sony_xml("s-log2", "s-gamut")
        source.write(box(b"moov", box(b"udta", box(b"nrtm", content))))
    assert capture_metadata(path) == ("s-log2", "s-gamut")


def test_conflicting_and_malformed_capture_metadata_abstain(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(box(b"xml ", sony_xml("s-log2", "s-gamut")) +
                     box(b"xml ", sony_xml("s-log3", "s-gamut3-cine")))
    assert detect_profile(path, "bt709", "bt709")[0] == "auto"
    path.write_bytes(box(b"xml ", b"<NonRealTimeMeta><broken>"))
    assert detect_profile(path, "bt709", "bt709")[0] == "auto"
    path.write_bytes(box(b"xml ", b"x" * (2 * 1024 * 1024 + 1)))
    assert detect_profile(path, "bt709", "bt709")[0] == "auto"


def test_cli_defaults_and_automatic_analysis(clip, tmp_path):
    from uwcc.project import Project
    assert parser().parse_args(["batch", str(clip), "--output-dir", str(tmp_path)]).profile == "auto"
    output = tmp_path / "project.json"
    assert main(["analyze", str(clip), "--auto-method", "adaptive", "--rotation", "0",
                 "--project", str(output)]) == 0
    assert Project.load(output).grade.input_profile == "rec709"


def test_cli_unknown_profile_requires_override(clip, tmp_path, monkeypatch, capsys):
    from dataclasses import replace
    from uwcc.media import probe
    info = replace(probe(clip), profile="auto", profile_evidence="Unknown transfer.")
    monkeypatch.setattr("uwcc.media.probe", lambda *_: info)
    output = tmp_path / "project.json"
    assert main(["analyze", str(clip), "--project", str(output)]) == 1
    assert "Pass --profile" in capsys.readouterr().err
    assert not output.exists()


def test_mixed_batch_resolves_each_profile_and_abstains_independently(clip, tmp_path, monkeypatch):
    import threading
    from dataclasses import replace
    from uwcc.jobs import BatchQueue
    from uwcc.media import probe
    from uwcc.project import Grade
    base = probe(clip)
    profiles = iter(("rec709", "hlg", "auto"))
    monkeypatch.setattr("uwcc.jobs.probe", lambda *_: replace(base, profile=next(profiles)))
    seen = []
    monkeypatch.setattr("uwcc.jobs.analyze", lambda info, grade, *_: seen.append(grade.input_profile) or {})
    monkeypatch.setattr("uwcc.jobs.render", lambda *_: None)
    queue = BatchQueue(tmp_path / "queue.json")
    for _ in range(3):
        queue.add(clip, tmp_path, Grade(input_profile="auto", rotation="0"), "h264")
    queue.process(threading.Event())
    assert seen == ["rec709", "hlg"]
    assert [job["state"] for job in queue.jobs] == ["Complete", "Complete", "Failed"]
