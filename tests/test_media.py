import threading
import subprocess

import pytest

from uwcc.analysis import analyze
from uwcc.errors import Cancelled, UWCCError
from uwcc.media import frame_timestamps, frame_timing, probe, read_frame
from uwcc.project import Grade, Project, fingerprint
from uwcc.render import render


def test_probe_and_seek(clip):
    info = probe(clip)
    assert (info.width, info.height, info.profile, info.audio_count) == (320, 180, "rec709", 1)
    assert frame_timing(info) == (18, False)
    assert read_frame(info, .8, 160).shape == (90, 160, 3)


@pytest.mark.parametrize("profile,extension", [("h264", ".mp4"), ("ffv1", ".mkv"), ("prores", ".mov"), ("dnxhr", ".mov"), ("hevc", ".mp4")])
def test_export_audio_frames_metadata(clip, tmp_path, profile, extension):
    info = probe(clip)
    project = Project(str(clip), Grade(input_profile="rec709"), fingerprint(clip))
    project.analysis = analyze(info, project.grade)
    project.grade.set_key("exposure", 0, 0)
    project.grade.set_key("exposure", 1, .2)
    target = tmp_path/(profile+extension)
    render(project, target, profile)
    result = probe(target)
    assert (result.width, result.height, result.audio_count, result.transfer) == (320, 180, 1, "bt709")
    assert frame_timing(result)[0] == 18
    assert abs(result.duration-info.duration) < 1/12
    with pytest.raises(UWCCError, match="exists"):
        render(project, target, profile)


def test_cancel_leaves_no_output(clip, tmp_path):
    cancel = threading.Event()
    cancel.set()
    project = Project(str(clip), Grade(input_profile="rec709"), fingerprint(clip))
    target = tmp_path/"cancel.mp4"
    with pytest.raises(Cancelled):
        render(project, target, cancel=cancel)
    assert not target.exists()


def test_source_cannot_be_overwritten(clip):
    project = Project(str(clip), Grade(input_profile="rec709"), fingerprint(clip))
    with pytest.raises(UWCCError, match="source"):
        render(project, clip, overwrite=True)


@pytest.mark.parametrize("finishing", [False, True])
def test_variable_frame_rate_keeps_timestamps(tmp_path, finishing):
    source = tmp_path/"variable.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=2",
                    "-vf", "select='if(lt(t,1),1,not(mod(n,2)))',setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "2", str(source)], check=True)
    info = probe(source)
    assert frame_timing(info)[1]
    project = Project(str(source), Grade(input_profile="rec709", stabilize=finishing,
                                         denoise=3 if finishing else 0), fingerprint(source))
    target = tmp_path/"vfr result.mp4"
    render(project, target)
    source_times = list(frame_timestamps(info))
    result_times = list(frame_timestamps(probe(target)))
    assert len(source_times) == len(result_times)
    assert result_times == pytest.approx(source_times, abs=.0001)


def test_audio_offset_preserved(tmp_path):
    import json
    source = tmp_path/"offset.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=2",
                    "-itsoffset", "0.4", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.4",
                    "-c:v", "libx264", "-threads", "2", "-c:a", "aac", str(source)], check=True)
    project = Project(str(source), Grade(input_profile="rec709"), fingerprint(source))
    target = tmp_path/"offset-out.mp4"
    render(project, target)
    def offset(path):
        raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time", "-of", "json", str(path)])
        return float(json.loads(raw)["streams"][0]["start_time"])
    assert abs(offset(source)-offset(target)) < .02


@pytest.mark.parametrize("output,transfer", [("hlg", "arib-std-b67"), ("pq", "smpte2084")])
def test_hdr_signaling(clip, tmp_path, output, transfer):
    project = Project(str(clip), Grade(input_profile="rec709", output_profile=output), fingerprint(clip))
    target = tmp_path/(output+".mp4")
    render(project, target, "hevc")
    result = probe(target)
    assert result.transfer == transfer
    assert result.primaries == "bt2020"
