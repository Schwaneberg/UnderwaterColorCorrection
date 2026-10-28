import os
import threading
from dataclasses import replace

import pytest

from uwcc import codecs
from uwcc.errors import Cancelled, UWCCError
from uwcc.media import FrameReader, frame_timestamps, frame_timing, probe
from uwcc.project import Grade, Project
from uwcc.render import render


def test_encoder_auto_fallback_and_explicit_failure(clip, monkeypatch):
    info = probe(clip)
    monkeypatch.setattr(codecs, "encoder_probe", lambda *_: (False, "driver unavailable"))
    args, label = codecs.encoder("h264", ["-c:v", "libx264"], info, "auto", None)
    assert args == ["-c:v", "libx264"] and "CPU" in label
    with pytest.raises(UWCCError, match="driver unavailable"):
        codecs.encoder("h264", [], info, "nvenc", None)
    monkeypatch.setattr(codecs, "encoder_probe", lambda *_: (True, ""))
    args, label = codecs.encoder("hevc", [], info, "auto", None)
    assert "hevc_nvenc" in args and "p010le" in args and "NVENC" in label
    assert codecs.encoder("prores", ["software"], info, "auto", None)[0] == ["software"]
    with pytest.raises(UWCCError, match="H.264 and HEVC"):
        codecs.encoder("prores", [], info, "nvenc", None)


def test_decoder_layout_and_unsupported_source_fallback(clip, monkeypatch):
    monkeypatch.setattr(codecs, "_cached", lambda *_: (True, ""))
    info = probe(clip)
    options, filters, label = codecs.decoder(info, "cuda", None)
    assert options == ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    assert filters == ["hwdownload", "format=nv12", "format=yuv420p"] and "NVDEC" in label
    assert codecs.decoder(replace(info, pixel_format="yuv420p10le"), "cuda", None)[1][-2:] == ["format=p010le", "format=yuv420p10le"]
    unsupported = replace(info, pixel_format="yuv422p10le")
    assert codecs.decoder(unsupported, "auto", None)[0] == []
    with pytest.raises(UWCCError, match="Unsupported NVIDIA"):
        codecs.decoder(unsupported, "cuda", None)
    monkeypatch.setattr(codecs, "_cached", lambda *_: (False, "unsupported GPU profile"))
    assert codecs.decoder(info, "auto", None)[0] == []
    with pytest.raises(UWCCError, match="unsupported GPU profile"):
        codecs.decoder(info, "cuda", None)


def test_cached_codec_check_obeys_cancellation(monkeypatch):
    cancel = threading.Event()
    calls = []
    monkeypatch.setattr(codecs, "_checks", {})
    monkeypatch.setattr(codecs, "_check", lambda *_: calls.append(1) or (True, ""))
    assert codecs._cached("test", [], cancel)[0]
    assert codecs._cached("test", [], cancel)[0]
    assert len(calls) == 1
    cancel.set()
    with pytest.raises(Cancelled):
        codecs._cached("test", [], cancel)


def test_backend_preferences_round_trip_and_old_defaults():
    from uwcc import __version__
    assert Project("unused").to_dict()["application_version"] == __version__
    assert Grade.from_dict({}).video_encoder == "cpu"
    assert Grade.from_dict({}).video_decoder == "cpu"
    grade = Grade(video_encoder="nvenc", video_decoder="cuda")
    restored = Grade.from_dict(grade.to_dict())
    assert (restored.video_encoder, restored.video_decoder) == ("nvenc", "cuda")
    with pytest.raises(UWCCError):
        Grade.from_dict({"video_encoder": "invalid"})


def test_default_encoder_never_probes_nvidia(monkeypatch):
    def unexpected_probe(*_):
        pytest.fail("CPU default must not initialize NVIDIA video encoding")
    monkeypatch.setattr(codecs, "encoder_probe", unexpected_probe)
    for grade in (Grade(), Grade.from_dict({})):
        args, label = codecs.encoder("h264", ["-c:v", "libx264"], None, grade.video_encoder, None)
        assert args == ["-c:v", "libx264"]
        assert label == "CPU encoding"


gpu = pytest.mark.skipif(os.environ.get("UWCC_TEST_VIDEO_GPU") != "1",
                         reason="Set UWCC_TEST_VIDEO_GPU=1 on a working NVIDIA host")


@gpu
@pytest.mark.parametrize("profile,output", [("h264", "rec709"), ("hevc", "hlg"), ("hevc", "pq")])
def test_real_nvidia_export_preserves_geometry_audio_and_color(clip, tmp_path, profile, output):
    grade = Grade(input_profile="rec709", output_profile=output, rotation="90",
                  video_encoder="nvenc", video_decoder="cuda", stabilize=True, denoise=2)
    project = Project(str(clip), grade)
    target = tmp_path / "nvidia.mp4"
    messages = []
    render(project, target, profile, progress=lambda _, message: messages.append(message))
    info = probe(target)
    assert (info.width, info.height, info.audio_count) == (180, 320, 1)
    assert frame_timing(info)[0] == 18
    assert info.transfer == {"rec709": "bt709", "hlg": "arib-std-b67", "pq": "smpte2084"}[output]
    assert any("NVIDIA NVENC encoding" in message for message in messages)
    assert any("NVIDIA NVDEC decoding" in message for message in messages)


@gpu
def test_real_nvidia_variable_frame_timing(tmp_path):
    import subprocess
    source = tmp_path / "variable.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=2",
                    "-vf", "select='if(lt(t,1),1,not(mod(n,2)))',setsar=4/3",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "2", str(source)], check=True)
    grade = Grade(input_profile="rec709", rotation="0", video_encoder="nvenc", video_decoder="cuda")
    target = tmp_path / "variable-out.mp4"
    render(Project(str(source), grade), target)
    assert list(frame_timestamps(probe(target))) == pytest.approx(list(frame_timestamps(probe(source))), abs=.0001)
    assert probe(target).sar == "4:3"


@gpu
def test_real_cuda_decoder_frame_count(clip):
    import numpy as np
    with FrameReader(probe(clip), decoder="cpu") as reader:
        expected = list(reader)
    with FrameReader(probe(clip), decoder="cuda") as reader:
        actual = list(reader)
        assert len(actual) == 18
        assert reader.decode_backend == "NVIDIA NVDEC decoding"
    np.testing.assert_allclose(actual, expected, atol=2/65535, rtol=0)


@gpu
def test_real_cuda_10bit_decoder_pixel_parity(tmp_path):
    import numpy as np
    import subprocess
    source = tmp_path / "ten-bit.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-filter_threads", "1", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=12:duration=0.5", "-pix_fmt", "yuv420p10le",
                    "-c:v", "libx265", "-x265-params", "pools=2:frame-threads=1:log-level=error",
                    "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
                    str(source)], check=True)
    info = probe(source)
    assert info.pixel_format == "yuv420p10le"
    with FrameReader(info, decoder="cpu") as reader:
        expected = list(reader)
    with FrameReader(info, decoder="cuda") as reader:
        actual = list(reader)
    np.testing.assert_allclose(actual, expected, atol=2/65535, rtol=0)
