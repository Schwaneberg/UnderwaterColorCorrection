import os
import subprocess

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def clip(tmp_path):
    source = tmp_path / "blue water ü test.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-filter_threads", "1", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=12:duration=1.5", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1.5", "-vf", "colorchannelmixer=rr=0.45:gg=0.85:bb=1,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                    "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                    "-shortest", str(source)], check=True)
    return source
