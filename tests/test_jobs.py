import threading
import subprocess

from uwcc.jobs import BatchQueue
from uwcc.project import Grade


def test_batch_independent_analysis_collision_and_restart(clip, tmp_path):
    path = tmp_path/"queue.json"
    queue = BatchQueue(path)
    first = queue.add(clip, tmp_path, Grade(input_profile="rec709"), "h264")
    second = queue.add(clip, tmp_path, Grade(input_profile="rec709"), "h264")
    assert first["destination"] != second["destination"]
    resumed = BatchQueue(path)
    assert all(job["state"] == "Paused" for job in resumed.jobs)
    resumed.process(threading.Event())
    assert all(job["state"] == "Complete" for job in resumed.jobs)
    assert all(job["state"] == "Complete" for job in BatchQueue(path).jobs)


def test_ambiguous_batch_fails_without_guessing(clip, tmp_path):
    ambiguous = tmp_path/"untagged.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
                    "setparams=color_primaries=unknown:color_trc=unknown:colorspace=unknown",
                    "-c:v", "libx264", "-threads", "2", "-an", str(ambiguous)], check=True)
    queue = BatchQueue(tmp_path/"queue.json")
    queue.add(ambiguous, tmp_path, Grade(input_profile="auto"), "h264")
    queue.process(threading.Event())
    assert queue.jobs[0]["state"] == "Failed"
    assert "ambiguous" in queue.jobs[0]["message"]
