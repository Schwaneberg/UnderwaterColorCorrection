"""Compare complete short exports using CPU codecs, NVENC, and NVDEC."""

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

import numpy as np

from uwcc.media import FrameReader, frame_timing, probe
from uwcc.project import Grade, Project
from uwcc.render import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("Use positive seconds and at least one repeat")
    report = {"clip": str(args.clip), "repeats": args.repeats, "exports": {}}
    with tempfile.TemporaryDirectory(prefix="uwcc-codec-benchmark-") as directory:
        directory = Path(directory)
        source = directory / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.clip), "-t", str(args.seconds),
                        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-crf", "12",
                        "-preset", "fast", "-threads", "2", "-c:a", "aac", str(source)], check=True)
        info = probe(source)
        if info.profile != "rec709":
            parser.error("This benchmark requires a declared Rec.709-compatible clip")
        report.update(width=info.width, height=info.height, fps=str(info.fps), frames=frame_timing(info)[0])
        outputs = {}
        for label, encoder, decoder in (("cpu", "cpu", "cpu"), ("nvenc", "nvenc", "cpu"),
                                         ("nvenc_nvdec", "nvenc", "cuda")):
            grade = Grade(input_profile="rec709", rotation="0", video_encoder=encoder, video_decoder=decoder)
            durations = []
            for repeat in range(args.repeats + 1):
                target = directory / f"{label}-{repeat}.mp4"
                start = time.perf_counter()
                render(Project(str(source), grade), target)
                elapsed = time.perf_counter() - start
                # Warm up hardware checks and driver initialization once per mode.
                if repeat:
                    durations.append(elapsed)
            outputs[label] = target
            report["exports"][label] = {"seconds": durations, "median_seconds": statistics.median(durations),
                                         "bytes": target.stat().st_size}
            print(label, report["exports"][label], flush=True)
        for label in ("nvenc", "nvenc_nvdec"):
            result = subprocess.run(["ffmpeg", "-v", "info", "-filter_complex_threads", "1",
                                     "-i", str(outputs["cpu"]), "-i", str(outputs[label]),
                                     "-lavfi", "ssim", "-f", "null", "-"], capture_output=True, text=True, check=True)
            match = re.search(r"All:([0-9.]+)", result.stderr)
            if match:
                report["exports"][label]["ssim_against_cpu_export"] = float(match[1])
        frames = {}
        for decoder in ("cpu", "cuda"):
            with FrameReader(info, single=True, decoder=decoder) as reader:
                frames[decoder] = next(iter(reader)).copy()
        delta = np.abs(frames["cpu"] - frames["cuda"])
        report["first_frame_decoder_rgb_difference"] = {"mean": float(delta.mean()), "max": float(delta.max())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
