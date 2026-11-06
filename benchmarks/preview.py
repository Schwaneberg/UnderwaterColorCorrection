"""Measure 640px preview CPU stages: python -m benchmarks.preview --repeats 20.

Synthetic input isolates correction costs from video decoding and GUI painting.
The legacy comparison includes the former redundant original-view transform.
"""
import argparse
import json
from contextlib import closing
from time import perf_counter

import numpy as np

from uwcc import color
from uwcc.gui import Playback, preview_original, qimage
from uwcc.media import FrameReader, frame_timestamps, prepare_frame, probe
from uwcc.project import Grade, Project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--clip", help="Also measure Rec.709 playback, decode, timestamps and image conversion")
    parser.add_argument("--seconds", type=float, help="Limit clip measurements to this many seconds")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip slow legacy clip comparison")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    raw = np.random.default_rng(7).random((360, 640, 3), dtype=np.float32)
    grade = Grade(input_profile="rec709")
    analysis = {"samples": [{"time": 0, "scene": 0, "recipe": {
        "gain": [1.5, 1, .8], "black": .01, "white": .8,
        "gamma": .9, "vibrance": .1, "contrast": 1.06}}]}
    manual = Grade(input_profile="rec709")
    manual.values["temperature"] = 15
    cases = {
        "legacy_original": lambda: color.encode(color.decode(raw, "rec709"), "rec709"),
        "original": lambda: preview_original(raw, "rec709"),
        "analyzed_correction": lambda: color.correct(raw, grade, 0, analysis, True),
        "manual_correction": lambda: color.correct(raw, manual, 0, analysis, True),
        "qimage": lambda: qimage(raw),
    }
    native = color._native
    report = {}
    try:
        for name, backend in (("native", native), ("numpy", None)):
            if name == "native" and backend is None:
                continue
            color._native = backend
            timings = {}
            for stage, operation in cases.items():
                operation()
                start = perf_counter()
                for _ in range(args.repeats):
                    operation()
                timings[stage] = (perf_counter()-start)*1000/args.repeats
            report[name] = timings
    finally:
        color._native = native
    result = {"resolution": [640, 360], "milliseconds": report}
    if args.clip:
        info = probe(args.clip)
        limit = min(info.duration, args.seconds) if args.seconds else info.duration
        result["clip"] = {"width": info.width, "height": info.height,
                          "fps": float(info.fps), "seconds": limit}
        project = Project(info.path, grade, analysis=analysis)
        start = perf_counter()
        decoded = 0
        with FrameReader(info, max_width=640) as reader:
            for frame in reader:
                decoded += 1
                if decoded >= round(limit*float(info.fps)):
                    break
        result["decode_only"] = {"frames": decoded, "seconds": perf_counter()-start}
        start = perf_counter()
        frames = 0
        # Previous playback processed every frame, even when its deadline passed.
        with closing(frame_timestamps(info, start=0)) as times, FrameReader(info, max_width=640) as reader:
            for frame in reader:
                if args.skip_legacy:
                    break
                timestamp = next(times)
                frame = prepare_frame(frame, info, "rec709")
                original = color.encode(color.decode(frame, "rec709"), "rec709")
                rgb = color.correct(frame, grade, timestamp, analysis, True)
                qimage(original)
                qimage(rgb)
                frames += 1
                if timestamp >= limit-1/float(info.fps):
                    break
        if not args.skip_legacy:
            result["legacy_unpaced"] = {"frames": frames, "seconds": perf_counter()-start}
        worker = Playback(info, project, 0)
        shown = []
        errors = []

        def display(original, corrected, timestamp):
            qimage(original)
            qimage(corrected)
            shown.append(timestamp)
            if timestamp >= limit-1/float(info.fps):
                worker.cancel.set()

        worker.frame.connect(display)
        worker.failed.connect(errors.append)
        start = perf_counter()
        worker.run()
        result["playback"] = {"frames": len(shown), "seconds": perf_counter()-start,
                              "last_timestamp": shown[-1] if shown else None, "errors": errors}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
