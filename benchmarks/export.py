"""Run from the repository root: python3 -m benchmarks.export [--clip video]."""
import argparse
from contextlib import nullcontext
import json
import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from uwcc import color
from uwcc.analysis import analyze
from uwcc.media import frame_timing, probe
from uwcc.project import Grade, Project, clone, fingerprint
from uwcc.render import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--project", help="Also compare full H.264 exports with a saved project")
    source.add_argument("--clip", help="Analyze and benchmark an unedited camera clip")
    parser.add_argument("--input-profile", choices=["rec709", "slog2", "slog3", "hlg", "pq"],
                        help="Input interpretation for --clip; required if metadata is ambiguous")
    parser.add_argument("--finishing", action="store_true",
                        help="Also measure stabilization, denoise (strength 3), and both using C")
    parser.add_argument("--backend", choices=["both", "native", "numpy"], default="both",
                        help="Backends for full exports; defaults to both")
    parser.add_argument("--output-dir", type=Path, help="Keep exports here for visual comparison")
    parser.add_argument("--report", type=Path, help="Write machine-readable export results as JSON")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    project = None
    analysis_seconds = None
    if args.clip:
        if not Path(args.clip).is_file():
            parser.error(f"Camera clip is not available: {args.clip}")
        info = probe(args.clip)
        profile = args.input_profile or info.profile
        if profile == "auto":
            parser.error("Clip metadata is ambiguous; specify --input-profile explicitly")
        project = Project(info.path, Grade(input_profile=profile), fingerprint(Path(info.path)))
    elif args.project:
        project = Project.load(args.project)
        info = probe(project.source)
    elif args.finishing or args.output_dir or args.report:
        parser.error("Export options require --clip or --project")
    native = color._native
    if native is None:
        parser.error("Build the C extension first: python3 setup.py build_ext --inplace")
    if args.clip:
        start = perf_counter()
        project.analysis = analyze(info, project.grade)
        analysis_seconds = perf_counter() - start
        print(f"Analysis: {analysis_seconds:.2f}s", flush=True)
    rgb = np.random.default_rng(1).random((1080, 1920, 3), dtype=np.float32)
    lut = color.identity_lut()
    grade = Grade(input_profile="rec709")
    operations = {
        "1080p LUT": lambda: color.apply_lut(rgb, lut),
        "1080p neutral correction": lambda: color.correct(rgb, grade),
    }
    try:
        for name, operation in operations.items():
            timings = []
            for backend in (None, native):
                color._native = backend
                operation()  # warm up
                start = perf_counter()
                for _ in range(args.repeats):
                    operation()
                timings.append((perf_counter() - start) / args.repeats)
            print(f"{name}: NumPy {timings[0]:.3f}s; C {timings[1]:.3f}s; {timings[0]/timings[1]:.2f}x", flush=True)
        if project:
            count, variable = frame_timing(info)
            print(f"Source: {info.width}×{info.height}, {count} frames, {info.duration:.3f}s, "
                  f"input {project.grade.input_profile}", flush=True)
            scenarios = [("numpy", None, project.grade.stabilize, project.grade.denoise),
                         ("native", native, project.grade.stabilize, project.grade.denoise)]
            if args.backend != "both":
                scenarios = [row for row in scenarios if row[0] == args.backend]
            if args.finishing:
                scenarios += [("native-stabilized", native, True, 0),
                              ("native-denoised", native, False, 3),
                              ("native-combined", native, True, 3)]
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
            exports = nullcontext(args.output_dir) if args.output_dir else TemporaryDirectory(prefix="uwcc-benchmark-")
            results = []
            with exports as directory:
                for label, backend, stabilize, denoise in scenarios:
                    color._native = backend
                    snapshot = clone(project)
                    snapshot.grade.stabilize, snapshot.grade.denoise = stabilize, denoise
                    # Delivery benchmark always writes SDR H.264, including HDR sources.
                    snapshot.grade.output_profile = "rec709"
                    destination = Path(directory) / f"{label}.mp4"
                    print(f"Starting {label} export…", flush=True)
                    start = perf_counter()
                    render(snapshot, destination)
                    elapsed = perf_counter() - start
                    results.append({"scenario": label, "backend": "numpy" if backend is None else "native",
                                    "stabilize": stabilize, "denoise": denoise, "seconds": elapsed,
                                    "frames_per_second": count / elapsed, "realtime_speed": info.duration / elapsed,
                                    "output_bytes": destination.stat().st_size,
                                    "output": str(destination.resolve()) if args.output_dir else None})
                    print(f"Full export ({label}): {elapsed:.2f}s, {count/elapsed:.2f} fps, "
                          f"{info.duration/elapsed:.2f}× real time", flush=True)
            if args.report:
                report = {"source": info.path, "fingerprint": project.source_fingerprint,
                          "machine": {"platform": platform.platform(), "python": platform.python_version(),
                                      "numpy": np.__version__, "cpu_count": os.cpu_count()},
                          "width": info.width, "height": info.height, "frames": count,
                          "duration": info.duration, "variable_frame_rate": variable,
                          "input_profile": project.grade.input_profile, "format": "h264",
                          "analysis_seconds": analysis_seconds, "exports": results}
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        color._native = native


if __name__ == "__main__":
    main()
