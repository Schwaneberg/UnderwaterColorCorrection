"""Desktop launcher and scriptable headless workflow."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .errors import UWCCError


def parser():
    result = argparse.ArgumentParser(prog="uwcc", description="Local underwater video color correction")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command")
    gui = commands.add_parser("gui", help="Open the desktop workspace (default)")
    gui.add_argument("video", nargs="?")
    inspect = commands.add_parser("probe", help="Show video metadata")
    inspect.add_argument("video")
    analysis = commands.add_parser("analyze", help="Analyze a video and save a project")
    analysis.add_argument("video")
    analysis.add_argument("--profile", required=True, choices=["rec709", "slog2", "slog3", "hlg", "pq"])
    analysis.add_argument("--filter", default="auto", choices=["auto", "none", "red", "magenta"])
    analysis.add_argument("--output-profile", default="rec709", choices=["rec709", "hlg", "pq"])
    analysis.add_argument("--model", help="Local verified ONNX model manifest")
    analysis.add_argument("--preset", help="UWCC preset JSON file")
    analysis.add_argument("--project", required=True, help="Output .uwcc-project.json file")
    export = commands.add_parser("render", help="Render a saved project")
    export.add_argument("project")
    export.add_argument("output")
    export.add_argument("--format", default="h264", choices=["h264", "hevc", "dnxhr", "prores", "ffv1"])
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--workers", type=int, help="Parallel color frames (1–16, memory capped)")
    batch = commands.add_parser("batch", help="Analyze and render several videos independently")
    batch.add_argument("videos", nargs="+")
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--profile", required=True, choices=["rec709", "slog2", "slog3", "hlg", "pq"])
    batch.add_argument("--preset")
    batch.add_argument("--format", default="h264", choices=["h264", "hevc", "dnxhr", "prores", "ffv1"])
    batch.add_argument("--queue", help="Persistent queue JSON path")
    relink = commands.add_parser("relink", help="Relink a moved/changed source, discarding stale analysis")
    relink.add_argument("project")
    relink.add_argument("video")
    learn = commands.add_parser("learn-reference", help="Learn a time-dependent grade from a corrected version of the source")
    learn.add_argument("video")
    learn.add_argument("reference")
    learn.add_argument("--profile", required=True, choices=["rec709"])
    learn.add_argument("--project", required=True)
    advice = commands.add_parser("advise", help="Opt in to cloud advice using three proxy frames")
    advice.add_argument("project")
    advice.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    advice.add_argument("--model", required=True, help="Vision-capable API model ID")
    advice.add_argument("--preference", default="Natural, balanced, vivid underwater footage")
    advice.add_argument("--allow-upload", action="store_true", help="Send three JPEG proxy frames to the selected provider")
    advice.add_argument("--output", required=True, help="Save a new project with the proposed grade")
    commands.add_parser("models", help="Show local AI model availability")
    for command in (analysis, batch):
        command.add_argument("--auto-method", choices=["ai", "adaptive", "legacy"], default="ai")
    for command in (analysis, export, batch):
        command.add_argument("--stabilize", action="store_true", default=None,
                             help="Stabilize the export with robust background tracking")
        command.add_argument("--denoise", type=float, default=None, metavar="0–10",
                             help="Export spatial/temporal denoise strength (0 disables)")
        command.add_argument("--ai-denoise", action="store_true", help="Use the local FastDVDnet neural denoiser")
        command.add_argument("--denoise-model", help="Explicit local temporal-denoise ONNX manifest")
        command.add_argument("--stabilization-region", choices=["seabed", "whole"])
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in (None, "gui"):
        try:
            from .gui import launch
            return launch(getattr(args, "video", None))
        except ImportError as exc:
            print(f"GUI dependencies missing: {exc}. Install PySide6 (or PyQt6), NumPy and Pillow.", file=sys.stderr)
            return 2
    from .analysis import analyze
    from .jobs import BatchQueue
    from .media import probe
    from .project import Grade, Project, fingerprint, load_preset
    from .render import render
    cancel = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: cancel.set())

    def progress(fraction, message):
        print(f"{fraction:6.1%}  {message}", file=sys.stderr)

    def finishing(grade):
        if args.stabilize is not None:
            grade.stabilize = args.stabilize
        if args.denoise is not None:
            grade.denoise = args.denoise
        if args.ai_denoise or args.denoise_model:
            from .neural import builtin_model
            grade.denoise_engine = "ai"
            grade.denoise_manifest = args.denoise_model or builtin_model("denoise")
            if args.denoise is None and not grade.denoise:
                grade.denoise = 3
        if args.stabilization_region:
            grade.stabilization_region = args.stabilization_region
        # Apply the same validation as persisted projects and presets.
        Grade.from_dict(grade.to_dict())
        return grade

    try:
        if args.command == "probe":
            data = asdict(probe(args.video, cancel))
            data["fps"] = str(data["fps"])
            print(json.dumps(data, indent=2))
        elif args.command == "analyze":
            info = probe(args.video, cancel)
            grade = load_preset(args.preset) if args.preset else Grade()
            grade.input_profile, grade.output_profile, grade.filter_hint = args.profile, args.output_profile, args.filter
            grade.model_manifest = args.model
            grade.auto_method = args.auto_method
            finishing(grade)
            project = Project(info.path, grade, fingerprint(Path(info.path)))
            project.analysis = analyze(info, grade, cancel, progress)
            project.save(args.project)
            print(f"Saved {args.project}")
            for warning in project.analysis["warnings"]:
                print(f"Warning: {warning}", file=sys.stderr)
        elif args.command == "render":
            project = Project.load(args.project)
            finishing(project.grade)
            render(project, args.output, args.format, cancel, progress, args.overwrite, workers=args.workers)
        elif args.command == "batch":
            grade = load_preset(args.preset) if args.preset else Grade()
            grade.input_profile = args.profile
            grade.auto_method = args.auto_method
            finishing(grade)
            queue = BatchQueue(args.queue or Path(args.output_dir)/"uwcc-queue.json")
            for video in args.videos:
                queue.add(video, args.output_dir, grade, args.format)
            queue.process(cancel, progress)
            if any(job["state"] != "Complete" for job in queue.jobs):
                return 1
        elif args.command == "relink":
            project = Project.load(args.project, replacement=args.video)
            project.save(args.project)
            print("Source relinked. Re-analyze before using automatic correction.")
        elif args.command == "learn-reference":
            from .analysis import learn_reference
            if Path(args.project).resolve() == Path(args.reference).resolve():
                raise UWCCError("A project cannot overwrite the corrected reference video.")
            info = probe(args.video, cancel)
            project = Project(info.path, Grade(input_profile=args.profile), fingerprint(Path(info.path)))
            project.analysis = learn_reference(info, project.grade, args.reference, cancel, progress)
            project.save(args.project)
            print(f"Learned reference grade; mean RGB error {project.analysis['reference_mae']:.4f}. Saved {args.project}")
        elif args.command == "advise":
            from .cloud import advise_project, apply_advice
            if not args.allow_upload:
                raise UWCCError("Cloud advice sends three proxy frames; pass --allow-upload to opt in.")
            project = Project.load(args.project)
            proposal = advise_project(project, args.provider, args.model, args.preference, cancel, progress)
            result = apply_advice(project, proposal)
            result.save(args.output)
            print(proposal["summary"])
        elif args.command == "models":
            import importlib.util
            from .neural import builtin_model
            from .project import model_directory
            print("Model directory:", model_directory())
            for task in ("enhance", "denoise"):
                manifest = builtin_model(task)
                print(f"{task}: {manifest or 'not installed'}")
            print("ONNX Runtime:", bool(importlib.util.find_spec("onnxruntime")))
            if importlib.util.find_spec("onnxruntime"):
                import onnxruntime as ort
                print("ONNX providers:", ", ".join(ort.get_available_providers()))
            print("OpenCV:", bool(importlib.util.find_spec("cv2")))
        return 130 if cancel.is_set() else 0
    except (UWCCError, OSError, ValueError, KeyError) as exc:
        print(f"UWCC: {exc}", file=sys.stderr)
        return 130 if cancel.is_set() else 1
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    raise SystemExit(main())
