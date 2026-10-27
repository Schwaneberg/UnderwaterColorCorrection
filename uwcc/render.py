"""Cancelable full-resolution exports with audio muxing and atomic publication."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .color import correct
from .errors import Cancelled, UWCCError
from .media import FrameReader, executable, frame_timestamps, frame_timing, prepare_frame, probe, run, NO_WINDOW, cpu_threads
from .project import fingerprint

RENDER_PROFILES = {
    "h264": ("H.264 SDR delivery", ".mp4", ["-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p"], "aac"),
    "hevc": ("HEVC 10-bit delivery (SDR/HDR)", ".mp4", ["-c:v", "libx265", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"], "aac"),
    "dnxhr": ("DNxHR HQX editing master", ".mov", ["-c:v", "dnxhd", "-profile:v", "dnxhr_hqx", "-pix_fmt", "yuv422p10le"], "pcm_s24le"),
    "prores": ("ProRes 422 HQ editing master", ".mov", ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"], "pcm_s24le"),
    "ffv1": ("FFV1 lossless archival master", ".mkv", ["-c:v", "ffv1", "-level", "3", "-slicecrc", "1", "-pix_fmt", "gbrp16le"], "flac"),
}


def color_workers(info, requested=None):
    try:
        requested = int(os.environ.get("UWCC_RENDER_WORKERS", min(4, cpu_threads()))) if requested is None else int(requested)
        budget = int(os.environ.get("UWCC_RENDER_MEMORY_MB", "768"))*1024*1024
    except (TypeError, ValueError) as exc:
        raise UWCCError("Render workers and memory budget must be positive integers.") from exc
    if not 1 <= requested <= 16 or budget < 64*1024*1024:
        raise UWCCError("Use 1–16 color workers and a memory budget of at least 64 MiB.")
    # Account for input, intermediate float arrays and the bounded result queue.
    per_worker = info.width*info.height*3*4*8
    return max(1, min(requested, cpu_threads(), budget//max(1, per_worker)))


def corrected_frames(reader, info, project, times, cancel, workers):
    """Sequential temporal processing, parallel independent color, ordered output."""
    def process(index, frame):
        if cancel.is_set():
            raise Cancelled("Export canceled.")
        result = correct(prepare_frame(frame, info, project.grade.input_profile), project.grade, times[index], project.analysis)
        return np.rint(result*65535).astype("<u2")
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="uwcc-color") as pool:
        try:
            for index, frame in enumerate(reader):
                if cancel.is_set():
                    raise Cancelled("Export canceled.")
                if index >= len(times):
                    raise UWCCError("Decoder produced more frames than the timestamp stream.")
                pending.append((index, pool.submit(process, index, frame)))
                if len(pending) >= workers*2 or pending[0][1].done():
                    position, future = pending.popleft()
                    yield position, future.result()
            while pending:
                if cancel.is_set():
                    raise Cancelled("Export canceled.")
                position, future = pending.popleft()
                yield position, future.result()
        finally:
            for _, future in pending:
                future.cancel()


def render(project, destination, profile="h264", cancel=None, progress=lambda *_: None, overwrite=False, workers=None):
    cancel = cancel or threading.Event()
    # Keep native encoder pools bounded even on large workstations.
    threads = cpu_threads()
    destination = Path(destination).resolve()
    source = Path(project.source).resolve()
    if destination == source:
        raise UWCCError("Choose a different output: the source video cannot be overwritten.")
    if destination.exists() and not overwrite:
        raise UWCCError("Output already exists. Choose a different name or explicitly overwrite it.")
    if project.grade.input_profile == "auto":
        raise UWCCError("Confirm the input color profile before exporting.")
    if (project.analysis and project.analysis.get("engine") == "Learned corrected-reference grade"
            and project.grade.output_profile != "rec709"):
        raise UWCCError("The learned reference grade is SDR Rec.709. Select SDR output or re-analyze for HDR.")
    if profile not in RENDER_PROFILES:
        raise UWCCError("Unknown export profile.")
    if project.grade.output_profile != "rec709" and profile not in ("hevc", "ffv1"):
        raise UWCCError("Select HEVC or FFV1 for HDR export, or change the output intent to SDR.")
    if project.source_fingerprint and fingerprint(source) != project.source_fingerprint:
        raise UWCCError("Source changed after opening. Reopen and analyze it before exporting.")
    info = probe(source, cancel)
    from .orientation import project_info
    info = project_info(project, info, cancel, progress)
    workers = color_workers(info, workers)
    progress(0, "Validating frame timing…")
    times = list(frame_timestamps(info, cancel))
    expected = len(times)
    if not expected:
        raise UWCCError("Source has no presentation timestamps.")
    variable = any(abs(b-a-1/float(info.fps)) > .00015 for a, b in zip(times, times[1:]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _, extension, codec_args, audio_codec = RENDER_PROFILES[profile]
    if destination.suffix.lower() != extension:
        raise UWCCError(f"This export profile requires a {extension} output file.")
    if info.width % 2 or info.height % 2:
        raise UWCCError("This encoder requires even video dimensions.")
    if profile == "dnxhr" and (info.width < 256 or info.height < 120):
        raise UWCCError("DNxHR requires at least 256×120 video dimensions.")
    output = project.grade.output_profile
    primaries, transfer, matrix = ("bt709", "bt709", "bt709") if output == "rec709" else (
        "bt2020", "arib-std-b67" if output == "hlg" else "smpte2084", "bt2020nc")
    sar = info.sar.replace(":", "/") if info.sar not in ("N/A", "0:1") else "1/1"
    stabilization_plan = None
    render_start = 0.
    if project.grade.stabilize:
        from .finishing import build_stabilization_plan
        stabilization_plan = build_stabilization_plan(info, project.grade, cancel,
                                                      lambda fraction, message: progress(fraction*.08, message))
        if len(stabilization_plan.matrices) != expected:
            raise UWCCError("Stabilization analysis and timestamp frame counts disagree.")
        render_start = .08
    with tempfile.TemporaryDirectory(prefix=".uwcc-render-", dir=destination.parent) as directory:
        directory = Path(directory)
        video = directory / ("video"+extension)
        final = directory / ("result"+extension)
        # Limit thread use: many native encoders otherwise allocate one pool per host CPU.
        args = [executable("ffmpeg"), "-nostdin", "-v", "error", "-filter_threads", str(min(4, threads)), "-f", "rawvideo", "-pixel_format", "rgb48le",
                "-video_size", f"{info.width}x{info.height}", "-framerate", str(info.fps), "-i", "pipe:0",
                "-vf", f"setsar={sar}", *codec_args, "-threads", str(threads),
                "-color_primaries", primaries, "-color_trc", transfer, "-colorspace", matrix,
                "-color_range", "pc" if profile == "ffv1" else "tv"]
        # Explicit RGB->YUV matrix selection; tags alone do not transform pixels.
        if profile != "ffv1":
            vf_index = args.index("-vf")+1
            scale_matrix = "bt2020" if matrix == "bt2020nc" else matrix
            args[vf_index] += f",scale=in_range=full:out_range=limited:out_color_matrix={scale_matrix}"
        # FFmpeg 8 can take transfer/primaries from AVFrame properties instead of
        # encoder options. Set both, otherwise a well-encoded file loses its tags.
        args[args.index("-vf")+1] += f",setparams=color_primaries={primaries}:color_trc={transfer}:colorspace={matrix}:range={'full' if profile == 'ffv1' else 'limited'}"
        if variable:
            # sendcmd is evaluated on raw-pipe cadence BEFORE setpts. Each frame
            # then receives its original presentation timestamp at microsecond
            # time base. A script avoids OS argument-length limits on long clips.
            commands = directory / "timestamps.cmd"
            commands.write_text("\n".join(
                f"{max(0, (i-.25)/float(info.fps)):.9f} setpts expr {timestamp:.9f}/TB;"
                for i, timestamp in enumerate(times)), encoding="utf-8")
            escaped = str(commands).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
            filter_index = args.index("-vf")
            graph = args[filter_index+1] + f",settb=AVTB,sendcmd=f='{escaped}',setpts=PTS"
            script = directory / "render.ffgraph"
            script.write_text(graph, encoding="utf-8")
            args[filter_index:filter_index+2] = ["-filter_script:v", str(script)]
            args += ["-enc_time_base", "1:1000000", "-fps_mode", "vfr"]
        if profile == "hevc":
            args += ["-x265-params", f"pools={threads}:frame-threads={min(threads, 3)}"]
        args += [str(video)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors, **NO_WINDOW)
            stop_watch = threading.Event()

            def watch():
                while not stop_watch.wait(.1):
                    if cancel.is_set() and process.poll() is None:
                        process.kill()
                        return

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            frames = 0
            last_frame_report = -float("inf")
            def finishing_progress(message):
                fraction = min(.93, render_start+frames/max(expected, 1)*(.93-render_start))
                progress(fraction, f"{message} · output {frames}/{expected}")
            try:
                progress(render_start, f"Rendering with {workers} color workers; up to {threads} codec threads")
                with FrameReader(info, cancel=cancel, finishing=project.grade, stabilization_plan=stabilization_plan,
                                 progress=finishing_progress) as reader:
                    for index, raw in corrected_frames(reader, info, project, times, cancel, workers):
                        if cancel.is_set():
                            raise Cancelled("Export canceled.")
                        process.stdin.write(memoryview(raw).cast("B"))
                        frames += 1
                        now = time.monotonic()
                        if index == 0 or frames == expected or now-last_frame_report >= .25:
                            progress(min(.93, render_start+frames/max(expected, 1)*(.93-render_start)), f"Rendering frame {frames}/{expected}")
                            last_frame_report = now
                process.stdin.close()
                if process.wait():
                    errors.seek(0)
                    raise UWCCError(errors.read()[-6000:].decode("utf-8", "replace"))
            except BrokenPipeError as exc:
                if cancel.is_set():
                    raise Cancelled("Export canceled.") from exc
                errors.seek(0)
                raise UWCCError(errors.read()[-6000:].decode("utf-8", "replace")) from exc
            finally:
                stop_watch.set()
                if process.poll() is None:
                    process.kill()
                process.wait()
                if not process.stdin.closed:
                    process.stdin.close()
                watcher.join(timeout=1)
        if frames != expected:
            raise UWCCError(f"Frame count changed during decode: expected {expected}, got {frames}.")
        progress(.94, "Preserving audio and metadata…")
        # Offset the original audio relative to the decoded video stream start.
        offset = info.format_start-info.start
        mux_audio = "copy" if audio_codec == "aac" and all(c == "aac" for c in info.audio_codecs) else audio_codec
        mux = [executable("ffmpeg"), "-nostdin", "-v", "error", "-i", str(video),
               "-itsoffset", str(offset), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
               "-map_metadata", "1", "-metadata", "location=", "-metadata", "location-eng=",
               "-metadata", "com.apple.quicktime.location.ISO6709=", "-metadata:s:v:0", "rotate=0",
               "-c:v", "copy", "-c:a", mux_audio]
        if mux_audio == "aac":
            mux += ["-b:a", "320k"]
        if extension == ".mp4":
            mux += ["-movflags", "+faststart"]
        mux += [str(final)]
        run(mux, cancel)
        progress(.97, "Validating rendered video…")
        result = probe(final, cancel)
        tolerance = 1/float(info.fps)+.003
        count, _ = frame_timing(result, cancel)
        if count != expected or abs(result.duration-info.duration) > tolerance or result.audio_count != info.audio_count:
            raise UWCCError("Output failed frame-count, duration, or audio-stream validation.")
        if result.transfer != transfer or result.primaries != primaries:
            raise UWCCError("Output color signaling did not match the requested color space.")
        if cancel.is_set():
            raise Cancelled("Export canceled.")
        if overwrite:
            os.replace(final, destination)
        else:
            # Exclusive publication avoids overwriting a file created during rendering.
            try:
                os.link(final, destination)
            except FileExistsError as exc:
                raise UWCCError("Output appeared while rendering; it was not overwritten.") from exc
        progress(1, f"Saved {destination.name}")
    return destination
