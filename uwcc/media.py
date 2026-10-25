"""Bounded FFmpeg pipes; no shell interpolation and no complete video in memory."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .errors import Cancelled, UWCCError

NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def cpu_threads():
    """Bound native pools; honor the CPU affinity of containers/workstations."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        requested = int(os.environ.get("UWCC_THREADS", min(8, available)))
    except ValueError as exc:
        raise UWCCError("UWCC_THREADS must be an integer from 1 to 32.") from exc
    if not 1 <= requested <= 32:
        raise UWCCError("UWCC_THREADS must be an integer from 1 to 32.")
    return min(requested, available)


def executable(name):
    found = shutil.which(name)
    if not found:
        raise UWCCError(f"{name} is missing. Install FFmpeg and make it available on PATH.")
    return found


def run(args, cancel=None):
    with tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as output:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output, stderr=errors, **NO_WINDOW)
        try:
            while True:
                try:
                    code = process.wait(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    if cancel and cancel.is_set():
                        raise Cancelled("Operation canceled.")
            if code:
                errors.seek(0)
                raise UWCCError(errors.read()[-6000:].decode("utf-8", "replace").strip())
            output.seek(0)
            return output.read()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    duration: float
    fps: Fraction
    video_index: int
    audio_count: int
    profile: str
    rotation: int = 0
    start: float = 0
    format_start: float = 0
    sar: str = "1:1"
    primaries: str = "unknown"
    transfer: str = "unknown"
    color_range: str = "unknown"
    audio_codecs: list = field(default_factory=list)
    audio_offsets: list = field(default_factory=list)


def probe(path, cancel=None):
    source = Path(path).resolve()
    if not source.is_file():
        raise UWCCError(f"Video does not exist: {source}")
    raw = run([executable("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)], cancel)
    try:
        data = json.loads(raw)
        streams = data.get("streams", [])
        video = next(s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic"))
        fps = Fraction(video.get("avg_frame_rate", "0/1"))
        if fps <= 0:
            fps = Fraction(video.get("r_frame_rate", "25/1"))
        duration = float(video.get("duration", data.get("format", {}).get("duration", 0)))
        rotation = int(round(float(next((s["rotation"] for s in video.get("side_data_list", []) if "rotation" in s), 0)))) % 360
        width, height = int(video["width"]), int(video["height"])
        if rotation in (90, 270):
            width, height = height, width
        trc, primaries = video.get("color_transfer", "unknown"), video.get("color_primaries", "unknown")
        profile = {"arib-std-b67": "hlg", "smpte2084": "pq"}.get(trc, "auto")
        if trc == "bt709" and primaries == "bt709":
            profile = "rec709"
        if duration <= 0 or not math.isfinite(duration) or width*height > 100_000_000:
            raise ValueError("unsupported duration or dimensions")
        return MediaInfo(str(source), width, height, duration, fps, int(video["index"]),
                         sum(s["codec_type"] == "audio" for s in streams), profile, rotation,
                         float(video.get("start_time", 0)), float(data.get("format", {}).get("start_time", 0)),
                         video.get("sample_aspect_ratio", "1:1"), primaries, trc, video.get("color_range", "unknown"),
                         [s.get("codec_name", "") for s in streams if s["codec_type"] == "audio"],
                         [float(s.get("start_time", 0))-float(video.get("start_time", 0)) for s in streams if s["codec_type"] == "audio"])
    except (ValueError, KeyError, StopIteration, ZeroDivisionError) as exc:
        raise UWCCError(f"No supported, finite-duration video stream: {exc}") from exc


def proxy_size(info, max_width):
    scale = min(1, max_width / max(info.width, info.height)) if max_width else 1
    return max(2, int(info.width*scale)//2*2), max(2, int(info.height*scale)//2*2)


class FrameReader:
    """Read RGB48LE frames while a watcher makes blocked pipe reads cancelable."""

    def __init__(self, info, max_width=None, start=0, sample_rate=None, cancel=None, single=False, finishing=None, stabilization_plan=None, progress=None):
        self.width, self.height = proxy_size(info, max_width)
        self.cancel = cancel or threading.Event()
        self.stabilizer = self.denoiser = None
        if finishing is not None:
            if finishing.denoise and finishing.denoise_engine == "ai" and not finishing.denoise_manifest:
                raise UWCCError("The selected AI denoise model is unavailable. Install FastDVDnet or select conventional denoise.")
            if finishing.stabilize:
                from .finishing import build_stabilization_plan
                if start or sample_rate or single:
                    raise UWCCError("Stabilized frame reads require a complete, unsampled video stream.")
                self.stabilizer = stabilization_plan or build_stabilization_plan(info, finishing, self.cancel)
            if finishing.denoise and finishing.denoise_manifest:
                if finishing.input_profile != "rec709":
                    raise UWCCError("Neural denoise currently requires Rec.709 input; use conventional denoise for log/HDR.")
                from .finishing import TemporalDenoiser
                if progress:
                    progress("Preparing AI denoise model…")
                self.denoiser = TemporalDenoiser(finishing.denoise_manifest, finishing.denoise, self.cancel, progress)
                if progress:
                    progress(f"AI denoise backend: {self.denoiser.network.execution_label}")
        self.closed = threading.Event()
        self.errors = tempfile.TemporaryFile()
        threads = cpu_threads()
        args = [executable("ffmpeg"), "-nostdin", "-v", "error", "-threads", str(threads),
                "-filter_threads", str(min(4, threads))]
        if start:
            args += ["-ss", str(max(0, start))]
        args += ["-i", info.path, "-map", f"0:{info.video_index}", "-an", "-sn", "-dn"]
        filters = []
        # Apply before scaling and RGB conversion, in the source YUV domain.
        # These streaming native filters preserve frame count and timestamps.
        if finishing is not None:
            if finishing.denoise and not self.denoiser:
                strength = float(finishing.denoise)
                filters += [f"hqdn3d={strength:g}:{strength*.75:g}:{strength*1.5:g}:{strength*1.125:g}"]
        if sample_rate:
            filters += [f"fps={sample_rate}"]
        filters += [f"scale={self.width}:{self.height}:flags=bilinear"]
        args += ["-vf", ",".join(filters), "-pix_fmt", "rgb48le", "-fps_mode", "passthrough"]
        if single:
            args += ["-frames:v", "1"]
        args += ["-threads", "1", "-f", "rawvideo", "pipe:1"]
        self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=self.errors, **NO_WINDOW)
        self.watcher = threading.Thread(target=self._watch, daemon=True)
        self.watcher.start()

    def _watch(self):
        while not self.closed.wait(.1):
            if self.cancel.is_set():
                if self.process.poll() is None:
                    self.process.kill()
                return

    def __iter__(self):
        frames = self._decoded()
        if self.denoiser:
            frames = self.denoiser.stream(frames)
        if self.stabilizer:
            frames = self.stabilizer.warp(frames)
        return iter(frames)

    def _decoded(self):
        nbytes = self.width*self.height*6
        while True:
            if self.cancel.is_set():
                raise Cancelled("Operation canceled.")
            raw = bytearray()
            while len(raw) < nbytes:
                chunk = self.process.stdout.read(nbytes-len(raw))
                if not chunk:
                    break
                raw.extend(chunk)
            if not raw:
                if self.cancel.is_set():
                    raise Cancelled("Operation canceled.")
                code = self.process.wait()
                if code:
                    self.errors.seek(0)
                    raise UWCCError(self.errors.read()[-6000:].decode("utf-8", "replace"))
                return
            if len(raw) != nbytes:
                raise UWCCError("Decoder returned an incomplete video frame.")
            yield np.frombuffer(raw, dtype="<u2").reshape(self.height, self.width, 3).astype(np.float32)/65535

    def close(self):
        self.closed.set()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdout.close()
        self.watcher.join(timeout=1)
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_frame(info, seconds=0, max_width=960, cancel=None):
    with FrameReader(info, max_width=max_width, start=min(seconds, max(0, info.duration-1/float(info.fps))),
                     cancel=cancel, single=True) as reader:
        try:
            return next(iter(reader))
        except StopIteration as exc:
            raise UWCCError("No frame could be decoded at this time.") from exc


def prepare_frame(frame, info, profile):
    """Restore Sony code-value scale after FFmpeg's legal-to-full expansion.

    Log inverse functions consume normalized stored code values, whereas SDR
    and HDR inverse functions consume the expanded signal. Unknown range is
    treated as limited for Sony camera YUV footage, matching FFmpeg's default.
    """
    if profile in ("slog2", "slog3") and info.color_range != "pc":
        return frame*(876/1023)+64/1023
    return frame


def frame_timestamps(info, cancel=None, start=None):
    """Yield presentation timestamps without retaining the clip in memory."""
    cancel = cancel or threading.Event()
    args = [executable("ffprobe"), "-v", "error", "-threads", str(cpu_threads()),
            "-select_streams", str(info.video_index)]
    if start:
        args += ["-read_intervals", f"{max(0, start+info.start)}%"]
    args += ["-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", info.path]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors, text=True, **NO_WINDOW)
        stopped = threading.Event()
        def watch():
            while not stopped.wait(.1):
                if cancel.is_set() and process.poll() is None:
                    process.kill()
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        previous = -float("inf")
        try:
            for line in process.stdout:
                if cancel.is_set():
                    raise Cancelled("Operation canceled.")
                try:
                    timestamp = float(line.strip().split(",")[0])
                except ValueError:
                    continue
                if not math.isfinite(timestamp) or timestamp < previous:
                    raise UWCCError("Video presentation timestamps are invalid or out of order.")
                previous = timestamp
                if start is not None and timestamp-info.start < start-.000001:
                    continue
                yield timestamp-info.start
            if cancel.is_set():
                raise Cancelled("Operation canceled.")
            if process.wait():
                errors.seek(0)
                raise UWCCError(errors.read().decode("utf-8", "replace"))
        finally:
            stopped.set()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            watcher.join(timeout=1)


def frame_timing(info, cancel=None):
    """Check cadence over the entire source."""
    last, count, variable = None, 0, False
    interval = 1/float(info.fps)
    for timestamp in frame_timestamps(info, cancel):
        if last is not None and abs(timestamp-last-interval) > max(.00015, interval*.02):
            variable = True
        last = timestamp
        count += 1
    if not count:
        raise UWCCError("Video has no decodable frame timestamps.")
    return count, variable
