"""Runtime-tested NVIDIA video codecs with software fallback before streaming."""

import subprocess
import tempfile
import threading
import time

from .errors import Cancelled, UWCCError

ENCODERS = {"cpu": "CPU (default)", "nvenc": "NVIDIA NVENC", "auto": "NVIDIA with CPU fallback (opt-in)"}
DECODERS = {"cpu": "CPU", "auto": "Auto (NVIDIA when supported)", "cuda": "NVIDIA NVDEC"}
_checks = {}
_lock = threading.Lock()


def _check(args, cancel):
    from .media import NO_WINDOW
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=errors, **NO_WINDOW)
        deadline = time.monotonic() + 15
        try:
            while process.poll() is None:
                if cancel and cancel.is_set():
                    raise Cancelled("Operation canceled.")
                if time.monotonic() > deadline:
                    return False, "Hardware codec check timed out."
                time.sleep(.05)
            errors.seek(0, 2)
            errors.seek(max(0, errors.tell() - 2000))
            message = errors.read(2000).decode("utf-8", "replace").strip()
            return process.returncode == 0, message
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _cached(key, args, cancel):
    if cancel and cancel.is_set():
        raise Cancelled("Operation canceled.")
    with _lock:
        result = _checks.get(key)
    if result is None:
        result = _check(args, cancel)
        with _lock:
            _checks[key] = result
    return result


def nvenc_args(profile):
    return ["-c:v", "h264_nvenc" if profile == "h264" else "hevc_nvenc",
            "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "17" if profile == "h264" else "18",
            "-b:v", "0", "-pix_fmt", "yuv420p" if profile == "h264" else "p010le",
            *(["-tag:v", "hvc1"] if profile == "hevc" else [])]


def encoder_probe(profile, info, cancel):
    from .media import executable
    binary = executable("ffmpeg")
    args = [binary, "-nostdin", "-v", "error", "-filter_threads", "1", "-f", "lavfi", "-i",
            f"color=size={info.width}x{info.height}:rate={info.fps}", "-frames:v", "2",
            *nvenc_args(profile), "-f", "null", "-"]
    return _cached((binary, profile, info.width, info.height, str(info.fps)), args, cancel)


def encoder(profile, software_args, info, preference, cancel):
    if preference not in ENCODERS:
        raise UWCCError("Unknown video encoder preference.")
    if profile not in ("h264", "hevc"):
        if preference == "nvenc":
            raise UWCCError("NVIDIA NVENC is available for H.264 and HEVC export only.")
        return list(software_args), "CPU encoding"
    if preference == "cpu":
        return list(software_args), "CPU encoding"
    available, reason = encoder_probe(profile, info, cancel)
    if available:
        return nvenc_args(profile), "NVIDIA NVENC encoding"
    if preference == "nvenc":
        raise UWCCError("NVIDIA encoder is unavailable: " + reason)
    return list(software_args), "CPU encoding (NVIDIA unavailable)"


def decoder(info, preference, cancel):
    """Return input options, download filters, and the actual backend label.

    Keep 8/10-bit source precision during download; the existing CPU filters
    and RGB48 transport follow. Unsupported layouts use software in auto mode.
    """
    from .media import executable
    if preference not in DECODERS:
        raise UWCCError("Unknown video decoder preference.")
    if preference == "cpu":
        return [], [], "CPU decoding"
    layout = {"yuv420p": "nv12", "yuvj420p": "nv12", "yuv420p10le": "p010le"}.get(info.pixel_format)
    if layout:
        binary = executable("ffmpeg")
        options = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        # Normalize the downloaded semiplanar frame before RGB48 conversion.
        # Direct NV12 -> RGB48 conversion loses a color channel with some
        # FFmpeg/swscale builds. Match the software decoder's planar layout.
        filters = ["hwdownload", f"format={layout}", f"format={info.pixel_format}"]
        args = [binary, "-nostdin", "-v", "error", "-filter_threads", "1", *options, "-i", info.path,
                "-map", f"0:{info.video_index}", "-an", "-sn", "-dn", "-frames:v", "2",
                "-vf", ",".join(filters), "-f", "null", "-"]
        from pathlib import Path
        source = Path(info.path)
        key = (binary, "decode", info.path, source.stat().st_mtime_ns, info.video_index, layout)
        available, reason = _cached(key, args, cancel)
        if available:
            return options, filters, "NVIDIA NVDEC decoding"
    else:
        reason = f"Unsupported NVIDIA download format: {info.pixel_format}."
    if preference == "cuda":
        raise UWCCError("NVIDIA decoder is unavailable: " + reason)
    return [], [], "CPU decoding (NVIDIA unavailable for this source)"
