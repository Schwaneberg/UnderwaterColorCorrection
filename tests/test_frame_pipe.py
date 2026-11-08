"""Frame-pipe ownership, EOF, startup failure, and independent decoded frames."""
import io
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from uwcc import _frame_pipe
from uwcc.errors import Cancelled, UWCCError
from uwcc.media import FrameReader, NO_WINDOW


@pytest.mark.parametrize("stream", ["stdin", "stdout"])
def test_frame_pipe_large_transfer_and_eof(stream):
    payload = b"rgb48\x00\xff"*200003  # Larger than the pipe, with a partial final block.
    script = ("import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)"
              if stream == "stdin" else f"import sys; sys.stdout.buffer.write({payload[:7]!r}*200003)")
    kwargs = {"stderr": subprocess.PIPE}
    kwargs["stdout" if stream == "stdin" else "stdin"] = subprocess.PIPE if stream == "stdin" else subprocess.DEVNULL
    process = _frame_pipe.frame_process([sys.executable, "-c", script], stream,
                                        buffer_bytes=4*_frame_pipe.MAX_PIPE_BYTES, **kwargs, **NO_WINDOW)
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt
            api = ctypes.WinDLL("kernel32", use_last_error=True)
            api.GetNamedPipeInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p,
                ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            api.GetNamedPipeInfo.restype = wintypes.BOOL
            outsize, insize = wintypes.DWORD(), wintypes.DWORD()
            handle = msvcrt.get_osfhandle(getattr(process, stream).fileno())
            assert api.GetNamedPipeInfo(handle, None, ctypes.byref(outsize), ctypes.byref(insize), None)
            assert max(outsize.value, insize.value) == _frame_pipe.MAX_PIPE_BYTES
        output, errors = process.communicate(payload if stream == "stdin" else None, timeout=15)
        assert process.returncode == 0, errors
        assert output == payload
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for file in (process.stdin, process.stdout, process.stderr):
            if file is not None:
                file.close()


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_windows_pipe_endpoints_close_on_startup_failure(monkeypatch, failure):
    parent, child = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(_frame_pipe, "_WINDOWS", True)
    monkeypatch.setattr(_frame_pipe, "_windows_pipe", lambda *args: (parent, child))

    def fail(*args, **kwargs):
        raise failure("startup failed")

    monkeypatch.setattr(_frame_pipe.subprocess, "Popen", fail)
    with pytest.raises(failure):
        _frame_pipe.frame_process(["missing"], "stdout")
    assert parent.closed and child.closed


def test_non_windows_uses_standard_subprocess_pipe(monkeypatch):
    monkeypatch.setattr(_frame_pipe, "_WINDOWS", False)
    calls = []
    monkeypatch.setattr(_frame_pipe.subprocess, "Popen", lambda args, **kwargs: calls.append(kwargs) or "process")
    assert _frame_pipe.frame_process(["decoder"], "stdout", stdin=subprocess.DEVNULL) == "process"
    assert calls == [{"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE}]
    with pytest.raises(ValueError):
        _frame_pipe.frame_process([], "stderr")
    with pytest.raises(ValueError):
        _frame_pipe.frame_process([], "stdout", buffer_bytes=0)


class ShortReads(io.BytesIO):
    def readinto(self, view):
        return super().readinto(view[:7])


def reader_for(stream, code=0, error=b""):
    reader = object.__new__(FrameReader)
    reader.width, reader.height = 3, 2
    reader.cancel = threading.Event()
    reader.process = SimpleNamespace(stdout=stream, wait=lambda: code)
    reader.errors = io.BytesIO(error)
    return reader


def test_readinto_handles_short_reads_without_reusing_yielded_frames():
    raw = np.arange(36, dtype="<u2").reshape(2, 2, 3, 3)
    raw[1] += 1000
    reader = reader_for(ShortReads(raw.tobytes()))
    frames = list(reader._decoded())
    expected = raw.astype(np.float32)/65535
    np.testing.assert_array_equal(frames, expected)
    assert not np.shares_memory(frames[0], frames[1])


def test_readinto_reports_truncated_frames_and_decoder_errors():
    with pytest.raises(UWCCError, match="incomplete"):
        list(reader_for(ShortReads(b"\x00"*35))._decoded())
    with pytest.raises(UWCCError, match="decoder failed"):
        list(reader_for(ShortReads(b""), 1, b"decoder failed")._decoded())


def test_readinto_cancellation_during_partial_frame():
    reader = reader_for(ShortReads(b"\x00"*7))
    original = reader.process.stdout.readinto

    def cancel(view):
        count = original(view)
        reader.cancel.set()
        return count

    reader.process.stdout.readinto = cancel
    with pytest.raises(Cancelled):
        list(reader._decoded())
