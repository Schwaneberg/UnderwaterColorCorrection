"""Bounded Windows frame pipes, without modifying subprocess globals."""
import os
import subprocess

_WINDOWS = os.name == "nt"
MAX_PIPE_BYTES = 1024*1024


def _windows_pipe(stream, buffer_bytes):
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_pipe = kernel32.CreatePipe
    create_pipe.argtypes = [ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(wintypes.HANDLE),
                            ctypes.c_void_p, wintypes.DWORD]
    create_pipe.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    read_handle, write_handle = wintypes.HANDLE(), wintypes.HANDLE()
    # NULL security attributes leave both ends non-inheritable. Popen duplicates
    # just the child endpoint into its standard-handle inheritance list.
    if not create_pipe(ctypes.byref(read_handle), ctypes.byref(write_handle), None, buffer_bytes):
        raise ctypes.WinError(ctypes.get_last_error())
    handles = [read_handle.value, write_handle.value]
    streams, pending_fds = [], []
    try:
        for i, mode in enumerate(("rb", "wb")):
            flags = os.O_BINARY | os.O_NOINHERIT | (os.O_RDONLY if i == 0 else os.O_WRONLY)
            fd = msvcrt.open_osfhandle(handles[i], flags)
            handles[i] = None  # The CRT descriptor now owns this handle.
            pending_fds.append(fd)
            os.set_inheritable(fd, False)
            file = os.fdopen(fd, mode)
            pending_fds.remove(fd)  # The file object now owns the descriptor.
            streams.append(file)
    except BaseException:
        for file in streams:
            file.close()
        for fd in pending_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    finally:
        for handle in handles:
            if handle is not None:
                close_handle(handle)
    return (streams[0], streams[1]) if stream == "stdout" else (streams[1], streams[0])


def frame_process(args, stream, *, buffer_bytes=MAX_PIPE_BYTES, **kwargs):
    """Start a frame producer/consumer with a normal Popen-compatible stream.

    Larger Windows buffers avoid the default 4 KiB pipe bottleneck. Other
    platforms retain subprocess's native pipe implementation. Caller owns the
    returned process and parent stream, just as with subprocess.PIPE.
    """
    if stream not in ("stdin", "stdout"):
        raise ValueError("Frame pipe must be stdin or stdout")
    if buffer_bytes <= 0:
        raise ValueError("Frame pipe buffer must be positive")
    if not _WINDOWS:
        kwargs[stream] = subprocess.PIPE
        return subprocess.Popen(args, **kwargs)
    parent, child = _windows_pipe(stream, max(4096, min(buffer_bytes, MAX_PIPE_BYTES)))
    try:
        kwargs[stream] = child
        process = subprocess.Popen(args, **kwargs)
    except BaseException:
        parent.close()
        raise
    finally:
        child.close()
    setattr(process, stream, parent)
    return process
