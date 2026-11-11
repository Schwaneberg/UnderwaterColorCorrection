"""Shared FFmpeg executable resolution for runtime and Windows packaging."""
from pathlib import Path


def resolve_shim(path: Path) -> Path:
    """Follow Chocolatey/Scoop launchers to the actual executable.

    Launchers start another process. Killing and waiting for a launcher does
    not stop its child, which can retain pipes and output-file handles. Copying
    a launcher into a bundle also breaks its relative target path.
    """
    parts = [part.lower() for part in path.parts]
    name = path.name
    candidates = []
    if "chocolatey" in parts and path.parent.name.lower() == "bin":
        candidates.extend(sorted((path.parent.parent / "lib").glob(f"ffmpeg*/tools/**/{name}")))
    if path.parent.name.lower() == "shims" and "scoop" in parts:
        candidates.extend(sorted((path.parent.parent / "apps").glob(f"ffmpeg*/current/**/{name}")))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > path.stat().st_size:
            return candidate.resolve()
    return path
