"""Detect supported transfer/gamut pairs from declared capture or stream metadata.

Sony capture XML takes precedence: a Rec.709 YUV matrix alone says nothing
about the recording gamma. No pixel, filename, or camera-model heuristics.
"""

import struct
import xml.etree.ElementTree as ET


def capture_metadata(path):
    """Read bounded MP4 metadata boxes, seeking past video payloads.

    Returns (gamma, gamut), None when absent, or empty values for unreadable
    capture declarations. The latter prevents a misleading stream fallback.
    """
    declarations = []
    remaining = 2 * 1024 * 1024
    boxes_left = 4096

    def xml(payload):
        start = payload.find(b"<NonRealTimeMeta")
        if start < 0:
            return
        end = payload.find(b"</NonRealTimeMeta>", start)
        if end < 0 or b"<!DOCTYPE" in payload or b"<!ENTITY" in payload:
            declarations.append(("", ""))
            return
        try:
            root = ET.fromstring(payload[start:end + len(b"</NonRealTimeMeta>")])
            fields = {}
            for item in root.iter():
                name = item.get("name")
                if name in ("CaptureGammaEquation", "CaptureColorPrimaries"):
                    fields.setdefault(name, set()).add(item.get("value", "").strip().lower())
            if fields:
                declarations.append(tuple(next(iter(fields.get(key, {""})))
                                          if len(fields.get(key, {""})) == 1 else ""
                                          for key in ("CaptureGammaEquation", "CaptureColorPrimaries")))
        except ET.ParseError:
            declarations.append(("", ""))

    try:
        with open(path, "rb") as source:
            source.seek(0, 2)
            length = source.tell()

            def walk(start, end, depth=0):
                nonlocal remaining, boxes_left
                while start + 8 <= end and boxes_left > 0:
                    boxes_left -= 1
                    source.seek(start)
                    header = source.read(8)
                    if len(header) != 8:
                        return
                    size, kind = struct.unpack(">I4s", header)
                    header_size = 8
                    if size == 1:
                        extended = source.read(8)
                        if len(extended) != 8:
                            return
                        size = struct.unpack(">Q", extended)[0]
                        header_size = 16
                    elif size == 0:
                        size = end - start
                    if size < header_size or start + size > end:
                        return
                    if kind in (b"moov", b"udta", b"meta") and depth < 8:
                        walk(start + header_size + (4 if kind == b"meta" else 0), start + size, depth + 1)
                    elif kind in (b"xml ", b"nrtm", b"uuid"):
                        count = size - header_size
                        if count <= remaining:
                            source.seek(start + header_size)
                            remaining -= count
                            xml(source.read(count))
                        else:
                            # Metadata exceeding our budget cannot be trusted as absent.
                            declarations.append(("", ""))
                    start += size

            walk(0, length)
            if boxes_left == 0:
                declarations.append(("", ""))
    except OSError:
        return ("", "")
    if not declarations:
        return None
    return declarations[0] if len(set(declarations)) == 1 else ("", "")


def detect_profile(path, transfer, primaries):
    """Return (profile, evidence); 'auto' means a user decision is required."""
    capture = capture_metadata(path)
    if capture is not None:
        supported = {
            ("rec709", "rec709"): "rec709",
            ("rec709-xvycc", "rec709"): "rec709",
            ("s-log2", "s-gamut"): "slog2",
            ("s-log3", "s-gamut3-cine"): "slog3",
            ("s-log3-cine", "s-gamut3-cine"): "slog3",
            ("rec2100-hlg", "rec2020"): "hlg",
            ("rec2100-pq", "rec2020"): "pq",
        }
        profile = supported.get(capture, "auto")
        detail = " / ".join(value[:100] or "unspecified" for value in capture)
        return profile, f"Sony capture metadata: {detail}." + (
            " Select the matching supported input profile manually." if profile == "auto" else "")
    if transfer in ("bt709", "iec61966-2-4") and primaries == "bt709":
        return "rec709", (f"Stream tags: {transfer} / {primaries}; using Rec.709. "
                          "Override if the camera recorded a custom or log profile.")
    if primaries == "bt2020" and transfer in ("arib-std-b67", "smpte2084"):
        return ("hlg" if transfer == "arib-std-b67" else "pq"), f"Stream tags: {transfer} / {primaries}."
    return "auto", f"No supported gamma/gamut pair found ({transfer} / {primaries}). Choose the input profile."
