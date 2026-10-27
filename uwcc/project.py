"""Validated editing state and atomic, portable persistence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from .errors import UWCCError

# name: (label, minimum, maximum, default, step)
CONTROLS = {
    "strength": ("Correction strength", 0, 200, 100, 1),
    "temperature": ("Temperature", -100, 100, 0, 1),
    "tint": ("Tint", -100, 100, 0, 1),
    "exposure": ("Exposure (EV)", -4, 4, 0, 0.05),
    "contrast": ("Contrast", -100, 100, 0, 1),
    "highlights": ("Highlights", -100, 100, 0, 1),
    "shadows": ("Shadows", -100, 100, 0, 1),
    "black_point": ("Black point", -100, 100, 0, 1),
    "saturation": ("Saturation", 0, 200, 100, 1),
    "hue": ("Hue (degrees)", -180, 180, 0, 1),
    "dehaze": ("Dehaze", -100, 100, 0, 1),
    "sharpness": ("Sharpness", 0, 100, 0, 1),
}
PROFILES = {
    "auto": "Auto-detect input profile",
    "rec709": "Rec.709 / standard gamma",
    "slog2": "Sony S-Log2 / S-Gamut",
    "slog3": "Sony S-Log3 / S-Gamut3.Cine",
    "hlg": "Rec.2020 HLG",
    "pq": "Rec.2020 PQ",
}
OUTPUTS = {"rec709": "SDR Rec.709", "hlg": "HDR HLG (1000 nits)", "pq": "HDR PQ (1000 nits)"}
FILTERS = {"auto": "Auto", "none": "No filter", "red": "Red filter", "magenta": "Magenta filter"}


def defaults():
    return {key: float(item[3]) for key, item in CONTROLS.items()}


def user_directory(kind="data", *, create=True) -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    else:
        variable, suffix = ("XDG_CACHE_HOME", ".cache") if kind == "cache" else ("XDG_DATA_HOME", ".local/share")
        root = Path(os.environ.get(variable, str(Path.home() / suffix)))
    result = root / "uwcc"
    if create:
        result.mkdir(parents=True, exist_ok=True)
    return result


def model_directory() -> Path:
    """User-owned model storage, independent of the application installation."""
    override = os.environ.get("UWCC_MODEL_DIR")
    return Path(override).expanduser() if override else user_directory(create=False)/"models"


def fingerprint(path: Path) -> dict:
    path = Path(path)
    stat = path.stat()
    with path.open("rb") as stream:
        first = stream.read(4 * 1024 * 1024)
        stream.seek(max(0, stat.st_size - 4 * 1024 * 1024))
        last = stream.read()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "first_4m_sha256": hashlib.sha256(first).hexdigest(),
            "last_4m_sha256": hashlib.sha256(last).hexdigest()}


def atomic_json(path: Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            name = stream.name
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def read_json(path: Path):
    if Path(path).stat().st_size > 32 * 1024 * 1024:
        raise UWCCError("Project or preset exceeds the 32 MiB safety limit.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise UWCCError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise UWCCError("Unsupported document version. This build supports schema 1.")
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise UWCCError(f"Invalid JSON document: {exc}") from exc


def valid_value(name, value):
    if name not in CONTROLS or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UWCCError(f"Invalid correction parameter: {name}")
    _, low, high, _, _ = CONTROLS[name]
    if not math.isfinite(value) or not low <= value <= high:
        raise UWCCError(f"{name} must be between {low} and {high}.")
    return float(value)


@dataclass
class Grade:
    values: dict = field(default_factory=defaults)
    keyframes: dict = field(default_factory=dict)
    input_profile: str = "auto"
    output_profile: str = "rec709"
    filter_hint: str = "auto"
    model_manifest: str | None = None
    stabilize: bool = False
    denoise: float = 0
    auto_method: str = "adaptive"
    denoise_manifest: str | None = None
    stabilization_region: str = "seabed"
    denoise_engine: str = "classical"
    rotation: str = "auto"

    def at(self, seconds: float) -> dict:
        result = dict(self.values)
        for name, keys in self.keyframes.items():
            ordered = sorted((float(Fraction(item["time"])), item["value"]) for item in keys)
            if not ordered:
                continue
            result[name] = ordered[0][1]
            for i, (time, value) in enumerate(ordered):
                if time <= seconds:
                    result[name] = value
                    continue
                if i:
                    t0, v0 = ordered[i - 1]
                    result[name] = v0 + (value - v0) * (seconds - t0) / (time - t0)
                break
        return result

    def set_key(self, name, seconds, value):
        value = valid_value(name, value)
        time = str(Fraction(seconds).limit_denominator(1_000_000))
        keys = [item for item in self.keyframes.get(name, []) if item["time"] != time]
        keys.append({"time": time, "value": value})
        self.keyframes[name] = sorted(keys, key=lambda k: Fraction(k["time"]))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        grade = cls()
        grade.rotation = data.get("rotation", "auto")
        if grade.rotation not in ("auto", "0", "90", "180", "270"):
            raise UWCCError("Unknown video rotation.")
        grade.auto_method = data.get("auto_method", "adaptive")
        if grade.auto_method not in ("ai", "adaptive", "legacy"):
            raise UWCCError("Unknown automatic correction method.")
        grade.stabilization_region = data.get("stabilization_region", "seabed")
        if grade.stabilization_region not in ("seabed", "whole"):
            raise UWCCError("Unknown stabilization reference region.")
        grade.denoise_engine = data.get("denoise_engine", "classical")
        if grade.denoise_engine not in ("classical", "ai"):
            raise UWCCError("Unknown denoise engine.")
        if grade.denoise_engine == "ai":
            from .neural import builtin_model
            grade.denoise_manifest = builtin_model("denoise")
        if not isinstance(data.get("stabilize", False), bool):
            raise UWCCError("Stabilization must be enabled or disabled.")
        grade.stabilize = data.get("stabilize", False)
        denoise = data.get("denoise", 0)
        if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(denoise) or not 0 <= denoise <= 10:
            raise UWCCError("Denoise strength must be between 0 and 10.")
        grade.denoise = float(denoise)
        for key, value in data.get("values", {}).items():
            grade.values[key] = valid_value(key, value)
        for key, choices in (("input_profile", PROFILES), ("output_profile", OUTPUTS), ("filter_hint", FILTERS)):
            value = data.get(key, getattr(grade, key))
            if value not in choices:
                raise UWCCError(f"Unknown {key}: {value}")
            setattr(grade, key, value)
        frames = data.get("keyframes", {})
        if not isinstance(frames, dict) or sum(len(v) for v in frames.values()) > 100_000:
            raise UWCCError("Invalid or excessive keyframes.")
        for key, keys in frames.items():
            if key not in CONTROLS:
                raise UWCCError(f"Unknown keyframed parameter: {key}")
            for item in keys:
                try:
                    time = Fraction(item["time"])
                except (ValueError, ZeroDivisionError, TypeError) as exc:
                    raise UWCCError("Invalid keyframe timestamp.") from exc
                if time < 0 or time > 10**8:
                    raise UWCCError("Keyframe timestamp is outside the supported range.")
                grade.set_key(key, time, item["value"])
        # Local model paths are never loaded implicitly from someone else's project.
        return grade


@dataclass
class Project:
    source: str
    grade: Grade = field(default_factory=Grade)
    source_fingerprint: dict = field(default_factory=dict)
    analysis: dict | None = None
    path: Path | None = None
    orientation: dict | None = None

    def to_dict(self, relative_to=None):
        source = self.source
        if relative_to:
            try:
                source = os.path.relpath(source, relative_to)
            except ValueError:
                pass
        return {"schema_version": 1, "application_version": "0.1.0", "source": source,
                "source_fingerprint": self.source_fingerprint,
                "grade": self.grade.to_dict(), "analysis": self.analysis, "orientation": self.orientation}

    def save(self, path):
        path = Path(path).resolve()
        if path == Path(self.source).resolve():
            raise UWCCError("A project cannot overwrite its source video.")
        atomic_json(path, self.to_dict(path.parent))
        self.path = path

    @classmethod
    def load(cls, path, replacement=None):
        from .analysis import validate_analysis
        path = Path(path).resolve()
        data = read_json(path)
        if not isinstance(data.get("source"), str):
            raise UWCCError("Project is missing a source path.")
        source = Path(replacement or data["source"])
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise UWCCError(f"Source is missing. Relink the video: {source}")
        actual = fingerprint(source)
        if actual != data.get("source_fingerprint") and not replacement:
            raise UWCCError("Source has changed. Explicitly relink it to invalidate old analysis.")
        analysis = data.get("analysis") if actual == data.get("source_fingerprint") else None
        if analysis is not None:
            validate_analysis(analysis)
        from .orientation import validate_decision
        orientation = data.get("orientation") if actual == data.get("source_fingerprint") else None
        if "orientation" not in data and not replacement:
            orientation = {"correction_cw": 0, "status": "legacy",
                           "message": "Saved project orientation retained. Manual rotation is available."}
        return cls(str(source), Grade.from_dict(data["grade"]), actual, analysis, path, validate_decision(orientation))


BUILTINS = {
    "Natural": {},
    "Neutralize only": {"contrast": -10, "strength": 65},
    "Red-filter natural": {"strength": 70},
    "Vivid": {"saturation": 120, "contrast": 15},
}


def save_preset(path, grade: Grade, name="Custom"):
    data = grade.to_dict()
    data["keyframes"] = {}
    data["model_manifest"] = None
    atomic_json(path, {"schema_version": 1, "kind": "uwcc-preset", "name": name, "grade": data})


def load_preset(path):
    data = read_json(path)
    if data.get("kind") != "uwcc-preset":
        raise UWCCError("This file is not a UWCC preset.")
    return Grade.from_dict(data["grade"])


def clone(value):
    return copy.deepcopy(value)
