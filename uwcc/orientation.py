"""Optional local quarter-turn detection and consistent video geometry."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import Cancelled, UWCCError
from .neural import builtin_model

MODEL_FILE = "max-l280-int8.onnx"
MODEL_SHA256 = "9e358f9c51587ec4472ac9496e52dbcaa428886613ebe338fb0654b10bdc3ccf"
ROTATIONS = {"auto": "Auto (metadata / local AI)", "0": "No additional rotation",
             "90": "90° clockwise", "180": "180°", "270": "90° counterclockwise"}


def input_tensor(image, size=280):
    """Upstream aspect-preserving bicubic letterbox and ImageNet normalization."""
    scale = size/max(image.size)
    width, height = (max(1, round(side*scale)) for side in image.size)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(image.resize((width, height), Image.Resampling.BICUBIC), ((size-width)//2, (size-height)//2))
    values = np.asarray(canvas, dtype=np.float32)/255
    values = (values-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def quarter_probabilities(output):
    probability = np.asarray(output)
    if (probability.shape != (1, 360) or not np.isfinite(probability).all()
            or probability.min() < 0 or not np.isclose(probability.sum(), 1, atol=.01)):
        raise UWCCError("Orientation model returned invalid probabilities.")
    sector = ((np.arange(360)+45)//90) % 4
    return {(-index*90) % 360: float(probability[0, sector == index].sum()) for index in range(4)}


def consensus(samples):
    votes = Counter()
    for sample in samples:
        probabilities = sample["probabilities"]
        best = max(probabilities, key=probabilities.get)
        if probabilities[best] >= .9:
            votes[best] += 1
    winner, agreement = votes.most_common(1)[0] if votes else (0, 0)
    mean = float(np.mean([item["probabilities"][winner] for item in samples])) if samples else 0.
    accepted = len(samples) == 10 and agreement >= 8 and mean >= .9
    direction = ROTATIONS[str(winner)]
    message = (f"AI orientation: {direction} ({agreement}/10 samples agree)." if accepted else
               f"AI orientation uncertain: suggested {direction} ({agreement}/10 samples agree). Choose rotation manually.")
    return {"correction_cw": winner if accepted else 0, "suggested_cw": winner,
            "status": "detected" if accepted else "uncertain", "agreement": agreement,
            "probability": mean, "message": message, "model": "RightWayUp Max INT8"}


def validate_decision(data):
    if data is None:
        return None
    if (not isinstance(data, dict) or type(data.get("correction_cw")) is not int
            or data["correction_cw"] not in (0, 90, 180, 270)
            or data.get("status") not in ("detected", "uncertain", "unavailable", "metadata", "legacy")
            or not isinstance(data.get("message"), str) or len(data["message"]) > 1000):
        raise UWCCError("Invalid saved video orientation.")
    if data["status"] != "detected" and data["correction_cw"] != 0:
        raise UWCCError("Unconfirmed video orientation must not rotate the clip.")
    return data


def oriented_info(info, clockwise):
    """Apply a right-angle correction relative to FFmpeg's metadata orientation."""
    if type(clockwise) is not int or clockwise not in (0, 90, 180, 270):
        raise UWCCError("Video rotation must be 0, 90, 180, or 270 degrees.")
    previous = getattr(info, "correction_rotation", 0)
    width, height, sar = info.width, info.height, info.sar
    if (previous in (90, 270)) != (clockwise in (90, 270)):
        width, height = height, width
        if sar not in ("N/A", "0:1"):
            value = Fraction(sar.replace(":", "/"))
            sar = f"{value.denominator}:{value.numerator}"
    return replace(info, width=width, height=height, sar=sar, correction_rotation=clockwise)


def detect(info, cancel=None, progress=lambda *_: None):
    from .media import read_frame
    raw_info = oriented_info(info, 0)
    if raw_info.rotation:
        return {"correction_cw": 0, "status": "metadata", "message": "Using the file's rotation metadata."}
    manifest = builtin_model("orientation")
    if not manifest:
        return {"correction_cw": 0, "status": "unavailable",
                "message": "No local orientation model. Use manual rotation or uwcc-models prepare-orientation."}
    if raw_info.duration*float(raw_info.fps) < 10:
        return {"correction_cw": 0, "status": "uncertain",
                "message": "Too few frames for ten-sample orientation detection. Choose rotation manually."}
    path = Path(manifest).parent/MODEL_FILE
    try:
        import onnxruntime as ort
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise UWCCError("Orientation model checksum mismatch")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        samples = []
        for index in range(10):
            if cancel and cancel.is_set():
                raise Cancelled("Orientation detection canceled.")
            progress(index/10, f"Detecting video orientation · sample {index+1}/10")
            seconds = min(raw_info.duration-1/float(raw_info.fps), raw_info.duration*(index+.5)/10)
            frame = read_frame(raw_info, max(0, seconds), 640, cancel)
            image = Image.fromarray(np.rint(np.clip(frame, 0, 1)*255).astype(np.uint8))
            probabilities = quarter_probabilities(session.run(None, {"image": input_tensor(image)})[0])
            samples.append({"time": seconds, "probabilities": probabilities})
        result = consensus(samples)
        progress(1, result["message"])
        return result
    except Cancelled:
        raise
    except Exception as exc:
        return {"correction_cw": 0, "status": "unavailable",
                "message": f"Orientation detection unavailable; use manual rotation. {str(exc)[:200]}"}


def project_info(project, info, cancel=None, progress=lambda *_: None):
    if project.grade.rotation != "auto":
        return oriented_info(info, int(project.grade.rotation))
    if project.orientation is None:
        project.orientation = detect(info, cancel, progress)
    return oriented_info(info, project.orientation["correction_cw"])
