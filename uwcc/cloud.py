"""Opt-in cloud grading advice. Credentials never enter projects or presets."""
import base64
from io import BytesIO
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .errors import Cancelled, UWCCError
from .project import CONTROLS, clone, defaults, valid_value

PROVIDERS = {"openai": ("https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
             "anthropic": ("https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY")}
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "values": {"type": "object", "additionalProperties": False,
               "properties": {k: {"type": "number", "minimum": v[1], "maximum": v[2]} for k, v in CONTROLS.items()},
               "required": list(CONTROLS)},
    "denoise": {"type": "number", "minimum": 0, "maximum": 10},
    "stabilize": {"type": "boolean"}, "summary": {"type": "string"}},
    "required": ["values", "denoise", "stabilize", "summary"]}


def validate_advice(data):
    if not isinstance(data, dict) or set(data) != set(SCHEMA["required"]):
        raise UWCCError("Cloud advice has an invalid structure.")
    if not isinstance(data["values"], dict) or set(data["values"]) != set(CONTROLS):
        raise UWCCError("Cloud advice contains missing or unknown grading controls.")
    values = {name: valid_value(name, value) for name, value in data["values"].items()}
    noise = data["denoise"]
    if isinstance(noise, bool) or not isinstance(noise, (float, int)) or not np.isfinite(noise) or not 0 <= noise <= 10:
        raise UWCCError("Cloud advice has invalid denoise strength.")
    if type(data["stabilize"]) is not bool or not isinstance(data["summary"], str) or len(data["summary"]) > 4000:
        raise UWCCError("Cloud advice has invalid stabilization or explanation.")
    return {"values": values, "denoise": float(noise), "stabilize": data["stabilize"], "summary": data["summary"]}


def request_advice(provider, model, images, preference="Natural, balanced, vivid underwater footage", cancel=None):
    if provider not in PROVIDERS or not isinstance(model, str) or not model.strip():
        raise UWCCError("Choose a provider and enter a vision-capable API model ID.")
    endpoint, variable = PROVIDERS[provider]
    key = os.environ.get(variable)
    if not key:
        raise UWCCError(f"Set {variable} in the environment to use this provider.")
    if cancel and cancel.is_set():
        raise Cancelled("Cloud advice canceled.")
    encoded = []
    for array in images:
        image = Image.fromarray(np.uint8(np.clip(array, 0, 1)*255))
        image.thumbnail((640, 640))
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=85)
        encoded.append(base64.b64encode(stream.getvalue()).decode("ascii"))
    if not 1 <= len(encoded) <= 3:
        raise UWCCError("Cloud advice requires one to three proxy frames.")
    prompt = ("Act as an underwater colorist. Inspect these SDR proxy frames after automatic correction, before manual controls. "
              "Propose absolute manual controls for the provided schema; strength should stay 100. "
              "Improve levels, white balance, natural contrast and vibrance, preserving blue water, skin, animals, and highlights. "
              "Do not invent detail. Still images cannot establish camera shake or temporal noise; do not enable stabilization "
              "based solely on these images. Explain uncertainty. Neutral controls are " + json.dumps(defaults()) +
              ". User preference: " + preference[:2000])
    if provider == "openai":
        payload = {"model": model, "store": False, "max_output_tokens": 1500,
                   "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]+
                              [{"type": "input_image", "image_url": "data:image/jpeg;base64,"+x, "detail": "auto"} for x in encoded]}],
                   "text": {"format": {"type": "json_schema", "name": "underwater_grade", "schema": SCHEMA, "strict": True}}}
        headers = {"Authorization": "Bearer "+key}
    else:
        payload = {"model": model, "max_tokens": 1500,
                   "messages": [{"role": "user", "content": [
                       {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": x}} for x in encoded]+
                       [{"type": "text", "text": prompt}]}],
                   "tools": [{"name": "propose_grade", "description": "Return bounded grading advice", "input_schema": SCHEMA}],
                   "tool_choice": {"type": "tool", "name": "propose_grade"}}
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    request = Request(endpoint, data=json.dumps(payload).encode(),
                      headers=headers | {"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise UWCCError("Cloud response exceeds the size limit.")
        result = json.loads(raw)
        if provider == "openai":
            texts = [c["text"] for item in result.get("output", []) for c in item.get("content", []) if c.get("type") == "output_text"]
            data = json.loads("".join(texts))
        else:
            data = next(c["input"] for c in result.get("content", []) if c.get("type") == "tool_use" and c.get("name") == "propose_grade")
    except HTTPError as exc:
        raise UWCCError(f"Cloud provider returned HTTP {exc.code}; check API key, model access, and quota.") from None
    except (URLError, TimeoutError, OSError):
        raise UWCCError("Cloud request failed; check your network connection.") from None
    except (ValueError, TypeError, KeyError, StopIteration):
        raise UWCCError("Cloud provider did not return a valid grading proposal.") from None
    if cancel and cancel.is_set():
        raise Cancelled("Cloud advice canceled.")
    return validate_advice(data)


def advise_project(project, provider, model, preference="Natural, balanced, vivid underwater footage", cancel=None, progress=lambda *_: None):
    from .media import probe, read_frame, prepare_frame
    from .color import correct
    info = probe(project.source, cancel)
    from .orientation import project_info
    info = project_info(project, info, cancel, progress)
    grade = clone(project.grade)
    grade.values, grade.keyframes = defaults(), {}
    if grade.input_profile == "auto":
        raise UWCCError("Confirm the input color profile before requesting cloud advice.")
    frames = []
    for fraction in (.15, .5, .85):
        time = min(info.duration-1/float(info.fps), info.duration*fraction)
        raw = prepare_frame(read_frame(info, time, 640, cancel), info, grade.input_profile)
        frames.append(correct(raw, grade, time, project.analysis, preview=True))
    progress(.5, "Sending three proxy frames for cloud grading advice…")
    result = request_advice(provider, model, frames, preference, cancel)
    progress(1, "Cloud proposal ready for review")
    return result


def apply_advice(project, advice):
    advice = validate_advice(advice)
    result = clone(project)
    result.grade.values = advice["values"]
    result.grade.keyframes.clear()
    result.grade.denoise = advice["denoise"]
    result.grade.stabilize = advice["stabilize"]
    return result
