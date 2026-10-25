"""Shot-aware adaptive analysis, confidence warnings, and temporal smoothing."""

from __future__ import annotations

import math
import threading
from contextlib import closing

import numpy as np

from .color import decode, gain_matrix, identity_lut, srgb_decode, to_reference, TO_AP1, encode, bt709_encode
from .errors import Cancelled, UWCCError
from .media import FrameReader, frame_timestamps, prepare_frame


def estimate(ap1, filter_hint="auto"):
    from .color import FROM_AP1, LUMA
    rgb = np.maximum(ap1 @ FROM_AP1["rec709"].T, 0)
    pixels = rgb.reshape(-1, 3)
    luminance = pixels @ LUMA
    good = pixels[(luminance > np.percentile(luminance, 15)) &
                  (luminance < np.percentile(luminance, 95)) & (pixels.max(axis=1) < 1)]
    if len(good) < 32:
        return np.ones(3), 0., .2, ["Very little usable midtone signal; correction is limited."]
    means = np.maximum(np.mean(good, axis=0), .0001)
    white = np.maximum(np.percentile(good, 90, axis=0), .001)
    balance = means**.7*white**.3
    target = np.exp(np.mean(np.log(balance)))
    gains = np.clip((target/balance)**.75, .4, 4)
    warnings = []
    confidence = .75
    if means[0] < .008 or means[0]/max(means[1], means[2]) < .06:
        gains[0] = min(gains[0], 1.8)
        confidence = .35
        warnings.append("Red channel has little usable signal; red recovery is limited.")
    if np.mean(pixels[:, 0] > .99) > .005:
        gains[0] = min(gains[0], 1.)
        warnings.append("Red highlights are clipped in the source.")
    if filter_hint == "red":
        gains[0] = min(gains[0], 1.4)
    elif filter_hint == "magenta":
        gains[[0, 2]] = np.minimum(gains[[0, 2]], 1.4)
    # Adapt gains from linear Rec.709 into a bounded AP1 diagonal approximation.
    exposure = float(np.clip(math.log2(.22/max(float(np.median(luminance)), .02)), -.75, .75)*.35)
    return gains.astype(np.float32), exposure, confidence, warnings


def fit_reference(source_ap1, target_srgb):
    """Robust regularized color matrix, then sample that fit into a smooth LUT.

    The matrix cannot add geometry or texture. Its residual is used to reject
    reference images that a global color transform cannot represent safely.
    """
    x = source_ap1.reshape(-1, 3)
    y = (srgb_decode(target_srgb) @ TO_AP1["rec709"].T).reshape(-1, 3)
    valid = (x.min(axis=1) > .002) & (x.max(axis=1) < .95) & (y.max(axis=1) < 1)
    x, y = x[valid][::3], y[valid][::3]
    if len(x) < 32:
        return np.eye(3), .0
    weights = np.ones(len(x))
    matrix = np.eye(3)
    for _ in range(4):
        ridge = np.eye(3)*max(.01, len(x)*.0005)
        matrix = np.linalg.solve(x.T @ (x*weights[:, None])+ridge,
                                 x.T @ (y*weights[:, None])+ridge)
        residual = np.linalg.norm(x @ matrix-y, axis=1)
        weights = np.minimum(1, .04/np.maximum(residual, 1e-6))
    matrix = np.clip(matrix, -.35, 4)
    confidence = float(np.clip(1-np.median(np.linalg.norm(x @ matrix-y, axis=1))*8, 0, 1))
    return matrix, confidence


def validate_analysis(data):
    if not isinstance(data, dict) or data.get("version") != 1:
        raise UWCCError("Unsupported automatic-analysis version; re-analyze the clip.")
    samples = data.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 100_000:
        raise UWCCError("Invalid automatic-analysis samples.")
    previous = -1
    from .enhance import validate_fit, validate_recipe
    if "reference_fit" in data:
        validate_fit(data["reference_fit"])
    for item in samples:
        time = item.get("time", -1)
        if not isinstance(time, (float, int)) or not math.isfinite(time) or time < previous:
            raise UWCCError("Analysis timestamps must be finite and ordered.")
        previous = time
        if "recipe" in item:
            validate_recipe(item["recipe"])
        if "encoded_fit" in item:
            validate_fit(item["encoded_fit"])
        gain = np.asarray(item.get("gain"), dtype=float)
        if gain.shape != (3,) or not np.isfinite(gain).all() or np.any((gain < .1) | (gain > 8)):
            raise UWCCError("Invalid automatic channel gain.")
        if not math.isfinite(item.get("exposure", float("nan"))) or abs(item["exposure"]) > 3:
            raise UWCCError("Invalid automatic exposure.")
        if "matrix" in item:
            matrix = np.asarray(item["matrix"], dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.abs(matrix).max() > 8:
                raise UWCCError("Invalid model transform.")


def histogram(frame):
    return np.concatenate([np.histogram(frame[..., i], bins=16, range=(0, 1), density=False)[0]
                           for i in range(3)]).astype(float)/(frame.shape[0]*frame.shape[1])


def stabilize(samples):
    for scene in sorted({s["scene"] for s in samples}):
        group = [s for s in samples if s["scene"] == scene]
        parameters = np.array([s["gain"]+[s["exposure"]]+np.array(s.get("matrix", np.eye(3))).ravel().tolist()
                               for s in group], dtype=float)
        # Symmetric timestamp-aware filtering, confined to the same shot.
        for direction in (range(1, len(group)), range(len(group)-2, -1, -1)):
            for i in direction:
                neighbor = i-1 if direction.step > 0 else i+1
                dt = abs(group[i]["time"]-group[neighbor]["time"])
                weight = 1-math.exp(-dt)
                parameters[i] = weight*parameters[i]+(1-weight)*parameters[neighbor]
        for row, item in zip(parameters, group):
            item["gain"], item["exposure"] = row[:3].tolist(), float(row[3])
            if "matrix" in item:
                item["matrix"] = row[4:].reshape(3, 3).tolist()
        # Smooth all new grading parameters within the shot as well.
        for key in ("recipe", "encoded_fit"):
            if not all(key in item for item in group):
                continue
            if key == "recipe":
                fields = ["black", "white", "gamma", "vibrance", "contrast"]
                extra = np.array([item[key]["gain"]+[item[key][f] for f in fields] for item in group])
            else:
                extra = np.array([item[key] for item in group])
            for direction in (range(1, len(group)), range(len(group)-2, -1, -1)):
                for i in direction:
                    neighbor = i-1 if direction.step > 0 else i+1
                    weight = 1-math.exp(-abs(group[i]["time"]-group[neighbor]["time"]))
                    extra[i] = weight*extra[i]+(1-weight)*extra[neighbor]
            for item, row in zip(group, extra):
                item[key] = ({"gain": row[:3].tolist()} | dict(zip(fields, row[3:].tolist()))
                             if key == "recipe" else row.tolist())


def analyze(info, grade, cancel=None, progress=lambda *_: None):
    if grade.input_profile == "auto":
        raise UWCCError("Confirm the input color profile before analyzing.")
    cancel = cancel or threading.Event()
    model = None
    warnings = set()
    from .neural import builtin_model
    manifest = grade.model_manifest or (builtin_model("enhance") if grade.auto_method == "ai" else None)
    if manifest:
        from .ml import ReferenceModel
        try:
            model = ReferenceModel(manifest)
        except UWCCError as exc:
            if grade.model_manifest:
                raise
            warnings.add(f"Local AI unavailable; using adaptive levels/white balance: {exc}")
    else:
        warnings.add("Classical engine: no verified ONNX model selected.")
    samples, scene = [], 0
    previous_hist, previous_frame, last_sample = None, None, -10.
    last_frame, last_time = None, 0

    def sample(frame, time):
        linear = decode(prepare_frame(frame, info, grade.input_profile), grade.input_profile)
        gains, exposure, confidence, messages = estimate(linear, grade.filter_hint)
        warnings.update(messages)
        item = {"time": float(time), "scene": scene, "gain": gains.tolist(), "exposure": exposure,
                "confidence": confidence}
        from .enhance import estimate_recipe, fit_colors, apply_recipe
        display = encode(linear, "rec709", grade.input_profile != "rec709")
        if grade.auto_method != "legacy":
            item["recipe"] = estimate_recipe(display)
        if model:
            try:
                target = model.predict(to_reference(linear))
                target_display = bt709_encode(srgb_decode(target))
                # The network handles color balance; finish with levels/contrast
                # without estimating a second illuminant over its corrected image.
                target_display = apply_recipe(target_display, estimate_recipe(target_display, white_balance=False))
                coefficients, error = fit_colors(display, target_display)
                if error < .08:
                    item["encoded_fit"] = coefficients
                    item["fit_mae"] = error
                matrix, fit = fit_reference(linear, target)
                if fit >= .4:
                    blend = np.clip((fit-.4)/.3, 0, 1)
                    baseline = gain_matrix(gains, exposure)
                    item["matrix"] = (matrix*blend+baseline*(1-blend)).tolist()
                    item["confidence"] = min(confidence, fit)
                else:
                    warnings.add("Model reference had a poor color fit; using classical correction.")
                if model.warning:
                    warnings.add(model.warning)
            except Exception as exc:
                warnings.add(f"Model inference failed; using classical correction: {exc}")
        return item

    with closing(frame_timestamps(info, cancel)) as timestamps, FrameReader(info, max_width=320, cancel=cancel) as reader:
        for frame in reader:
            if cancel.is_set():
                raise Cancelled("Analysis canceled.")
            try:
                time = next(timestamps)
            except StopIteration as exc:
                raise UWCCError("Decoder and timestamp stream disagree.") from exc
            current_hist = histogram(frame)
            distance = np.abs(current_hist-previous_hist).sum()/6 if previous_hist is not None else 0
            difference = float(np.mean(np.abs(frame-previous_frame))) if previous_frame is not None else 0
            cut = distance > .25 and difference > .12
            if cut:
                if last_frame is not None and last_time > samples[-1]["time"]:
                    samples.append(sample(last_frame, last_time))
                scene += 1
            if not samples or cut or time-last_sample >= .5:
                samples.append(sample(frame, time))
                last_sample = time
                progress(min(.99, time/info.duration), f"Analyzing shot {scene+1} · {time:.1f}s")
            previous_hist, previous_frame = current_hist, frame
            last_frame, last_time = frame, time
    if not samples:
        raise UWCCError("No frames were available for analysis.")
    if last_time > samples[-1]["time"]:
        samples.append(sample(last_frame, last_time))
    stabilize(samples)
    result = {"version": 1, "engine": model.name+" + automatic levels" if model else (
              "Adaptive levels + gray-edge white balance" if grade.auto_method != "legacy" else "Classical adaptive balance"),
              "input_profile": grade.input_profile, "filter_hint": grade.filter_hint,
              "warnings": sorted(warnings), "samples": samples, "shots": scene+1,
              "confidence": float(np.mean([s["confidence"] for s in samples]))}
    validate_analysis(result)
    progress(1., "Analysis complete")
    return result


def learn_reference(info, grade, reference, cancel=None, progress=lambda *_: None):
    """Learn a global grade from a matching, unstabilized corrected video."""
    from .enhance import fit_colors
    from .media import probe, read_frame
    from .project import fingerprint
    from pathlib import Path
    target = probe(reference, cancel)
    if (target.width, target.height) != (info.width, info.height) or abs(target.duration-info.duration) > .1:
        raise UWCCError("The corrected reference must match the source dimensions and duration.")
    if grade.input_profile != "rec709" or grade.output_profile != "rec709":
        raise UWCCError("Reference learning currently requires Rec.709 input and SDR output.")
    learned, errors = [], []
    times = np.linspace(0, max(0, info.duration-1/float(info.fps)), min(60, max(3, int(info.duration*2)+1)))
    for i, time in enumerate(times):
        if cancel and cancel.is_set():
            raise Cancelled("Reference learning canceled.")
        raw = read_frame(info, float(time), 320, cancel)
        corrected = read_frame(target, float(time), 320, cancel)
        # Reject unrelated framing: color edits preserve the positions of edges.
        edge_a = np.diff(raw.mean(axis=-1), axis=1).ravel()
        edge_b = np.diff(corrected.mean(axis=-1), axis=1).ravel()
        correlation = float(np.corrcoef(edge_a, edge_b)[0, 1])
        if not np.isfinite(correlation) or correlation < .3:
            raise UWCCError("Reference frames do not align. Use an untrimmed, unstabilized correction of the same clip.")
        coefficients, error = fit_colors(raw[::2, ::2], corrected[::2, ::2])
        learned.append({"time": float(time), "scene": 0, "gain": [1., 1., 1.], "exposure": 0.,
                        "confidence": 1-error, "encoded_fit": coefficients})
        errors.append(error)
        progress((i+1)/len(times)*.9, f"Learning reference color · sample {i+1}/{len(times)}")
    error = float(np.mean(errors))
    if error > .08:
        raise UWCCError("Reference color could not be represented reliably by a global grade.")
    result = {"version": 1, "engine": "Learned corrected-reference grade",
              "input_profile": grade.input_profile, "filter_hint": grade.filter_hint,
              "warnings": ["Reference interpreted as SDR Rec.709; verify its color profile."],
              "samples": learned,
              "shots": 1, "confidence": 1-error,
              "reference_mae": error, "reference_fingerprint": fingerprint(Path(reference))}
    validate_analysis(result)
    progress(1, "Reference grade learned")
    return result


def transform_at(data, seconds):
    import bisect
    samples = data["samples"]
    times = [s["time"] for s in samples]
    i = max(0, bisect.bisect_right(times, seconds)-1)
    a, b = samples[i], samples[min(i+1, len(samples)-1)]
    mix = 0 if a["scene"] != b["scene"] or a["time"] == b["time"] else np.clip((seconds-a["time"])/(b["time"]-a["time"]), 0, 1)
    gain = np.array(a["gain"])*(1-mix)+np.array(b["gain"])*mix
    exposure = a["exposure"]*(1-mix)+b["exposure"]*mix
    lut = None
    if "matrix" in a or "matrix" in b:
        ma = np.array(a.get("matrix", gain_matrix(a["gain"], a["exposure"])))
        mb = np.array(b.get("matrix", gain_matrix(b["gain"], b["exposure"])))
        matrix = ma*(1-mix)+mb*mix
        lut = np.maximum(identity_lut() @ matrix, 0).astype(np.float32)
    return gain, exposure, lut
