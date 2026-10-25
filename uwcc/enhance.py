"""Automatic levels/illuminant/contrast and compact, learned color transforms.

Fits are regularized polynomial regressions, baked into RGB cubes. Neural
references are sampled during analysis, never independently on each video frame.
"""
from functools import lru_cache
import json

import numpy as np

from .color import bt709_decode, bt709_encode, gamut_compress, LUMA
from .errors import UWCCError


def features(rgb):
    r, g, b = np.moveaxis(rgb, -1, 0)
    return np.stack([np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b,
                     r**3, g**3, b**3, r*r*g, r*r*b, g*g*r, g*g*b,
                     b*b*r, b*b*g, r*g*b], axis=-1)


def fit_colors(source, target):
    """Learn a smooth RGB mapping; no geometry or image detail can be generated."""
    x = np.asarray(source, np.float32).reshape(-1, 3)
    y = np.asarray(target, np.float32).reshape(-1, 3)
    if x.shape != y.shape or len(x) < 100 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise UWCCError("Reference fitting needs matching, finite RGB samples.")
    # Bound memory independently of source resolution and balance frequent colors.
    stride = max(1, len(x)//40000)
    x, y = np.clip(x[::stride], 0, 1), np.clip(y[::stride], 0, 1)
    cells = np.minimum((x*15).astype(int), 14)
    bins = cells[:, 0]*225+cells[:, 1]*15+cells[:, 2]
    counts = np.bincount(bins, minlength=3375)
    base_weight = 1/np.sqrt(counts[bins])
    base_weight /= np.mean(base_weight)
    design = features(x).astype(np.float64)
    prior = np.zeros((20, 3))
    prior[1:4] = np.eye(3)
    ridge = np.eye(20)*max(.01, len(x)*.00002)
    weights = base_weight
    for _ in range(3):
        coef = np.linalg.solve(design.T @ (design*weights[:, None])+ridge,
                               design.T @ (y*weights[:, None])+ridge @ prior)
        residual = np.linalg.norm(design @ coef-y, axis=1)
        weights = base_weight*np.minimum(1, .06/np.maximum(residual, 1e-6))
    error = float(np.mean(np.abs(np.clip(design @ coef, 0, 1)-y)))
    return coef.astype(np.float32).tolist(), error


def estimate_recipe(encoded, white_balance=True):
    """Gray-edge illuminant estimate followed by robust levels and midtone lift."""
    linear = bt709_decode(np.asarray(encoded, np.float32))
    # Gradient norms use textured surfaces, rather than the mean blue water.
    dx = np.abs(linear[:, 1:]-linear[:, :-1]).reshape(-1, 3)
    dy = np.abs(linear[1:]-linear[:-1]).reshape(-1, 3)
    gradients = np.concatenate([dx, dy])
    # Reject rare clipping/colored-edge outliers before a Minkowski p=6 norm.
    cap = np.percentile(gradients, 98, axis=0)
    edge = np.mean(np.minimum(gradients, cap)**6, axis=0)**(1/6)
    white = np.percentile(linear.reshape(-1, 3), 95, axis=0)
    illuminant = np.maximum(edge, .0001)**.55 * np.maximum(white, .0001)**.45
    neutral = np.exp(np.mean(np.log(illuminant)))
    gains = np.clip((neutral/illuminant)**.7, .35, 5)
    # Never amplify a virtually absent signal into red noise.
    if white[0] < .002:
        gains[0] = min(gains[0], 1.8)
    if not white_balance:
        gains = np.ones(3, np.float32)
    balanced = linear*gains
    luminance = np.maximum(balanced @ LUMA, 0)
    black = min(float(np.percentile(luminance, .5)), .08)
    white_level = max(float(np.percentile(luminance, 99.5)), black+.35)
    median = np.clip((np.median(luminance)-black)/(white_level-black), .03, .9)
    gamma = float(np.clip(np.log(.28)/np.log(median), .65, 1.25))
    return {"gain": gains.tolist(), "black": black, "white": white_level,
            "gamma": gamma, "vibrance": .1, "contrast": 1.06}


def apply_recipe(encoded, recipe):
    rgb = bt709_decode(encoded)*np.array(recipe["gain"], np.float32)
    rgb = np.maximum(rgb-recipe["black"], 0)/(recipe["white"]-recipe["black"])
    y = np.maximum(rgb @ LUMA, 1e-7)
    target = y**recipe["gamma"]
    # A bounded S curve increases contrast without a hard highlight threshold.
    bounded = np.clip(target, 0, 1)
    target = bounded+(recipe["contrast"]-1)*(bounded-.5)*4*bounded*(1-bounded)
    rgb *= (target/y)[..., None]
    y = rgb @ LUMA
    chroma = rgb.max(axis=-1)-rgb.min(axis=-1)
    vibrance = 1+recipe["vibrance"]*(1-np.clip(chroma, 0, 1))
    rgb = y[..., None]+(rgb-y[..., None])*vibrance[..., None]
    return np.clip(bt709_encode(gamut_compress(rgb)), 0, 1).astype(np.float32)


@lru_cache(maxsize=8)
def _cube(serialized, size=33):
    data = json.loads(serialized)
    v = np.linspace(0, 1, size, dtype=np.float32)
    rgb = np.stack(np.meshgrid(v, v, v, indexing="ij"), axis=-1)
    if "encoded_fit" in data:
        result = features(rgb) @ np.asarray(data["encoded_fit"], np.float32)
    else:
        result = apply_recipe(rgb, data["recipe"])
    return np.ascontiguousarray(np.clip(result, 0, 1), dtype=np.float32)


def cube_at(analysis, seconds):
    import bisect
    if "reference_fit" in analysis:
        return _cube(json.dumps({"encoded_fit": analysis["reference_fit"]}))
    samples = analysis["samples"]
    i = max(0, bisect.bisect_right([s["time"] for s in samples], seconds)-1)
    a, b = samples[i], samples[min(i+1, len(samples)-1)]
    if not ("encoded_fit" in a or "recipe" in a):
        return None
    def cube(item):
        return _cube(json.dumps({k: item[k] for k in ("encoded_fit", "recipe") if k in item}, sort_keys=True))
    left = cube(a)
    if a["scene"] != b["scene"] or a["time"] == b["time"]:
        return left
    mix = np.float32(np.clip((seconds-a["time"])/(b["time"]-a["time"]), 0, 1))
    return left*(1-mix)+cube(b)*mix


def validate_fit(coef):
    try:
        a = np.asarray(coef, dtype=float)
    except (TypeError, ValueError) as exc:
        raise UWCCError("Invalid learned color transform.") from exc
    if a.shape != (20, 3) or not np.isfinite(a).all() or np.abs(a).max() > 256:
        raise UWCCError("Invalid learned color transform.")


def validate_recipe(recipe):
    try:
        gain = np.asarray(recipe["gain"], dtype=float)
        valid = gain.shape == (3,) and np.isfinite(gain).all() and np.all((gain >= .1) & (gain <= 8))
        valid &= all(np.isfinite(recipe[k]) for k in ("black", "white", "gamma", "vibrance", "contrast"))
        valid &= 0 <= recipe["black"] < recipe["white"] <= 20
        valid &= .2 <= recipe["gamma"] <= 3 and 0 <= recipe["vibrance"] <= 1 and .5 <= recipe["contrast"] <= 2
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise UWCCError("Invalid automatic levels/white balance recipe.")
