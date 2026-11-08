"""Float32 color transforms and a shared CPU correction graph.

Transfer definitions: ITU-R BT.709, BT.2100 and Sony S-Log white papers.
Internal RGB is ACEScg (AP1, D60); model references are encoded sRGB.
"""

from __future__ import annotations

import numpy as np

from .errors import UWCCError

try:
    from . import _native
except ImportError:
    _native = None

RGB_XYZ = np.array([[.4123908, .35758434, .18048079],
                    [.21263901, .71516868, .07219232],
                    [.01933082, .11919478, .95053215]])
AP1_XYZ = np.array([[.66245418, .13400421, .15618769],
                    [.27222872, .67408177, .05368952],
                    [-.00557465, .00406073, 1.0103391]])
BT2020_XYZ = np.array([[.63695805, .1446169, .16888098],
                       [.26270021, .67799807, .05930172], [0, .02807269, 1.06098506]])
SGAMUT_XYZ = np.array([[.70648271, .12880105, .11517216],
                       [.27097967, .78660641, -.05758608],
                       [-.00967785, .00460004, 1.09413556]])
SGAMUT3CINE_XYZ = np.array([[.59908392, .24892552, .10244649],
                            [.21507582, .88506850, -.10014432],
                            [-.03206585, -.02765839, 1.14878199]])
BRADFORD = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])
D65, D60 = np.array([.95045593, 1., 1.08905775]), np.array([.95264607, 1., 1.00882518])
D65_D60 = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ D60) / (BRADFORD @ D65)) @ BRADFORD
TO_AP1 = {"rec709": np.linalg.inv(AP1_XYZ) @ D65_D60 @ RGB_XYZ,
          "slog2": np.linalg.inv(AP1_XYZ) @ D65_D60 @ SGAMUT_XYZ,
          "slog3": np.linalg.inv(AP1_XYZ) @ D65_D60 @ SGAMUT3CINE_XYZ,
          "hlg": np.linalg.inv(AP1_XYZ) @ D65_D60 @ BT2020_XYZ,
          "pq": np.linalg.inv(AP1_XYZ) @ D65_D60 @ BT2020_XYZ}
FROM_AP1 = {key: np.linalg.inv(value) for key, value in TO_AP1.items()}
TO_AP1 = {key: value.astype(np.float32) for key, value in TO_AP1.items()}
FROM_AP1 = {key: value.astype(np.float32) for key, value in FROM_AP1.items()}
LUMA = np.array([.21263901, .71516868, .07219232], dtype=np.float32)


def bt709_decode(x):
    x = np.maximum(x, 0)
    return np.where(x < .081, x / 4.5, np.power((x + .099) / 1.099, 1 / .45))


def bt709_encode(x):
    x = np.maximum(x, 0)
    return np.where(x < .018, x * 4.5, 1.099 * np.power(x, .45) - .099)


def srgb_encode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)


def srgb_decode(x):
    x = np.maximum(x, 0)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def pq_decode(x):
    m1, m2, c1, c2, c3 = 2610/16384, 2523/32, 3424/4096, 2413/128, 2392/128
    p = np.maximum(x, 0) ** (1/m2)
    return (np.maximum(p-c1, 0) / np.maximum(c2-c3*p, 1e-8)) ** (1/m1) * 10000 / 100


def pq_encode(x):
    m1, m2, c1, c2, c3 = 2610/16384, 2523/32, 3424/4096, 2413/128, 2392/128
    p = (np.maximum(x, 0) * 100 / 10000) ** m1
    return ((c1+c2*p)/(1+c3*p)) ** m2


def hlg_decode(x):
    a, b, c = .17883277, .28466892, .55991073
    return np.where(x <= .5, x*x/3, (np.exp((x-c)/a)+b)/12)


def hlg_encode(x):
    a, b, c = .17883277, .28466892, .55991073
    return np.where(x <= 1/12, np.sqrt(np.maximum(3*x, 0)), a*np.log(np.maximum(12*x-b, 1e-8))+c)


def decode(x, profile):
    x = np.asarray(x, dtype=np.float32)
    if profile == "rec709":
        rgb = bt709_decode(x)
    elif profile == "slog2":
        # Published code values include video-range scaling. Reference gray
        # .18 maps to .3395325246 (approximately 347/1023).
        signal = (x*1023-64)/876
        rgb = np.where(signal >= .03000122285,
                       10 ** ((signal-.616596-.03)/.432699)-.037584,
                       (signal-.03000122285)/5) * .9*219/155
    elif profile == "slog3":
        code = x * 1023
        rgb = np.where(code >= 171.2102946929,
                       10 ** ((code-420)/261.5) * .19 - .01,
                       (code-95)*.01125/(171.2102946929-95))
    elif profile == "pq":
        rgb = pq_decode(x)
    elif profile == "hlg":
        scene = hlg_decode(x)
        y = np.maximum(scene @ np.array([.2627, .678, .0593]), 1e-8)
        rgb = scene * y[..., None] ** .2 * 10  # 1000-nit display, 100-nit working white
    else:
        raise UWCCError("Select the input color profile before processing.")
    return np.asarray(rgb @ TO_AP1[profile].T, dtype=np.float32)


def to_reference(ap1):
    """Model domain: sRGB encoded, with highlights compressed for analysis only."""
    rgb = np.maximum(ap1 @ FROM_AP1["rec709"].T, 0)
    return np.clip(srgb_encode(rgb), 0, 1).astype(np.float32)


def tone_map(rgb):
    # Luminance-preserving Reinhard with 4x scene white, normalized at diffuse white.
    y = np.maximum(rgb @ LUMA, 1e-8)
    mapped = y * (1 + y / 16) / (1+y) * 1.6
    return rgb * (mapped / y)[..., None]


def gamut_compress(rgb):
    """Desaturate toward a bounded neutral until all channels are in gamut."""
    y = np.clip(rgb @ LUMA, 0, 1)
    delta = rgb-y[..., None]
    room = np.where(delta > 0, (1-y[..., None])/np.maximum(delta, 1e-8),
                    -y[..., None]/np.minimum(delta, -1e-8))
    scale = np.minimum(1, np.min(room, axis=-1))
    return np.clip(y[..., None] + delta * scale[..., None], 0, 1)


def encode(ap1, profile, tone=False):
    rgb = ap1 @ FROM_AP1["rec709" if profile == "rec709" else profile].T
    if profile == "rec709":
        if _native is not None:
            rgb = np.ascontiguousarray(rgb, dtype=np.float32)
            result = np.empty_like(rgb)
            _native.encode709(rgb, result, tone)
            return result
        if tone:
            rgb = tone_map(np.maximum(rgb, 0))
        return np.clip(bt709_encode(gamut_compress(rgb)), 0, 1).astype(np.float32)
    rgb = np.clip(rgb, 0, 10)
    if profile == "pq":
        return np.clip(pq_encode(rgb), 0, 1).astype(np.float32)
    y = np.maximum(rgb @ np.array([.2627, .678, .0593]), 1e-8)
    scene_y = (y/10) ** (1/1.2)
    scene = rgb * (scene_y/y)[..., None]
    return np.clip(hlg_encode(scene), 0, 1).astype(np.float32)


def box_blur(x, radius):
    radius = max(1, int(radius))
    out = x
    for axis in (0, 1):
        pads = [(0, 0)] * out.ndim
        pads[axis] = (radius, radius)
        padded = np.pad(out, pads, mode="edge")
        prefix = np.cumsum(padded, axis=axis, dtype=np.float32)
        shape = list(prefix.shape)
        shape[axis] = 1
        prefix = np.concatenate([np.zeros(shape, dtype=np.float32), prefix], axis=axis)
        n = 2*radius+1
        left, right = [slice(None)] * out.ndim, [slice(None)] * out.ndim
        left[axis], right[axis] = slice(None, -n), slice(n, None)
        out = (prefix[tuple(right)] - prefix[tuple(left)]) / n
    return out


def manual(ap1, p):
    if all(p[k] == v for k, v in {
        "temperature": 0, "tint": 0, "exposure": 0, "contrast": 0,
        "shadows": 0, "highlights": 0, "black_point": 0,
        "hue": 0, "saturation": 100, "dehaze": 0,
    }.items()):
        return np.maximum(ap1, 0)
    # Use a Bradford chromatic adaptation in XYZ, converted to the working space.
    temp, tint = p["temperature"] / 100, p["tint"] / 100
    cone_gains = np.diag(np.exp([temp*.25-tint*.08, tint*.16, -temp*.25-tint*.08]))
    matrix = np.linalg.inv(AP1_XYZ) @ np.linalg.inv(BRADFORD) @ cone_gains @ BRADFORD @ AP1_XYZ
    rgb = np.maximum(ap1 @ matrix.T.astype(np.float32), 0) * 2 ** p["exposure"]
    rgb = rgb @ FROM_AP1["rec709"].T
    y = np.maximum(rgb @ LUMA, 1e-7)
    # Monotonic power curves; neutral controls evaluate to identity.
    contrast = 2 ** (p["contrast"]/100)
    target = .18 * np.maximum(y/.18, 1e-7) ** contrast
    shadow = np.clip(1-y/.5, 0, 1)**2
    high = np.clip(y/.8, 0, 1)**2
    target *= 2 ** (p["shadows"]/100 * shadow + p["highlights"]/100 * high)
    target = np.maximum(0, target - p["black_point"] / 100 * .04)
    rgb *= (target/y)[..., None]
    y = rgb @ LUMA
    # Rotation in an orthogonal chroma plane; neutral colors remain neutral.
    u, v = rgb[..., 0]-rgb[..., 1], rgb[..., 2]-rgb[..., 1]
    angle = np.deg2rad(p["hue"])
    u1 = (u*np.cos(angle)-v*np.sin(angle))*p["saturation"]/100
    v1 = (u*np.sin(angle)+v*np.cos(angle))*p["saturation"]/100
    green = y-LUMA[0]*u1-LUMA[2]*v1
    rgb = np.stack([green+u1, green, green+v1], axis=-1)
    if p["dehaze"]:
        # Bounded edge-preserving local contrast; no depth or texture synthesis.
        base = box_blur(rgb, max(2, rgb.shape[1]//150))
        detail = rgb-base
        rgb += np.clip(detail, -.12, .12) * p["dehaze"]/100
    return np.asarray(rgb @ TO_AP1["rec709"].T, dtype=np.float32)


def apply_lut(rgb, lut, uniform=False):
    """Tetrahedral interpolation of a log-shaped ACEScg RGB cube."""
    if _native is not None:
        rgb = np.ascontiguousarray(rgb, dtype=np.float32)
        lut = np.ascontiguousarray(lut, dtype=np.float32)
        n = lut.shape[0]
        if lut.shape != (n, n, n, 3) or not 2 <= n <= 256 or rgb.shape[-1] != 3:
            raise ValueError("Expected RGB pixels and a cubic RGB LUT of size 2–256")
        result = np.empty_like(rgb)
        _native.apply_lut(rgb, lut, result, n, uniform)
        return result
    return _numpy_lut(rgb, lut, uniform)


def _numpy_lut(rgb, lut, uniform):
    """Bound interpolation scratch space independently of export resolution."""
    shape = rgb.shape
    pixels = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    cube = np.asarray(lut, dtype=np.float32).reshape(-1, 3)
    result = np.empty_like(pixels)
    n = lut.shape[0]
    strides = np.array([n*n, n, 1], dtype=np.int32)
    diagonal = int(strides.sum())
    knots = 2 ** (np.linspace(0, 1, n)*np.log2(17))-1 if not uniform else None
    # Large whole-frame arrays, especially float64 fractions and per-pixel
    # argsort indices, overwhelm caches when several export workers overlap.
    for start in range(0, len(pixels), 65536):
        values = pixels[start:start+65536]
        coords = np.clip(values if uniform else np.log2(1+np.maximum(values, 0))/np.log2(17), 0, 1)*(n-1)
        indices = np.minimum(coords.astype(np.int32), n-2)
        if uniform:
            fractions = coords-indices.astype(np.float32)
        else:
            # Log shaper chooses cells; interpolate linearly within each cell,
            # using the same double-precision knots as the compiled kernel.
            fractions = np.clip((values-knots[indices])/(knots[indices+1]-knots[indices]), 0, 1).astype(np.float32)
        r, g, b = fractions.T
        # Only the largest/smallest axis is needed for four tetrahedron corners.
        # Equality keeps the original stable R,G,B order, including neutral ties.
        first = np.where(r >= g, np.where(r >= b, 0, 2), np.where(g >= b, 1, 2))
        last = np.where(b <= g, np.where(b <= r, 2, 0), np.where(g <= r, 1, 0))
        rows = np.arange(len(values))
        high = fractions[rows, first, None]
        middle = fractions[rows, 3-first-last, None]
        low = fractions[rows, last, None]
        base = indices @ strides
        block = cube[base]*(1-high)
        block += cube[base+strides[first]]*(high-middle)
        block += cube[base+diagonal-strides[last]]*(middle-low)
        block += cube[base+diagonal]*low
        result[start:start+len(values)] = block
    return result.reshape(shape)


def identity_lut(size=17):
    values = 2 ** (np.linspace(0, 1, size, dtype=np.float32)*np.log2(17))-1
    return np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1)


def gain_matrix(gain, exposure=0):
    # Analysis gains arrive from JSON as float64. Do not promote every pixel
    # in the full-resolution correction graph to double precision.
    gain = np.asarray(gain, dtype=np.float32)
    return np.asarray(FROM_AP1["rec709"].T @ np.diag(gain) @ TO_AP1["rec709"].T * 2**exposure,
                      dtype=np.float32)


def correct(encoded, grade, seconds=0, analysis=None, preview=False):
    p = grade.at(seconds)
    output = "rec709" if preview else grade.output_profile
    if analysis and p["strength"] and grade.input_profile == "rec709" and output == "rec709":
        from .enhance import cube_at
        cube = cube_at(analysis, seconds)
        if cube is not None:
            auto = apply_lut(encoded, cube, uniform=True)
            adjusted = np.clip(encoded+(auto-encoded)*np.float32(p["strength"]/100), 0, 1)
            neutral = all(p[k] == v for k, v in {
                "temperature": 0, "tint": 0, "exposure": 0, "contrast": 0,
                "highlights": 0, "shadows": 0, "black_point": 0, "saturation": 100,
                "hue": 0, "dehaze": 0, "sharpness": 0}.items())
            if neutral:
                return np.nan_to_num(adjusted, copy=False).astype(np.float32, copy=False)
            encoded = adjusted
            analysis = None  # Apply manual controls once, after the learned/automatic look.
    rgb = decode(encoded, grade.input_profile)
    if analysis and p["strength"]:
        from .analysis import transform_at
        gain, exposure, lut = transform_at(analysis, seconds)
        auto = apply_lut(rgb, lut) if lut is not None else rgb @ gain_matrix(gain, exposure)
        rgb = rgb+(auto-rgb)*p["strength"]/100
    rgb = manual(rgb, p)
    result = encode(rgb, output, tone=output == "rec709" and grade.input_profile != "rec709")
    if p["sharpness"]:
        detail = result-box_blur(result, 1)
        result += np.where(np.abs(detail) > .008, np.clip(detail, -.05, .05), 0)*p["sharpness"]/50
    return np.clip(np.nan_to_num(result), 0, 1).astype(np.float32)


def correct_rgb48(encoded, grade, seconds=0, analysis=None):
    """Export-only fused Rec.709 path, preserving the full graph as fallback."""
    if (_native is not None and hasattr(_native, "lut_rgb48") and analysis
            and grade.input_profile == "rec709" and grade.output_profile == "rec709"):
        p = grade.at(seconds)
        neutral = all(p[k] == v for k, v in {
            "temperature": 0, "tint": 0, "exposure": 0, "contrast": 0,
            "highlights": 0, "shadows": 0, "black_point": 0, "saturation": 100,
            "hue": 0, "dehaze": 0, "sharpness": 0}.items())
        # Strength zero follows the original color graph, not an identity
        # shortcut; active manual controls and log/HDR also keep their semantics.
        if p["strength"] and neutral:
            from .enhance import cube_at
            cube = cube_at(analysis, seconds)
            if cube is not None:
                frame = np.require(encoded, dtype=np.float32, requirements=["C", "A"])
                if frame.ndim < 1 or frame.shape[-1] != 3:
                    raise ValueError("Expected an RGB frame with three channels")
                cube = np.require(cube, dtype=np.float32, requirements=["C", "A"])
                output = np.empty(frame.shape, dtype=np.uint16)
                _native.lut_rgb48(frame, cube, output, cube.shape[0], p["strength"]/100)
                return output.astype("<u2", copy=False)
    result = correct(encoded, grade, seconds, analysis)
    scaled = result*65535
    np.rint(scaled, out=scaled)
    return scaled.astype("<u2")
