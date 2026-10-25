import numpy as np

from uwcc.analysis import analyze, estimate, stabilize, transform_at, validate_analysis
from uwcc.color import decode
from uwcc.media import probe
from uwcc.project import Grade


def test_red_signal_is_not_unbounded():
    rng = np.random.default_rng(1)
    frame = rng.uniform(.1, .8, (40, 40, 3)).astype(np.float32)
    frame[..., 0] *= .001
    gain, exposure, confidence, warnings = estimate(decode(frame, "rec709"))
    assert gain[0] <= 1.8
    assert confidence < .4
    assert any("Red channel" in message for message in warnings)


def test_smoothing_and_interpolation_never_cross_cuts():
    samples = [{"time": t, "scene": scene, "gain": [gain]*3, "exposure": 0}
               for t, scene, gain in [(0, 0, 1), (1, 0, 1), (1.1, 1, 3), (2, 1, 3)]]
    stabilize(samples)
    data = {"samples": samples}
    assert np.allclose(transform_at(data, 1.09)[0], 1)
    assert np.allclose(transform_at(data, 1.1)[0], 3)


def test_analysis_real_video(clip):
    result = analyze(probe(clip), Grade(input_profile="rec709"))
    validate_analysis(result)
    assert len(result["samples"]) >= 3
    assert result["engine"] == "Adaptive levels + gray-edge white balance"
    assert all("recipe" in sample for sample in result["samples"])
    assert result["warnings"]
