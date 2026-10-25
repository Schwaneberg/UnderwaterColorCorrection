import hashlib

import numpy as np
import pytest

from uwcc.analysis import fit_reference
from uwcc.color import TO_AP1, srgb_encode
from uwcc.errors import UWCCError
from uwcc.ml import ReferenceModel
from uwcc.project import atomic_json


def test_reference_fit_cannot_invent_detail():
    source = np.random.default_rng(1).uniform(.02, .4, (48, 48, 3)).astype(np.float32)
    ap1 = source @ TO_AP1["rec709"].T
    reference = srgb_encode(source*[1.3, 1, .9])
    matrix, confidence = fit_reference(ap1, reference)
    expected = (source*[1.3, 1, .9]) @ TO_AP1["rec709"].T
    assert np.mean(np.abs(ap1 @ matrix-expected)) < .01
    assert confidence > .8


@pytest.mark.parametrize("field,value,match", [("weights_license", "unknown", "license"),
                                               ("onnx_sha256", "bad", "checksum")])
def test_model_rejects_unverified_artifacts(tmp_path, field, value, match):
    model = tmp_path/"model.onnx"
    model.write_bytes(b"not a model")
    manifest = {"schema_version": 1, "code_license": "MIT", "weights_license": "MIT",
                "file": "model.onnx", "onnx_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    manifest[field] = value
    path = tmp_path/"manifest.json"
    atomic_json(path, manifest)
    with pytest.raises(UWCCError, match=match):
        ReferenceModel(path)
