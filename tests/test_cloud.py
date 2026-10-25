from io import BytesIO
import json
from urllib.error import HTTPError

import numpy as np
import pytest

from uwcc import cloud
from uwcc.errors import UWCCError
from uwcc.project import Grade, Project, defaults


def proposal():
    return {"values": defaults() | {"contrast": 12., "saturation": 115.},
            "denoise": 3., "stabilize": False, "summary": "Increase contrast and color slightly."}


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_cloud_payload_is_opt_in_bounded_and_provider_specific(monkeypatch, provider):
    monkeypatch.setenv(cloud.PROVIDERS[provider][1], "test-secret")
    seen = []
    def fake(request, timeout):
        payload = json.loads(request.data)
        seen.append(payload)
        assert timeout == 30
        assert "test-secret" not in request.data.decode()
        if provider == "openai":
            assert payload["store"] is False
            assert payload["text"]["format"]["strict"]
            images = [c for c in payload["input"][0]["content"] if c["type"] == "input_image"]
            assert len(images) == 3
            response = {"output": [{"content": [{"type": "output_text", "text": json.dumps(proposal())}]}]}
        else:
            assert payload["tool_choice"]["name"] == "propose_grade"
            assert len([c for c in payload["messages"][0]["content"] if c["type"] == "image"]) == 3
            response = {"content": [{"type": "tool_use", "name": "propose_grade", "input": proposal()}]}
        return BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(cloud, "urlopen", fake)
    actual = cloud.request_advice(provider, "test-vision", [np.ones((20, 30, 3), np.float32)*.5]*3)
    assert actual["values"]["contrast"] == 12
    assert len(seen) == 1
    p = Project("source.mp4", Grade(input_profile="rec709"))
    p.grade.set_key("exposure", 0, .1)
    updated = cloud.apply_advice(p, actual)
    assert updated.grade.values["saturation"] == 115
    assert not updated.grade.keyframes
    assert p.grade.keyframes


def test_no_credentials_no_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cloud, "urlopen", lambda *_: pytest.fail("Unexpected network call"))
    with pytest.raises(UWCCError, match="OPENAI_API_KEY"):
        cloud.request_advice("openai", "test", [np.zeros((2, 2, 3))])


@pytest.mark.parametrize("field,value", [("denoise", float("nan")), ("denoise", True), ("stabilize", "yes")])
def test_bad_cloud_proposals_rejected(field, value):
    data = proposal()
    data[field] = value
    with pytest.raises(UWCCError):
        cloud.validate_advice(data)


def test_provider_errors_do_not_expose_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    def fail(*_, **__):
        raise HTTPError("url", 401, "private-key", {}, None)
    monkeypatch.setattr(cloud, "urlopen", fail)
    with pytest.raises(UWCCError) as error:
        cloud.request_advice("openai", "test", [np.zeros((2, 2, 3))])
    assert "401" in str(error.value) and "private-key" not in str(error.value)


def test_cli_requires_explicit_upload_opt_in(monkeypatch, capsys):
    from uwcc.__main__ import main
    monkeypatch.setattr(cloud, "advise_project", lambda *_: pytest.fail("Unexpected upload"))
    result = main(["advise", "missing-project.json", "--provider", "openai", "--model", "vision", "--output", "new.json"])
    assert result == 1
    assert "--allow-upload" in capsys.readouterr().err


def test_cloud_cannot_add_unknown_controls():
    data = proposal()
    data["values"]["source"] = "/tmp/anything"
    with pytest.raises(UWCCError):
        cloud.validate_advice(data)
