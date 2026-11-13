import json
import time

import pytest


def header(request, name):
    try:
        return bytes(request.rawHeader(name))
    except TypeError:
        return bytes(request.rawHeader(name.encode()))


@pytest.fixture
def picker(monkeypatch):
    from uwcc import cloud_models as cm

    app = cm.W.QApplication.instance() or cm.W.QApplication([])
    widget = cm.ModelPicker()
    requests = []

    def wait(predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate()

    class Reply(cm.C.QObject):
        finished = cm.C.Signal() if hasattr(cm.C, "Signal") else cm.C.pyqtSignal()

        def __init__(self):
            super().__init__(widget)
            self.data = {"data": []}
            self.code = 200
            self.aborted = False

        def abort(self):
            self.aborted = True
            self.finished.emit()

        def error(self):
            return cm.N.QNetworkReply.NetworkError.NoError

        def attribute(self, _):
            return self.code

        def readAll(self):
            return json.dumps(self.data).encode()

    def get(request):
        reply = Reply()
        requests.append((request, reply))
        return reply

    monkeypatch.setattr(widget.manager, "get", get)
    yield widget, requests, wait
    widget.stop()
    widget.deleteLater()
    app.processEvents()


def test_discovery_debounces_key_and_preserves_manual_entry(picker):
    widget, requests, wait = picker
    widget.configure("openai", "first-secret")
    widget.configure("openai", "second-secret")
    assert not requests
    wait(lambda: len(requests) == 1)
    request, reply = requests[0]
    assert request.url().toString() == "https://api.openai.com/v1/models"
    assert header(request, "Authorization") == b"Bearer second-secret"
    widget.combo.setEditText("custom-vision")
    reply.data = {"data": [{"id": "vision-b"}, {"id": "vision-a"}, {"id": "vision-a"}]}
    reply.finished.emit()
    assert [widget.combo.itemText(i) for i in range(widget.combo.count())] == ["vision-a", "vision-b"]
    assert widget.combo.currentText() == "custom-vision"


def test_anthropic_pagination_and_provider_switch_cancels_old_reply(picker):
    widget, requests, _ = picker
    widget.configure("openai", "openai-secret")
    widget.refresh()
    old = requests[-1][1]
    widget.configure("anthropic", "claude-secret")
    assert old.aborted
    widget.refresh()
    request, reply = requests[-1]
    assert header(request, "x-api-key") == b"claude-secret"
    assert header(request, "anthropic-version") == b"2023-06-01"
    reply.data = {"data": [{"id": "claude-a"}], "has_more": True, "last_id": "claude-a"}
    reply.finished.emit()
    request, reply = requests[-1]
    assert "after_id=claude-a" in request.url().toString()
    reply.data = {"data": [{"id": "claude-b"}], "has_more": False}
    reply.finished.emit()
    assert widget.combo.count() == 2
    old.data = {"data": [{"id": "stale-openai"}]}
    old.finished.emit()
    assert widget.combo.findText("stale-openai") == -1


def test_discovery_environment_fallback_failure_and_stop(picker, monkeypatch):
    widget, requests, _ = picker
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    widget.configure("openai", "")
    widget.refresh()
    request, reply = requests[-1]
    assert header(request, "Authorization") == b"Bearer environment-secret"
    widget.combo.setEditText("manual-model")
    reply.code = 401
    reply.finished.emit()
    assert "Could not load" in widget.status.text()
    assert "environment-secret" not in widget.status.text()
    assert widget.combo.currentText() == "manual-model"
    widget.refresh()
    reply = requests[-1][1]
    widget.stop()
    assert reply.aborted
    assert not widget.timer.isActive()


def test_discovery_without_key_makes_no_request(picker, monkeypatch):
    widget, requests, _ = picker
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    widget.configure("openai", "")
    widget.refresh()
    assert not requests
    assert widget.combo.isEditable()

