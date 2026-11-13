"""Nonblocking model discovery for the cloud advice dialog."""
import json
import os
from urllib.parse import urlencode

from .cloud import PROVIDERS
from .qt import QtCore as C, QtWidgets as W

try:
    from PySide6 import QtNetwork as N
except ImportError:
    from PyQt6 import QtNetwork as N


class ModelPicker(W.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider = self.key = ""
        self.reply = None
        self.ids = set()
        self.cursors = set()
        self.manager = N.QNetworkAccessManager(self)
        self.timer = C.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(700)
        self.timer.timeout.connect(self.refresh)
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = W.QComboBox()
        self.combo.setObjectName("cloud_model")
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(W.QComboBox.InsertPolicy.NoInsert)
        self.combo.lineEdit().setPlaceholderText("Select or enter a vision-capable model ID")
        layout.addWidget(self.combo)
        self.status = W.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.retry = W.QPushButton("Refresh models")
        self.retry.setAutoDefault(False)
        self.retry.clicked.connect(self.refresh)
        layout.addWidget(self.retry)

    def configure(self, provider, key):
        key = key.strip() or os.environ.get(PROVIDERS[provider][1], "")
        if (provider, key) == (self.provider, self.key):
            return
        self.timer.stop()
        self.abort()
        changed_provider = provider != self.provider
        self.provider, self.key = provider, key
        current = "" if changed_provider else self.combo.currentText()
        self.combo.clear()
        self.combo.setEditText(current)
        self.retry.setEnabled(bool(key))
        self.status.setText("Waiting to load models…" if key else "Enter an API key to load models, or type a model ID.")
        if key:
            self.timer.start()

    def abort(self):
        if self.reply is not None:
            reply, self.reply = self.reply, None
            reply.abort()
            reply.deleteLater()

    def stop(self):
        self.timer.stop()
        self.abort()

    def refresh(self):
        self.timer.stop()
        self.abort()
        if not self.key:
            return
        self.ids, self.cursors = set(), set()
        self.status.setText("Loading available models…")
        self.fetch()

    def fetch(self, cursor=None):
        base = ("https://api.openai.com/v1/models" if self.provider == "openai"
                else "https://api.anthropic.com/v1/models")
        if self.provider == "anthropic":
            base += "?" + urlencode({"limit": 1000} | ({"after_id": cursor} if cursor else {}))
        request = N.QNetworkRequest(C.QUrl(base))
        request.setTransferTimeout(10000)
        request.setAttribute(N.QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             N.QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        if self.provider == "openai":
            request.setRawHeader(b"Authorization", ("Bearer " + self.key).encode())
        else:
            request.setRawHeader(b"x-api-key", self.key.encode())
            request.setRawHeader(b"anthropic-version", b"2023-06-01")
        self.reply = self.manager.get(request)
        self.reply.finished.connect(self.received)

    def received(self):
        reply = self.sender()
        if reply is not self.reply:
            return
        self.reply = None
        try:
            status = reply.attribute(N.QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if reply.error() != N.QNetworkReply.NetworkError.NoError or status != 200:
                self.status.setText("Could not load models. Check the API key and connection, retry, or type a model ID.")
                return
            raw = bytes(reply.readAll())
            if len(raw) > 1024 * 1024:
                raise ValueError
            data = json.loads(raw)
            models = data["data"]
            if not isinstance(models, list) or any(
                    not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in models):
                raise ValueError
            self.ids.update(m["id"] for m in models if m["id"])
            if self.provider == "anthropic" and data.get("has_more"):
                cursor = data.get("last_id")
                if not isinstance(cursor, str) or not cursor or cursor in self.cursors or len(self.cursors) >= 20:
                    raise ValueError
                self.cursors.add(cursor)
                self.fetch(cursor)
                return
            current = self.combo.currentText()
            self.combo.clear()
            self.combo.addItems(sorted(self.ids))
            self.combo.setEditText(current)
            self.status.setText(
                f"{len(self.ids)} models available. Choose a model that supports images and structured grading output."
                if self.ids else "No models returned. You can still enter a model ID.")
        except (ValueError, KeyError, TypeError):
            self.status.setText("Invalid model list received. Retry or type a model ID.")
        finally:
            reply.deleteLater()
