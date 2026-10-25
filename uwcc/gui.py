"""Desktop workspace. All native decode, analysis and render work runs off-thread."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import analyze
from .color import correct, decode, encode
from .errors import Cancelled
from .jobs import BatchQueue
from .media import FrameReader, frame_timestamps, prepare_frame, probe, read_frame
from .project import (BUILTINS, CONTROLS, FILTERS, OUTPUTS, PROFILES, Grade, Project,
                      clone, fingerprint, load_preset, save_preset, user_directory)
from .qt import QtCore as C, QtGui as G, QtWidgets as W, Signal, multimedia
from .render import RENDER_PROFILES, render


class Worker(C.QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(float, str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.cancel = threading.Event()

    def run(self):
        try:
            result = self.operation(self.cancel, self.progress.emit)
            self.completed.emit(result)
        except Cancelled:
            self.failed.emit("Operation canceled.")
        except Exception as exc:
            self.failed.emit(str(exc))


def qimage(array):
    raw = np.ascontiguousarray(np.clip(array*255, 0, 255), dtype=np.uint8)
    return G.QImage(raw.data, raw.shape[1], raw.shape[0], raw.strides[0], G.QImage.Format.Format_RGB888).copy()


class Viewer(W.QWidget):
    def __init__(self):
        super().__init__()
        self.original = self.corrected = None
        self.mode = "Split"
        self.split = .5
        self.zoom = 0
        self.pan = C.QPointF(0, 0)
        self.drag_position = None
        self.sar = 1.
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setAccessibleName("Video comparison viewer")

    def paintEvent(self, event):
        painter = G.QPainter(self)
        painter.fillRect(self.rect(), G.QColor("#0b1118"))
        if self.original is None:
            painter.setPen(G.QColor("#91a5b6"))
            painter.drawText(self.rect(), C.Qt.AlignmentFlag.AlignCenter,
                             "UNDERWATER COLOR CORRECTION\n\nOpen a video to start\n\nLocal processing · No cloud · No account")
            return
        width, height = self.original.width()*self.sar, self.original.height()
        scale = min(self.width()/width, self.height()/height) if self.zoom == 0 else self.zoom
        target = C.QRectF((self.width()-width*scale)/2+self.pan.x(), (self.height()-height*scale)/2+self.pan.y(), width*scale, height*scale)
        painter.setRenderHint(G.QPainter.RenderHint.SmoothPixmapTransform)
        image = self.original if self.mode == "Original" else self.corrected
        painter.drawImage(target, image)
        if self.mode == "Split":
            cut = int(self.width()*self.split)
            painter.save()
            painter.setClipRect(0, 0, cut, self.height())
            painter.drawImage(target, self.original)
            painter.restore()
            painter.setPen(G.QPen(G.QColor("#63ddcf"), 2))
            painter.drawLine(cut, 0, cut, self.height())
        painter.setPen(G.QColor("#e0eded"))
        painter.drawText(12, 24, "ORIGINAL" if self.mode == "Original" else "ORIGINAL  |  CORRECTED" if self.mode == "Split" else "CORRECTED")

    def mouseMoveEvent(self, event):
        if event.buttons() & C.Qt.MouseButton.LeftButton and self.mode == "Split":
            self.split = max(0, min(1, event.position().x()/self.width()))
            self.update()
        elif event.buttons() & (C.Qt.MouseButton.MiddleButton | C.Qt.MouseButton.RightButton) and self.drag_position is not None:
            self.pan += event.position()-self.drag_position
            self.drag_position = event.position()
            self.update()

    def mousePressEvent(self, event):
        self.drag_position = event.position()

    def mouseReleaseEvent(self, event):
        self.drag_position = None


class Scopes(W.QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.mode = "Histogram"
        self.setMinimumHeight(140)
        self.setMaximumHeight(220)
        self.setAccessibleName("Color scopes at preview resolution")

    def paintEvent(self, event):
        painter = G.QPainter(self)
        painter.fillRect(self.rect(), G.QColor("#0d141e"))
        if self.data is None:
            return
        frame = self.data[::2, ::2]
        colors = [G.QColor("#f27383"), G.QColor("#65d99c"), G.QColor("#6f9fff")]
        if self.mode == "Histogram":
            for channel, color in enumerate(colors):
                hist, _ = np.histogram(frame[..., channel], bins=128, range=(0, 1))
                values = np.log1p(hist)
                values /= max(float(values.max()), 1)
                path = G.QPainterPath()
                for i, y in enumerate(values):
                    point = C.QPointF(i/127*self.width(), self.height()-y*(self.height()-20))
                    path.moveTo(point) if i == 0 else path.lineTo(point)
                painter.setPen(G.QPen(color, 1))
                painter.drawPath(path)
        else:
            image = np.zeros((128, 256, 3), dtype=np.uint8)
            rgb = np.clip(frame, 0, 1)
            if self.mode == "Vectorscope":
                y = rgb @ np.array([.2126, .7152, .0722])
                u, v = (rgb[..., 2]-y)*.5+.5, (rgb[..., 0]-y)*.5+.5
                xx, yy = np.clip(u*255, 0, 255).astype(int), np.clip((1-v)*127, 0, 127).astype(int)
                image[yy.ravel(), xx.ravel()] = (100, 225, 195)
            else:
                for c in range(3):
                    xs = np.linspace(0, 255 if self.mode == "Waveform" else 84, rgb.shape[1]).astype(int)
                    xs = xs + (c*85 if self.mode == "RGB parade" else 0)
                    ys = np.clip((1-rgb[..., c])*127, 0, 127).astype(int)
                    image[ys.ravel(), np.broadcast_to(xs, ys.shape).ravel(), c] = 220
            painter.drawImage(self.rect(), qimage(image.astype(np.float32)/255))
        painter.setPen(G.QColor("#8397a6"))
        painter.drawText(8, 16, self.mode+" · Preview resolution")


class Playback(C.QThread):
    frame = Signal(object, object, float)
    failed = Signal(str)
    audio_start = Signal(float)

    def __init__(self, info, project, start, parent=None):
        super().__init__(parent)
        self.info, self.project, self.start_time = info, clone(project), start
        self.cancel = threading.Event()

    def run(self):
        try:
            beginning = None
            with closing(frame_timestamps(self.info, self.cancel, start=self.start_time)) as times, FrameReader(
                self.info, max_width=640, start=self.start_time, cancel=self.cancel
            ) as reader:
                for raw in reader:
                    if self.cancel.is_set():
                        return
                    timestamp = next(times, None)
                    if timestamp is None:
                        return
                    raw = prepare_frame(raw, self.info, self.project.grade.input_profile)
                    rgb = correct(raw, self.project.grade, timestamp, self.project.analysis, preview=True)
                    original = encode(decode(raw, self.project.grade.input_profile), "rec709", self.project.grade.input_profile != "rec709")
                    if beginning is None:
                        beginning, start = time.monotonic(), timestamp
                        self.audio_start.emit(timestamp)
                    delay = timestamp-start-(time.monotonic()-beginning)
                    if delay > 0 and self.cancel.wait(delay):
                        return
                    self.frame.emit(original, rgb, timestamp)
        except Cancelled:
            return
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchDialog(W.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle("Batch correction")
        self.resize(900, 420)
        self.window, self.worker = window, None
        self.queue = BatchQueue(user_directory()/"queue.json")
        layout = W.QVBoxLayout(self)
        layout.addWidget(W.QLabel("Each clip is analyzed independently using the current settings. Failed items can be retried."))
        self.table = W.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Source", "Output", "State", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        buttons = W.QHBoxLayout()
        self.add_button = W.QPushButton("Add videos…")
        self.add_button.clicked.connect(self.add)
        self.start_button = W.QPushButton("Start / resume / retry")
        self.start_button.clicked.connect(self.start)
        self.pause_button = W.QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        self.clear_button = W.QPushButton("Clear completed")
        self.clear_button.clicked.connect(self.clear)
        for button in (self.add_button, self.start_button, self.pause_button, self.clear_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.timer = C.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(300)
        self.refresh()

    def refresh(self):
        self.table.setRowCount(len(self.queue.jobs))
        for row, job in enumerate(self.queue.jobs):
            for col, text in enumerate((Path(job["source"]).name, job["destination"], job["state"], job["message"])):
                self.table.setItem(row, col, W.QTableWidgetItem(str(text)))
        active = self.worker is not None and self.worker.isRunning()
        self.add_button.setEnabled(not active)
        self.start_button.setEnabled(not active)
        self.clear_button.setEnabled(not active)
        self.pause_button.setEnabled(active)

    def add(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, "Choose videos", "", "Video (*.mp4 *.mov *.mkv *.mts *.m2ts *.avi);;All files (*)")
        if not paths:
            return
        folder = W.QFileDialog.getExistingDirectory(self, "Output directory")
        if not folder:
            return
        grade = clone(self.window.project.grade) if self.window.project else Grade()
        grade.keyframes.clear()
        for path in paths:
            self.queue.add(path, folder, grade, self.window.export_profile.currentData())
        self.refresh()

    def start(self):
        self.worker = Worker(lambda cancel, progress: self.queue.process(cancel, progress), self)
        self.worker.finished.connect(self.refresh)
        self.worker.failed.connect(lambda message: W.QMessageBox.warning(self, "Batch error", message))
        self.worker.start()

    def pause(self):
        if self.worker:
            self.worker.cancel.set()

    def clear(self):
        self.queue.jobs = [job for job in self.queue.jobs if job["state"] != "Complete"]
        self.queue.save()
        self.refresh()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.pause()
            event.ignore()
            W.QMessageBox.information(self, "Pausing batch", "The current operation is stopping. Close this window once it is paused.")
        else:
            event.accept()


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"UWCC {__version__} · Underwater Color Correction")
        self.resize(1380, 900)
        self.project = self.info = None
        self.pending_preset = None
        self.dirty = False
        self.history, self.future, self.workers = [], [], []
        self.position = 0.
        self.playback = self.operation = self.preview_worker = None
        self.preview_pending = False
        self.generation = 0
        self.batch_dialog = None
        self.recovery = user_directory()/"recovery.uwcc-project.json"
        self._build()
        self._audio()
        self.preview_timer = C.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self.request_preview)
        self.autosave = C.QTimer(self)
        self.autosave.setSingleShot(True)
        self.autosave.setInterval(2000)
        self.autosave.timeout.connect(self.save_recovery)
        self.setAcceptDrops(True)
        self.update_enabled()

    def _build(self):
        toolbar = self.addToolBar("Workspace")
        toolbar.setMovable(False)
        def action(text, callback, shortcut=None):
            item = G.QAction(text, self)
            item.triggered.connect(callback)
            if shortcut:
                item.setShortcut(shortcut)
            toolbar.addAction(item)
            return item
        action("Open video…", self.open_video, "Ctrl+O")
        action("Open project…", self.open_project, "Ctrl+Shift+O")
        self.save_action = action("Save project", self.save_project, "Ctrl+S")
        self.undo_action = action("Undo", self.undo, "Ctrl+Z")
        self.redo_action = action("Redo", self.redo, "Ctrl+Shift+Z")
        toolbar.addSeparator()
        self.analyze_action = action("Auto correct", self.auto_correct, "Ctrl+Return")
        action("Batch…", self.batch)
        self.export_action = action("Export…", self.export_video, "Ctrl+E")
        toolbar.addSeparator()
        action("About / capabilities", self.about)
        central = W.QWidget()
        layout = W.QHBoxLayout(central)
        self.setCentralWidget(central)
        left = W.QVBoxLayout()
        self.viewer = Viewer()
        left.addWidget(self.viewer, 1)
        view_controls = W.QHBoxLayout()
        self.view_mode = W.QComboBox()
        self.view_mode.addItems(["Split", "Corrected", "Original"])
        self.view_mode.currentTextChanged.connect(self.change_view)
        view_controls.addWidget(self.view_mode)
        self.zoom = W.QComboBox()
        self.zoom.addItems(["Fit", "50%", "100%", "200%"])
        self.zoom.currentIndexChanged.connect(self.change_zoom)
        view_controls.addWidget(self.zoom)
        self.scope_mode = W.QComboBox()
        self.scope_mode.addItems(["Scopes hidden", "Histogram", "Waveform", "RGB parade", "Vectorscope"])
        self.scope_mode.currentTextChanged.connect(self.change_scope)
        view_controls.addWidget(self.scope_mode)
        view_controls.addStretch()
        left.addLayout(view_controls)
        self.scopes = Scopes()
        self.scopes.hide()
        left.addWidget(self.scopes)
        transport = W.QHBoxLayout()
        self.play_button = W.QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_playback)
        transport.addWidget(self.play_button)
        self.time_label = W.QLabel("00:00:00.000 / 00:00:00.000")
        transport.addWidget(self.time_label)
        transport.addStretch()
        self.audio_label = W.QLabel("Proxy preview · 640 px playback")
        transport.addWidget(self.audio_label)
        left.addLayout(transport)
        self.timeline = W.QSlider(C.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 10000)
        self.timeline.setAccessibleName("Clip playhead")
        self.timeline.sliderPressed.connect(self.stop_playback)
        self.timeline.valueChanged.connect(self.seek)
        left.addWidget(self.timeline)
        self.key_list = W.QComboBox()
        self.key_list.setAccessibleName("Manual keyframes")
        self.key_list.activated.connect(self.jump_key)
        left.addWidget(self.key_list)
        layout.addLayout(left, 1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(440)
        inspector = W.QWidget()
        form = W.QVBoxLayout(inspector)
        title = W.QLabel("CORRECTION")
        title.setStyleSheet("font-size: 19px; font-weight: bold; color: #72e1d3; padding: 8px 0")
        form.addWidget(title)
        self.source_label = W.QLabel("No video open")
        self.source_label.setWordWrap(True)
        form.addWidget(self.source_label)
        self.input_profile = self.combo(form, "Input color profile", PROFILES)
        self.input_profile.currentIndexChanged.connect(self.profile_changed)
        self.filter_hint = self.combo(form, "Capture filter", FILTERS)
        self.filter_hint.currentIndexChanged.connect(self.profile_changed)
        self.output_profile = self.combo(form, "Output color intent", OUTPUTS)
        self.output_profile.currentIndexChanged.connect(self.output_changed)
        self.auto_method = self.combo(form, "Automatic correction", {
            "ai": "Local AI enhancement + white balance", "adaptive": "Automatic levels / white balance",
            "legacy": "Legacy balance"})
        self.auto_method.currentIndexChanged.connect(self.method_changed)
        self.stabilize = W.QCheckBox("Stabilize video on export")
        self.stabilize.setToolTip("Track background features and apply one fixed crop/zoom so no exposed or mirrored edges appear. Export only; float RGB precision.")
        self.stabilize.toggled.connect(self.finishing_changed)
        form.addWidget(self.stabilize)
        self.stabilization_region = self.combo(form, "Stabilization reference", {"seabed": "Seabed / lower half", "whole": "Whole image"})
        self.stabilization_region.currentIndexChanged.connect(self.finishing_changed)
        form.addWidget(W.QLabel("Denoise on export · 0 = off"))
        self.denoise = W.QDoubleSpinBox()
        self.denoise.setRange(0, 10)
        self.denoise.setSingleStep(.5)
        self.denoise.setAccessibleName("Export denoise strength")
        self.denoise.setToolTip("Spatial and temporal noise reduction. Start at 2–4; high strengths may soften detail. Export only.")
        self.denoise.valueChanged.connect(self.finishing_changed)
        form.addWidget(self.denoise)
        self.ai_denoise = W.QCheckBox("Use FastDVDnet AI denoise (slow on CPU)")
        self.ai_denoise.setToolTip("Five-frame, full-resolution neural denoise. Much slower on CPU; uses the installed local model.")
        self.ai_denoise.toggled.connect(self.finishing_changed)
        form.addWidget(self.ai_denoise)
        for label, callback in (("Learn from corrected reference…", self.learn_reference),
                                ("Cloud grading advice…", self.cloud_advice)):
            button = W.QPushButton(label)
            button.clicked.connect(callback)
            form.addWidget(button)
        preset_row = W.QHBoxLayout()
        self.preset = W.QComboBox()
        self.preset.addItems(list(BUILTINS))
        preset_row.addWidget(self.preset)
        apply = W.QPushButton("Apply")
        apply.clicked.connect(self.builtin_preset)
        preset_row.addWidget(apply)
        form.addLayout(preset_row)
        preset_buttons = W.QHBoxLayout()
        self.load_preset_button = W.QPushButton("Load preset…")
        self.load_preset_button.clicked.connect(self.load_preset)
        self.save_preset_button = W.QPushButton("Save preset…")
        self.save_preset_button.clicked.connect(self.save_preset)
        for button in (self.load_preset_button, self.save_preset_button):
            preset_buttons.addWidget(button)
        form.addLayout(preset_buttons)
        self.engine_label = W.QLabel("Engine: classical adaptive balance\nOptional ONNX model can be selected below.")
        self.engine_label.setWordWrap(True)
        form.addWidget(self.engine_label)
        model_row = W.QHBoxLayout()
        for text, callback in (("Select ONNX manifest…", self.select_model), ("Use automatic levels", self.clear_model)):
            button = W.QPushButton(text)
            button.clicked.connect(callback)
            model_row.addWidget(button)
        form.addLayout(model_row)
        self.controls = {}
        self.control_area = W.QWidget()
        control_layout = W.QVBoxLayout(self.control_area)
        control_layout.setContentsMargins(0, 0, 0, 0)
        for name, (label, low, high, default, step) in CONTROLS.items():
            group = W.QVBoxLayout()
            row = W.QHBoxLayout()
            row.addWidget(W.QLabel(label), 1)
            spin = W.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2 if step < 1 else 0)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setAccessibleName(label)
            row.addWidget(spin)
            key = W.QToolButton()
            key.setText("◇")
            key.setToolTip("Add or replace a keyframe at the playhead")
            key.setAccessibleName("Keyframe "+label)
            key.clicked.connect(lambda checked=False, n=name: self.add_key(n))
            row.addWidget(key)
            reset = W.QToolButton()
            reset.setText("↺")
            reset.setToolTip("Reset control and remove its animation")
            reset.clicked.connect(lambda checked=False, n=name: self.reset_control(n))
            row.addWidget(reset)
            slider = W.QSlider(C.Qt.Orientation.Horizontal)
            slider.setRange(round(low/step), round(high/step))
            slider.setValue(round(default/step))
            slider.setAccessibleName(label+" slider")
            slider.valueChanged.connect(lambda value, s=spin, st=step: s.setValue(value*st))
            spin.valueChanged.connect(lambda value, n=name: self.control_changed(n, value))
            group.addLayout(row)
            group.addWidget(slider)
            control_layout.addLayout(group)
            self.controls[name] = (spin, slider, key)
        form.addWidget(self.control_area)
        delete_key = W.QPushButton("Delete selected keyframe")
        delete_key.clicked.connect(self.delete_key)
        form.addWidget(delete_key)
        reset_all = W.QPushButton("Reset all manual controls")
        reset_all.clicked.connect(self.reset_all)
        form.addWidget(reset_all)
        self.export_profile = self.combo(form, "Export format", {k: v[0] for k, v in RENDER_PROFILES.items()})
        form.addStretch()
        scroll.setWidget(inspector)
        layout.addWidget(scroll)
        self.progress = W.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 1000)
        self.statusBar().addPermanentWidget(self.progress)
        self.cancel_button = W.QPushButton("Cancel task")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage("Open a clip to begin. Processing stays on your computer.")
        self.shortcut = G.QShortcut(G.QKeySequence("Space"), self)
        self.shortcut.activated.connect(self.toggle_playback)

    def combo(self, layout, label, items):
        layout.addWidget(W.QLabel(label))
        combo = W.QComboBox()
        combo.setAccessibleName(label)
        for key, value in items.items():
            combo.addItem(value, key)
        layout.addWidget(combo)
        return combo

    def _audio(self):
        module = multimedia()
        self.audio_player = None
        if module:
            self.audio_player = module.QMediaPlayer(self)
            self.audio_output = module.QAudioOutput(self)
            self.audio_player.setAudioOutput(self.audio_output)
        else:
            self.audio_label.setText("Silent proxy preview · audio preserved on export")

    def error(self, message):
        if "canceled" in message.lower():
            self.statusBar().showMessage(message)
        else:
            W.QMessageBox.warning(self, "UWCC", message)

    def task(self, operation, success):
        if self.operation and self.operation.isRunning():
            self.error("Wait for the current task or cancel it first.")
            return
        worker = Worker(operation, self)
        self.operation = worker
        worker.progress.connect(self.task_progress)
        worker.completed.connect(success)
        worker.failed.connect(self.error)
        worker.finished.connect(self.task_finished)
        self.workers.append(worker)
        worker.start()
        self.update_enabled()

    def task_progress(self, fraction, message):
        self.progress.setValue(round(fraction*1000))
        self.statusBar().showMessage(message)

    def task_finished(self):
        self.operation = None
        worker = self.sender()
        self.workers = [item for item in self.workers if item is not worker]
        worker.deleteLater()
        self.update_enabled()

    def cancel_operation(self):
        if self.operation:
            self.operation.cancel.set()

    def update_enabled(self):
        ready = self.project is not None
        busy = self.operation is not None and self.operation.isRunning()
        for action in (self.save_action, self.analyze_action, self.export_action):
            action.setEnabled(ready and not busy)
        self.control_area.setEnabled(ready)
        self.stabilize.setEnabled(ready)
        self.denoise.setEnabled(ready)
        self.ai_denoise.setEnabled(ready)
        self.auto_method.setEnabled(ready)
        self.stabilization_region.setEnabled(ready)
        self.load_preset_button.setEnabled(not busy)
        self.save_preset_button.setEnabled(ready and not busy)
        self.play_button.setEnabled(ready)
        self.timeline.setEnabled(ready)
        self.undo_action.setEnabled(bool(self.history) and not busy)
        self.redo_action.setEnabled(bool(self.future) and not busy)
        self.cancel_button.setEnabled(busy)

    def confirm_discard(self):
        if self.operation and self.operation.isRunning():
            self.error("Cancel or finish the current operation before switching videos.")
            return False
        if not self.dirty:
            return True
        answer = W.QMessageBox.question(self, "Unsaved changes", "Save changes before continuing?",
                                       W.QMessageBox.StandardButton.Save | W.QMessageBox.StandardButton.Discard | W.QMessageBox.StandardButton.Cancel)
        if answer == W.QMessageBox.StandardButton.Cancel:
            return False
        if answer == W.QMessageBox.StandardButton.Save:
            return self.save_project()
        self.recovery.unlink(missing_ok=True)
        self.autosave.stop()
        return True

    def open_video(self, checked=False, path=None):
        if not self.confirm_discard():
            return
        if not path:
            path, _ = W.QFileDialog.getOpenFileName(self, "Open underwater video", "", "Video (*.mp4 *.mov *.mkv *.mts *.m2ts *.avi);;All files (*)")
        if not path:
            return
        self.stop_playback()
        pending = clone(self.pending_preset)
        def operation(cancel, progress):
            info = probe(path, cancel)
            grade = pending if pending is not None else Grade(
                input_profile=info.profile, auto_method="ai",
                output_profile=info.profile if info.profile in ("hlg", "pq") else "rec709")
            if grade.input_profile == "auto":
                grade.input_profile = info.profile
            project = Project(info.path, grade, fingerprint(Path(info.path)))
            return info, project
        self.task(operation, self.opened)

    def opened(self, pair):
        self.info, self.project = pair
        self.pending_preset = None
        self.generation += 1
        self.position = 0
        self.history, self.future = [], []
        self.dirty = self.project.path == self.recovery
        self.source_label.setText(f"{Path(self.info.path).name}\n{self.info.width} × {self.info.height} · {float(self.info.fps):.3f} fps · {self.info.duration:.1f}s")
        try:
            self.viewer.sar = float(Fraction(self.info.sar.replace(":", "/"))) if self.info.sar not in ("N/A", "0:1") else 1
        except (ValueError, ZeroDivisionError):
            self.viewer.sar = 1
        if self.audio_player:
            self.audio_player.setSource(C.QUrl.fromLocalFile(self.info.path))
        self.sync_controls()
        self.request_preview()
        self.update_enabled()
        self.statusBar().showMessage("Confirm the input color profile." if self.project.grade.input_profile == "auto" else "Ready. Use Auto correct to analyze the clip.")

    def open_project(self):
        if not self.confirm_discard():
            return
        path, _ = W.QFileDialog.getOpenFileName(self, "Open UWCC project", "", "UWCC project (*.json)")
        if path:
            self.load_project_path(path)

    def load_project_path(self, path):
        self.stop_playback()
        def operation(cancel, progress):
            project = Project.load(path)
            return probe(project.source, cancel), project
        self.task(operation, self.opened)

    def save_project(self):
        if not self.project:
            return False
        path = self.project.path
        if path is None or path == self.recovery:
            proposed = str(Path(self.project.source).with_suffix(".uwcc-project.json"))
            path, _ = W.QFileDialog.getSaveFileName(self, "Save UWCC project", proposed, "UWCC project (*.json)")
        if not path:
            return False
        try:
            self.project.save(path)
            self.dirty = False
            self.autosave.stop()
            self.recovery.unlink(missing_ok=True)
            self.statusBar().showMessage(f"Project saved: {Path(path).name}")
            return True
        except Exception as exc:
            self.error(str(exc))
            return False

    def save_recovery(self):
        if self.project and self.dirty:
            try:
                from .project import atomic_json
                atomic_json(self.recovery, self.project.to_dict())
            except OSError as exc:
                self.statusBar().showMessage(f"Recovery save failed: {exc}")

    def recover(self):
        if self.recovery.exists():
            if W.QMessageBox.question(self, "Recover edits", "Recover the previous unsaved editing session?") == W.QMessageBox.StandardButton.Yes:
                self.load_project_path(self.recovery)

    def remember(self):
        if self.project:
            self.history.append((clone(self.project.grade), self.project.analysis))
            self.history = self.history[-100:]
            self.future.clear()

    def changed(self):
        self.dirty = True
        self.generation += 1
        self.autosave.start()
        self.preview_timer.start()
        self.update_enabled()

    def undo(self):
        if self.history:
            self.stop_playback()
            self.future.append((clone(self.project.grade), self.project.analysis))
            self.project.grade, self.project.analysis = self.history.pop()
            self.sync_controls()
            self.changed()

    def redo(self):
        if self.future:
            self.stop_playback()
            self.history.append((clone(self.project.grade), self.project.analysis))
            self.project.grade, self.project.analysis = self.future.pop()
            self.sync_controls()
            self.changed()

    def sync_controls(self):
        if not self.project:
            return
        grade = self.project.grade
        for combo, value in ((self.auto_method, grade.auto_method), (self.stabilization_region, grade.stabilization_region)):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        self.ai_denoise.blockSignals(True)
        self.ai_denoise.setChecked(grade.denoise_engine == "ai")
        self.ai_denoise.blockSignals(False)
        for widget, value in ((self.stabilize, grade.stabilize), (self.denoise, grade.denoise)):
            widget.blockSignals(True)
            if widget is self.stabilize:
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        for combo, value in ((self.input_profile, grade.input_profile), (self.filter_hint, grade.filter_hint), (self.output_profile, grade.output_profile)):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        values = grade.at(self.position)
        for name, (spin, slider, key) in self.controls.items():
            spin.blockSignals(True)
            slider.blockSignals(True)
            spin.setValue(values[name])
            slider.setValue(round(values[name]/CONTROLS[name][4]))
            spin.blockSignals(False)
            slider.blockSignals(False)
            key.setText("◆" if grade.keyframes.get(name) else "◇")
        self.key_list.clear()
        self.key_list.addItem("Manual keyframes · select to jump", None)
        for name, keys in grade.keyframes.items():
            for item in keys:
                seconds = float(Fraction(item["time"]))
                self.key_list.addItem(f"{self.timestamp(seconds)}  {CONTROLS[name][0]} = {item['value']:g}", (name, item["time"]))
        self.update_engine()

    def update_engine(self):
        if self.project and self.project.analysis:
            a = self.project.analysis
            self.engine_label.setText(f"{a['engine']}\n{a['shots']} shots · confidence {a['confidence']:.0%}\n"+"\n".join(a["warnings"]))
        elif self.project and self.project.grade.model_manifest:
            self.engine_label.setText("ONNX model selected · run Auto correct to analyze.")
        else:
            method = self.project.grade.auto_method if self.project else "ai"
            self.engine_label.setText("Local AI requested; adaptive levels/white balance if unavailable.\nRun Auto correct to analyze."
                                     if method == "ai" else "Automatic levels and gray-edge white balance.\nRun Auto correct to analyze.")

    def control_changed(self, name, value):
        if not self.project:
            return
        self.stop_playback()
        self.remember()
        grade = self.project.grade
        if grade.keyframes.get(name):
            grade.set_key(name, self.position, value)
        else:
            grade.values[name] = value
        _, slider, _ = self.controls[name]
        slider.blockSignals(True)
        slider.setValue(round(value/CONTROLS[name][4]))
        slider.blockSignals(False)
        self.changed()

    def profile_changed(self):
        if not self.project:
            return
        self.stop_playback()
        self.remember()
        self.project.grade.input_profile = self.input_profile.currentData()
        self.project.grade.filter_hint = self.filter_hint.currentData()
        self.project.analysis = None
        self.update_engine()
        self.changed()

    def output_changed(self):
        if self.project:
            self.remember()
            self.project.grade.output_profile = self.output_profile.currentData()
            self.changed()

    def finishing_changed(self):
        if self.project:
            self.remember()
            self.project.grade.stabilize = self.stabilize.isChecked()
            self.project.grade.denoise = self.denoise.value()
            self.project.grade.stabilization_region = self.stabilization_region.currentData()
            self.project.grade.denoise_engine = "ai" if self.ai_denoise.isChecked() else "classical"
            from .neural import builtin_model
            self.project.grade.denoise_manifest = builtin_model("denoise") if self.ai_denoise.isChecked() else None
            self.changed()

    def method_changed(self):
        if self.project:
            self.stop_playback()
            self.remember()
            self.project.grade.auto_method = self.auto_method.currentData()
            if self.project.grade.auto_method != "ai":
                self.project.grade.model_manifest = None
            self.project.analysis = None
            self.update_engine()
            self.changed()

    def add_key(self, name):
        if not self.project:
            return
        self.remember()
        self.project.grade.set_key(name, self.position, self.controls[name][0].value())
        self.sync_controls()
        self.changed()

    def delete_key(self):
        data = self.key_list.currentData()
        if self.project and data:
            self.remember()
            name, timestamp = data
            self.project.grade.keyframes[name] = [k for k in self.project.grade.keyframes[name] if k["time"] != timestamp]
            self.sync_controls()
            self.changed()

    def jump_key(self):
        data = self.key_list.currentData()
        if data and self.info:
            self.timeline.setValue(round(float(Fraction(data[1]))/self.info.duration*10000))

    def reset_control(self, name):
        if self.project:
            self.remember()
            self.project.grade.values[name] = CONTROLS[name][3]
            self.project.grade.keyframes.pop(name, None)
            self.sync_controls()
            self.changed()

    def reset_all(self):
        if self.project:
            self.remember()
            self.project.grade.values = Grade().values
            self.project.grade.keyframes.clear()
            self.sync_controls()
            self.changed()

    def builtin_preset(self):
        if self.project:
            self.remember()
            self.project.grade.values = Grade().values | BUILTINS[self.preset.currentText()]
            self.project.grade.keyframes.clear()
            if self.preset.currentText() == "Red-filter natural":
                self.project.grade.filter_hint = "red"
                self.project.analysis = None
            self.sync_controls()
            self.changed()

    def save_preset(self):
        if self.project:
            path, _ = W.QFileDialog.getSaveFileName(self, "Save preset", "custom.uwcc-preset.json", "UWCC preset (*.json)")
            if path:
                try:
                    save_preset(path, self.project.grade, Path(path).stem)
                except Exception as exc:
                    self.error(str(exc))

    def load_preset(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Load preset", "", "UWCC preset (*.json)")
        if path:
            try:
                grade = load_preset(path)
                if self.project:
                    self.stop_playback()
                    self.remember()
                    self.project.grade, self.project.analysis = grade, None
                    self.sync_controls()
                    self.changed()
                else:
                    self.pending_preset = grade
                    self.statusBar().showMessage(f"Preset loaded: {Path(path).name}. Open a video to apply it.")
            except Exception as exc:
                self.error(str(exc))

    def select_model(self):
        if not self.project:
            return
        path, _ = W.QFileDialog.getOpenFileName(self, "Select local ONNX manifest", "", "Model manifest (*.json)")
        if path:
            def select(cancel, progress):
                from .ml import ReferenceModel
                model = ReferenceModel(path)
                return path, model.name
            def selected(result):
                self.remember()
                self.project.grade.model_manifest = result[0]
                self.project.grade.auto_method = "ai"
                self.project.analysis = None
                self.sync_controls()
                self.changed()
            self.task(select, selected)

    def clear_model(self):
        if self.project:
            self.remember()
            self.project.grade.model_manifest = None
            self.project.grade.auto_method = "adaptive"
            self.project.analysis = None
            self.sync_controls()
            self.changed()

    def auto_correct(self):
        if not self.project:
            return
        if self.project.grade.input_profile == "auto":
            self.error("Choose the input color profile first. Sony log profiles cannot be inferred reliably from the file.")
            return
        self.stop_playback()
        snapshot, info = clone(self.project), self.info
        def complete(result):
            if (self.project.grade.input_profile != snapshot.grade.input_profile
                or self.project.grade.filter_hint != snapshot.grade.filter_hint
                or self.project.grade.auto_method != snapshot.grade.auto_method
                or self.project.grade.model_manifest != snapshot.grade.model_manifest):
                self.error("Input settings changed during analysis; run Auto correct again.")
                return
            self.remember()
            self.project.analysis = result
            self.update_engine()
            self.changed()
        self.task(lambda cancel, progress: analyze(info, snapshot.grade, cancel, progress), complete)

    def learn_reference(self):
        if not self.project:
            return
        path, _ = W.QFileDialog.getOpenFileName(self, "Choose the corrected version of this clip", "",
                                               "Video (*.mp4 *.mov *.mkv);;All files (*)")
        if not path:
            return
        from .analysis import learn_reference
        self.stop_playback()
        snapshot, generation, info = clone(self.project), self.generation, self.info
        def complete(result):
            if generation != self.generation:
                self.error("Editing settings changed while learning; choose the reference again.")
                return
            self.remember()
            self.project.analysis = result
            self.update_engine()
            self.changed()
        self.task(lambda cancel, progress: learn_reference(info, snapshot.grade, path, cancel, progress), complete)

    def cloud_advice(self):
        if not self.project:
            return
        from .cloud import advise_project, apply_advice
        dialog = W.QDialog(self)
        dialog.setWindowTitle("Cloud grading advice")
        layout = W.QVBoxLayout(dialog)
        label = W.QLabel("Send three JPEG proxy frames (up to 640 px) to the selected API.\n"
                        "Uses OPENAI_API_KEY or ANTHROPIC_API_KEY from your environment.\n"
                        "The full video and audio remain local.")
        label.setWordWrap(True)
        layout.addWidget(label)
        provider = self.combo(layout, "Provider", {"openai": "OpenAI", "anthropic": "Claude / Anthropic"})
        layout.addWidget(W.QLabel("Vision-capable API model ID"))
        model = W.QLineEdit()
        model.setPlaceholderText("Enter a model available to your API account")
        layout.addWidget(model)
        layout.addWidget(W.QLabel("Desired look"))
        preference = W.QLineEdit("Natural, balanced, vivid underwater footage")
        layout.addWidget(preference)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Cancel)
        send = buttons.addButton("Send frames and get advice", W.QDialogButtonBox.ButtonRole.AcceptRole)
        send.setEnabled(False)
        model.textChanged.connect(lambda value: send.setEnabled(bool(value.strip())))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != W.QDialog.DialogCode.Accepted:
            return
        self.stop_playback()
        snapshot, generation = clone(self.project), self.generation
        def complete(proposal):
            if generation != self.generation:
                self.error("Editing settings changed during the request; the cloud proposal was not applied.")
                return
            values = proposal["values"]
            message = (proposal["summary"]+f"\n\nExposure: {values['exposure']:+.2f} EV"
                       f"\nContrast: {values['contrast']:g}\nSaturation: {values['saturation']:g}"
                       "\n\nApply this proposal? Manual keyframes will be replaced; Undo restores them.")
            if W.QMessageBox.question(self, "Review cloud grade", message) == W.QMessageBox.StandardButton.Yes:
                self.remember()
                self.project.grade = apply_advice(snapshot, proposal).grade
                self.sync_controls()
                self.changed()
        self.task(lambda cancel, progress: advise_project(snapshot, provider.currentData(), model.text().strip(),
                                                         preference.text(), cancel, progress), complete)

    def export_video(self):
        if not self.project:
            return
        profile = self.export_profile.currentData()
        extension = RENDER_PROFILES[profile][1]
        proposed = str(Path(self.project.source).with_name(Path(self.project.source).stem+"_corrected"+extension))
        path, _ = W.QFileDialog.getSaveFileName(self, "Export corrected video", proposed, f"Video (*{extension})")
        if not path:
            return
        overwrite = Path(path).exists()
        # QFileDialog already obtains explicit overwrite confirmation.
        snapshot = clone(self.project)
        self.stop_playback()
        self.task(lambda cancel, progress: render(snapshot, path, profile, cancel, progress, overwrite),
                  lambda output: self.statusBar().showMessage(f"Export complete: {output}"))

    def batch(self):
        self.batch_dialog = BatchDialog(self)
        self.batch_dialog.exec()

    def request_preview(self):
        if not self.project or self.playback:
            return
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_pending = True
            return
        generation, position, info, snapshot = self.generation, self.position, self.info, clone(self.project)
        def preview(cancel, progress):
            raw = read_frame(info, position, 960, cancel)
            # An unconfirmed clip has a clearly provisional Rec.709 preview.
            if snapshot.grade.input_profile == "auto":
                snapshot.grade.input_profile = "rec709"
            raw = prepare_frame(raw, info, snapshot.grade.input_profile)
            original = encode(decode(raw, snapshot.grade.input_profile), "rec709", snapshot.grade.input_profile != "rec709")
            return original, correct(raw, snapshot.grade, position, snapshot.analysis, preview=True)
        worker = Worker(preview, self)
        self.preview_worker = worker
        def ready(images):
            if generation == self.generation and position == self.position:
                self.display(*images, position)
        worker.completed.connect(ready)
        worker.failed.connect(lambda message: self.statusBar().showMessage(message))
        worker.finished.connect(self.preview_finished)
        self.workers.append(worker)
        worker.start()

    def preview_finished(self):
        self.preview_worker = None
        worker = self.sender()
        self.workers = [item for item in self.workers if item is not worker]
        worker.deleteLater()
        if self.preview_pending:
            self.preview_pending = False
            self.request_preview()

    def display(self, original, corrected, position):
        self.viewer.original, self.viewer.corrected = qimage(original), qimage(corrected)
        self.viewer.update()
        self.scopes.data = corrected
        self.scopes.update()
        self.position = position
        self.timeline.blockSignals(True)
        self.timeline.setValue(round(position/self.info.duration*10000))
        self.timeline.blockSignals(False)
        self.time_label.setText(self.timestamp(position)+" / "+self.timestamp(self.info.duration))
        if self.playback:
            self.sync_controls()

    @staticmethod
    def timestamp(seconds):
        hours, rem = divmod(max(seconds, 0), 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

    def seek(self, value):
        if self.info:
            self.position = min(value/10000*self.info.duration, max(0, self.info.duration-1/float(self.info.fps)))
            self.sync_controls()
            self.preview_timer.start()

    def toggle_playback(self):
        if self.playback:
            self.stop_playback()
        elif self.project:
            if self.project.grade.input_profile == "auto":
                self.error("Confirm the input color profile before playback.")
                return
            playback = Playback(self.info, self.project, self.position, self)
            self.playback = playback
            playback.frame.connect(lambda original, corrected, position: self.display(original, corrected, position)
                                   if self.playback is playback else None)
            playback.failed.connect(self.error)
            playback.audio_start.connect(lambda position: self.start_audio(position) if self.playback is playback else None)
            playback.finished.connect(lambda: self.playback_finished(playback))
            playback.start()
            self.play_button.setText("Ⅱ Pause")

    def start_audio(self, position):
        if self.audio_player:
            self.audio_player.setPosition(round((position+self.info.start-self.info.format_start)*1000))
            self.audio_player.play()

    def stop_playback(self):
        if self.playback:
            self.playback.cancel.set()
            self.playback.wait(3000)
            self.playback = None
        if self.audio_player:
            self.audio_player.pause()
        self.play_button.setText("▶ Play")

    def playback_finished(self, worker):
        if self.playback is worker:
            self.stop_playback()
        worker.deleteLater()

    def change_view(self, mode):
        self.viewer.mode = mode
        self.viewer.update()

    def change_zoom(self, index):
        self.viewer.zoom = [0, .5, 1, 2][index]
        self.viewer.pan = C.QPointF(0, 0)
        self.viewer.update()

    def change_scope(self, text):
        self.scopes.setVisible(text != "Scopes hidden")
        self.scopes.mode = text
        self.scopes.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.open_video(path=urls[0].toLocalFile())

    def about(self):
        W.QMessageBox.information(self, "About UWCC", f"UWCC {__version__} · GPL-3.0-or-later\n\n"
            "Local underwater color correction.\nClassical adaptive engine included; optional local ONNX model.\n"
            "Shared float32 CPU correction for preview and render.\n\n"
            "Current limitations: no bundled ML weights;\n"
            "CPU rendering; SDR preview for HDR media; no certified HDR monitor path.\n"
            "See docs/implementation-status.md for specification coverage.")

    def closeEvent(self, event):
        if not self.confirm_discard():
            event.ignore()
            return
        self.stop_playback()
        if self.preview_worker:
            self.preview_worker.cancel.set()
            self.preview_worker.wait(3000)
        self.autosave.stop()
        event.accept()


STYLE = """
QMainWindow, QDialog, QWidget { background: #141e2a; color: #e2ebf1; }
QToolBar { spacing: 7px; padding: 8px; border-bottom: 1px solid #304357; }
QToolButton, QPushButton { background: #243747; border: 1px solid #3d5669; padding: 7px 10px; border-radius: 4px; }
QToolButton:hover, QPushButton:hover { background: #365365; border-color: #63ddcf; }
QPushButton:disabled, QToolButton:disabled { color: #687e8d; }
QComboBox, QDoubleSpinBox { background: #0d1722; border: 1px solid #3b5366; padding: 5px; border-radius: 3px; }
QAbstractItemView { background-color: #141e2a; alternate-background-color: #1d2d3d; color: #e2ebf1;
                    selection-background-color: #365365; selection-color: #ffffff; }
QHeaderView::section, QTableCornerButton::section { background-color: #243747; color: #e2ebf1;
                                                border: 1px solid #3d5669; padding: 5px; }
QSlider::groove:horizontal { background: #304659; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #67d6c8; width: 12px; margin: -5px 0; border-radius: 6px; }
QProgressBar { border: 1px solid #3b5366; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #287f7b; }
QScrollArea { border: none; }
QStatusBar { border-top: 1px solid #304357; }
QToolTip { background: #253c4f; color: white; border: 1px solid #67d6c8; }
"""


def launch(path=None):
    app = W.QApplication.instance() or W.QApplication(sys.argv[:1])
    app.setApplicationName("UWCC")
    app.setOrganizationName("UWCC")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if path:
        C.QTimer.singleShot(0, lambda: window.open_video(path=path))
    else:
        C.QTimer.singleShot(0, window.recover)
    return app.exec()
