"""Prefer LGPL PySide6; support system-installed GPL PyQt6 as an offline option."""

__all__ = ["QtCore", "QtGui", "QtWidgets", "Signal", "Slot", "multimedia"]

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Signal, Slot
except ImportError:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


def multimedia():
    try:
        if QtCore.__name__.startswith("PySide6"):
            from PySide6 import QtMultimedia
        else:
            from PyQt6 import QtMultimedia
        return QtMultimedia
    except ImportError:
        return None
