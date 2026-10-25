"""Verified ONNX inference, shared by enhancement and temporal denoising."""
import hashlib
import os
from pathlib import Path

import numpy as np

from .errors import UWCCError
from .project import read_json

ALLOWED = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "GPL-3.0-or-later"}


def builtin_model(task):
    root = Path(os.environ.get("UWCC_MODEL_DIR", Path(__file__).resolve().parent.parent/"models"))
    path = root/("udnet" if task == "enhance" else "fastdvdnet")/"manifest.json"
    return str(path) if path.is_file() else None


def artifact(manifest, task):
    manifest = Path(manifest).resolve()
    data = read_json(manifest)
    for name in ("code_license", "weights_license"):
        if data.get(name) not in ALLOWED:
            raise UWCCError(f"Unverified or unsupported model {name}.")
    if data.get("task", "enhance") != task:
        raise UWCCError(f"Expected an ONNX model for {task}.")
    path = (manifest.parent/data.get("file", "model.onnx")).resolve()
    if path.parent != manifest.parent or not path.is_file():
        raise UWCCError("Model must be a local ONNX file beside its manifest.")
    if path.stat().st_size > 512*1024*1024:
        raise UWCCError("Model exceeds the 512 MiB limit.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != data.get("onnx_sha256"):
        raise UWCCError("Model checksum does not match the manifest.")
    size = data.get("size", 256)
    if type(size) is not int or not 32 <= size <= 1024 or size % 4:
        raise UWCCError("Model size must be a multiple of four from 32 to 1024.")
    if data.get("color_space") != "srgb" or data.get("layout") != "NCHW":
        raise UWCCError("This engine requires an sRGB float32 NCHW model.")
    return data, path


class Network:
    def __init__(self, data, path, cpu_only=False, allow_cpu_fallback=True):
        self.backend = data.get("runtime", "onnxruntime")
        self.warning = ""
        self.allow_cpu_fallback = allow_cpu_fallback
        if self.backend == "opencv" and not cpu_only and os.environ.get("UWCC_INFERENCE_DEVICE", "auto") != "cpu":
            try:
                import onnxruntime as ort
                if any(p in ort.get_available_providers() for p in ("CUDAExecutionProvider", "DmlExecutionProvider")):
                    self.backend = "onnxruntime"
            except ImportError:
                pass
        try:
            if self.backend == "opencv":
                import cv2
                cv2.setNumThreads(2)
                self.net = cv2.dnn.readNetFromONNX(str(path))
            else:
                import onnxruntime as ort
                ort.disable_telemetry_events()
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                options.inter_op_num_threads = 1
                self.ort, self.path, self.options = ort, str(path), options
                providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                             if p in ort.get_available_providers()]
                if cpu_only or os.environ.get("UWCC_INFERENCE_DEVICE") == "cpu":
                    providers = ["CPUExecutionProvider"]
                if "DmlExecutionProvider" in providers:
                    # DirectML cannot initialize with memory patterns or parallel execution.
                    options.enable_mem_pattern = False
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
                    # Make pip-installed CUDA/cuDNN libraries available before
                    # ORT loads its provider. CPU-only installations skip this.
                    try:
                        ort.preload_dlls()
                    except (OSError, RuntimeError):
                        pass  # Provider initialization below determines availability.
                provider_options = [({"gpu_mem_limit": str(4*1024**3), "cudnn_conv_algo_search": "HEURISTIC",
                                      "cudnn_conv_use_max_workspace": "0"}
                                     if p == "CUDAExecutionProvider" and data.get("gpu_variant") else {}) for p in providers]
                try:
                    self.net = ort.InferenceSession(str(path), options, providers=providers, provider_options=provider_options)
                except Exception:
                    if not allow_cpu_fallback:
                        raise UWCCError("GPU model could not initialize.") from None
                    self.net = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                    self.warning = "Accelerated inference unavailable; using CPU."
                if (any(p in providers for p in ("CUDAExecutionProvider", "DmlExecutionProvider"))
                        and not any(p in self.net.get_providers() for p in ("CUDAExecutionProvider", "DmlExecutionProvider"))):
                    self.warning = "GPU provider could not initialize; using CPU."
                if not allow_cpu_fallback and not any(p in self.net.get_providers() for p in ("CUDAExecutionProvider", "DmlExecutionProvider")):
                    raise UWCCError("GPU model requires an active GPU provider.")
        except ImportError as exc:
            raise UWCCError("Install ONNX Runtime ('pip install uwcc[ml]') or OpenCV for this model's runtime.") from exc
        self.input_names = ([x.name for x in self.net.get_inputs()] if self.backend != "opencv" else None)

    @property
    def uses_cuda(self):
        return self.backend != "opencv" and "CUDAExecutionProvider" in self.net.get_providers()

    @property
    def execution_label(self):
        if self.backend == "opencv":
            return "CPU · OpenCV DNN"
        providers = self.net.get_providers()
        if "CUDAExecutionProvider" in providers:
            return "CUDA GPU · ONNX Runtime"
        if "DmlExecutionProvider" in providers:
            return "DirectML GPU · ONNX Runtime"
        return "CPU · ONNX Runtime"

    def run(self, values):
        if self.backend == "opencv":
            for name, tensor in values.items():
                self.net.setInput(np.ascontiguousarray(tensor, np.float32), name)
            result = self.net.forward()
        else:
            try:
                result = self.net.run(None, values)[0]
            except Exception:
                if not self.allow_cpu_fallback:
                    raise UWCCError("GPU model inference failed.") from None
                if self.net.get_providers() == ["CPUExecutionProvider"]:
                    raise UWCCError("ONNX inference failed on the CPU.") from None
                self.net = self.ort.InferenceSession(self.path, self.options, providers=["CPUExecutionProvider"])
                self.warning = "Accelerated inference failed; using CPU."
                result = self.net.run(None, values)[0]
        if not np.isfinite(result).all():
            raise UWCCError("Neural model returned non-finite pixels.")
        return result
