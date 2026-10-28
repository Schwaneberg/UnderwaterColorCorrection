"""Streaming background stabilization and optional five-frame neural denoising."""
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait
import math
import time
import os
import threading
from pathlib import Path

import numpy as np

from .errors import Cancelled, UWCCError
from .neural import artifact, Network


def opencv():
    try:
        import cv2
    except ImportError as exc:
        raise UWCCError("OpenCV is required for stabilization/AI denoise (install the 'vision' extra).") from exc
    return cv2


class BackgroundStabilizer:
    """Robust background tracking with a smoothed translation/rotation path.

    This is geometric vision, not a neural network. Float RGB is warped without
    an 8-bit round trip. Only the tracking proxy is quantized.
    """
    def __init__(self, fps, region="seabed", smoothing=.2):
        self.cv = opencv()
        self.fps, self.region, self.smoothing = float(fps), region, smoothing
        self.previous = None
        self.pose = np.eye(3)
        self.smooth = np.zeros(3)
        self.previous_angle = 0.
        self.last_inliers = 0

    def estimate(self, frame):
        cv = self.cv
        height, width = frame.shape[:2]
        scale = min(1, 640/max(height, width))
        proxy = cv.resize(frame, (max(8, round(width*scale)), max(8, round(height*scale))))
        gray = cv.cvtColor(np.uint8(np.clip(proxy, 0, 1)*255), cv.COLOR_RGB2GRAY)
        current_hist = np.bincount((gray.ravel()//16), minlength=16)/gray.size
        cut = False
        if self.previous is not None:
            previous_hist = np.bincount((self.previous.ravel()//16), minlength=16)/gray.size
            cut = (np.mean(abs(gray.astype(float)-self.previous.astype(float))) > 45
                   and abs(current_hist-previous_hist).sum() > .4)
        if self.previous is None or cut:
            self.previous = gray
            self.pose = np.eye(3)
            self.smooth[:] = 0
            self.previous_angle = 0
            return np.eye(3)
        h, w = gray.shape
        points = []
        top = .5 if self.region == "seabed" else .05
        # Distribute features spatially so one diver does not own the fit.
        for y0, y1 in zip(np.linspace(top, .95, 3)[:-1], np.linspace(top, .95, 3)[1:]):
            for x0, x1 in zip(np.linspace(.05, .95, 5)[:-1], np.linspace(.05, .95, 5)[1:]):
                mask = np.zeros_like(gray)
                mask[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 255
                p = cv.goodFeaturesToTrack(self.previous, maxCorners=40, qualityLevel=.008,
                                           minDistance=5, mask=mask, blockSize=7)
                if p is not None:
                    points.append(p)
        motion = np.eye(3)
        self.last_inliers = 0
        if points:
            p = np.concatenate(points)
            q, status, _ = cv.calcOpticalFlowPyrLK(self.previous, gray, p, None, winSize=(21, 21), maxLevel=3)
            back, good, _ = cv.calcOpticalFlowPyrLK(gray, self.previous, q, None, winSize=(21, 21), maxLevel=3)
            valid = (status.ravel() > 0) & (good.ravel() > 0) & (np.linalg.norm(p-back, axis=(1, 2)) < 1)
            if valid.sum() >= 8:
                matrix, inliers = cv.estimateAffinePartial2D(p[valid], q[valid], method=cv.RANSAC,
                                                           ransacReprojThreshold=1.5, maxIters=1000)
                if matrix is not None and inliers is not None and inliers.sum() >= max(8, valid.sum()*.5):
                    angle = math.atan2(matrix[1, 0], matrix[0, 0])
                    zoom = math.hypot(matrix[0, 0], matrix[1, 0])
                    if .97 < zoom < 1.03 and abs(angle) < .06 and np.linalg.norm(matrix[:, 2]) < max(w, h)*.12:
                        # Suppress scale breathing; retain observed rigid motion.
                        motion[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
                        motion[:2, 2] = matrix[:, 2]/scale
                        self.last_inliers = int(inliers.sum())
        self.previous = gray
        self.pose = motion @ self.pose
        angle = math.atan2(self.pose[1, 0], self.pose[0, 0])
        angle = self.previous_angle + (angle-self.previous_angle+math.pi)%(2*math.pi)-math.pi
        self.previous_angle = angle
        position = np.array([self.pose[0, 2], self.pose[1, 2], angle])
        alpha = 1-math.exp(-1/(self.fps*self.smoothing))
        self.smooth += alpha*(position-self.smooth)
        desired = np.eye(3)
        a = self.smooth[2]
        desired[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        desired[:2, 2] = self.smooth[:2]
        correction = desired @ np.linalg.inv(self.pose)
        correction[0, 2] = np.clip(correction[0, 2], -.08*width, .08*width)
        correction[1, 2] = np.clip(correction[1, 2], -.08*height, .08*height)
        return correction

    def process(self, frame):
        """Standalone conservative crop; exports use the minimal fixed plan below."""
        height, width = frame.shape[:2]
        matrix = safe_center(self.estimate(frame), width, height)
        required = required_zoom(matrix, width, height)
        self.zoom = max(getattr(self, "zoom", 1.), required)
        return warp_cropped(frame, matrix, self.zoom)


def safe_center(matrix, width, height):
    """Keep the inverse crop center in the image even for extreme rotations."""
    matrix = np.asarray(matrix, np.float64).copy()
    center = np.array([(width-1)/2, (height-1)/2])
    inverse_center = (np.linalg.inv(matrix) @ np.r_[center, 1])[:2]
    margin = min(1.01, (width-1)*.1, (height-1)*.1)
    valid_center = np.clip(inverse_center, [2*margin, 2*margin],
                           [width-1-2*margin, height-1-2*margin])
    if not np.array_equal(inverse_center, valid_center):
        matrix[:2, 2] = center-matrix[:2, :2] @ valid_center
    return matrix


def required_zoom(matrix, width, height):
    """Exact corner bounds for an affine warp plus a bilinear sampling margin."""
    inverse = np.linalg.inv(matrix)
    center = np.array([(width-1)/2, (height-1)/2])
    origin = (inverse @ np.r_[center, 1])[:2]
    margin = min(1.01, (width-1)*.1, (height-1)*.1)
    lower, upper = np.array([margin, margin]), np.array([width-1-margin, height-1-margin])
    zoom = 1.
    for x, y in ((0, 0), (width-1, 0), (0, height-1), (width-1, height-1)):
        direction = inverse[:2, :2] @ (np.array([x, y])-center)
        for axis in range(2):
            room = upper[axis]-origin[axis] if direction[axis] >= 0 else origin[axis]-lower[axis]
            if room <= 0:
                raise UWCCError("Unable to find a valid stabilization crop.")
            zoom = max(zoom, abs(direction[axis])/room)
    return zoom*1.00002


def crop_matrix(matrix, width, height, zoom):
    center = np.array([(width-1)/2, (height-1)/2])
    crop = np.eye(3)
    crop[0, 0] = crop[1, 1] = zoom
    crop[:2, 2] = center*(1-zoom)
    return crop @ matrix


def warp_cropped(frame, matrix, zoom):
    cv = opencv()
    height, width = frame.shape[:2]
    transform = crop_matrix(matrix, width, height, zoom)
    # Constant fill is a safeguard only: the inverse corner bounds guarantee
    # that every output sample lies in the original image. No mirrored pixels.
    return cv.warpAffine(frame, transform[:2].astype(np.float32), (width, height),
                         flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)


@dataclass
class StabilizationPlan:
    matrices: list
    width: int
    height: int
    zoom: float

    def warp(self, frames):
        count = 0
        for count, frame in enumerate(frames, 1):
            if count > len(self.matrices):
                raise UWCCError("Stabilization plan and decoder frame counts disagree.")
            height, width = frame.shape[:2]
            scale = np.diag([width/self.width, height/self.height, 1.])
            matrix = scale @ self.matrices[count-1] @ np.linalg.inv(scale)
            yield warp_cropped(frame, matrix, self.zoom)
        if count != len(self.matrices):
            raise UWCCError("Stabilization plan and decoder frame counts disagree.")


def build_stabilization_plan(info, grade, cancel=None, progress=lambda *_: None):
    """Analyze once, then choose one minimal, edge-free zoom for the whole clip."""
    from .media import FrameReader
    tracker = BackgroundStabilizer(info.fps, grade.stabilization_region)
    matrices, zoom = [], 1.
    with FrameReader(info, max_width=640, cancel=cancel, decoder=grade.video_decoder) as reader:
        scale = np.diag([info.width/reader.width, info.height/reader.height, 1.])
        for index, frame in enumerate(reader):
            if cancel and cancel.is_set():
                raise Cancelled("Stabilization analysis canceled.")
            matrix = scale @ tracker.estimate(frame) @ np.linalg.inv(scale)
            matrix = safe_center(matrix, info.width, info.height)
            matrices.append(matrix)
            zoom = max(zoom, required_zoom(matrix, info.width, info.height))
            if index % max(1, int(float(info.fps))) == 0:
                progress(min(.99, index/max(1, info.duration*float(info.fps))),
                         f"Analyzing stabilization crop · frame {index+1}")
    if not matrices:
        raise UWCCError("No frames available for stabilization analysis.")
    progress(1, f"Stabilization crop: {zoom:.3f}× fixed zoom")
    return StabilizationPlan(matrices, info.width, info.height, zoom)


class TemporalDenoiser:
    def __init__(self, manifest, strength, cancel=None, progress=None):
        data, path = artifact(manifest, "denoise")
        if data.get("frames") != 5:
            raise UWCCError("Temporal denoising requires a five-frame FastDVDnet model.")
        self.network = Network(data, path)
        self.cpu_artifact = (data, path)
        self.gpu_variant = False
        self.precision = "float32"
        if data.get("gpu_variant_manifest") and self.network.uses_cuda:
            variant = data["gpu_variant_manifest"]
            if os.environ.get("UWCC_GPU_PRECISION") == "float32":
                variant = "gpu-float32.json"
            manifest_path = Path(manifest).resolve()
            candidate = (manifest_path.parent/variant).resolve()
            if candidate.parent != manifest_path.parent:
                raise UWCCError("GPU denoise variant must be beside its manifest.")
            try:
                gpu_data, gpu_path = artifact(candidate, "denoise")
                if not gpu_data.get("gpu_validated") or gpu_data.get("frames") != 5:
                    raise UWCCError("GPU denoise variant has not been validated.")
                network = Network(gpu_data, gpu_path, allow_cpu_fallback=False)
                if not network.uses_cuda:
                    raise UWCCError("GPU denoise variant requires CUDA.")
                self.network, data = network, gpu_data
                self.gpu_variant = True
                self.precision = data.get("precision", "float32")
            except UWCCError:
                if progress:
                    progress("GPU full-frame model unavailable; using the standard tiled model.")
        self.size = data["size"]
        self.halo = 64 if self.gpu_variant else 16
        self.sigma = float(strength)*2.5/255
        self.cancel = cancel
        self.progress = progress or (lambda *_: None)
        self.last_progress = -float("inf")
        self.frame_number = 1

    def report(self, done, total):
        now = time.monotonic()
        if now-self.last_progress >= .5:
            self.last_progress = now
            mode = f" · {self.precision}" if self.gpu_variant else ""
            self.progress(f"AI denoise · {self.network.execution_label}{mode} · frame {self.frame_number} · tile {done}/{total}")

    def predict(self, frames, prepared=None):
        if self.cancel and self.cancel.is_set():
            raise Cancelled("AI denoising canceled.")
        try:
            if self.gpu_variant and frames[2].shape[0]*frames[2].shape[1] <= 1920*1080:
                return self.predict_full_frame(frames, prepared)
            return self.predict_tiles(frames)
        except Cancelled:
            raise
        except UWCCError:
            if not self.gpu_variant:
                raise
            self.progress("GPU denoise failed; falling back to CPU tiles.")
            data, path = self.cpu_artifact
            self.network = Network(data, path, cpu_only=True)
            self.gpu_variant = False
            self.size, self.halo = data["size"], 16
            self.last_progress = -float("inf")
            return self.predict_tiles(frames)

    def prepare_full_frame(self, frames):
        height, width = frames[2].shape[:2]
        ph, pw = (-height) % 4, (-width) % 4
        # Write directly into contiguous NCHW storage, avoiding five padded
        # frame copies plus a second full-size concatenate/contiguous copy.
        batch = np.empty((1, 15, height+ph, width+pw), np.float32)
        for index, frame in enumerate(frames):
            if ph or pw:
                frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            batch[0, index*3:(index+1)*3] = frame.transpose(2, 0, 1)
        noise = np.full((1, 1, height+ph, width+pw), self.sigma, np.float32)
        return {"frames": batch, "noise": noise}

    def predict_full_frame(self, frames, prepared=None):
        height, width = frames[2].shape[:2]
        values = self.prepare_full_frame(frames) if prepared is None else prepared
        shape = values["frames"].shape
        self.report(0, 1)
        result = self.network.run(values)
        if result.shape != (1, 3, shape[2], shape[3]):
            raise UWCCError("GPU denoise returned an invalid shape.")
        self.report(1, 1)
        return np.clip(result[0, :, :height, :width].transpose(1, 2, 0), 0, 1)

    def predict_tiles(self, frames):
        # Convert model-domain pixels once before this method (see stream).
        height, width = frames[2].shape[:2]
        size, halo = self.size, self.halo
        core = size-2*halo
        pads = ((halo, size), (halo, size), (0, 0))
        padded = [np.pad(f, pads, mode="reflect") for f in frames]
        output = np.empty_like(frames[2])
        noise = np.full((1, 1, size, size), self.sigma, np.float32)
        total = math.ceil(height/core)*math.ceil(width/core)
        done = 0
        self.report(done, total)
        for y in range(0, height, core):
            for x in range(0, width, core):
                if self.cancel and self.cancel.is_set():
                    raise Cancelled("AI denoising canceled.")
                tensors = [f[y:y+size, x:x+size].transpose(2, 0, 1) for f in padded]
                batch = np.ascontiguousarray(np.concatenate(tensors, axis=0)[None], np.float32)
                result = self.network.run({"frames": batch, "noise": noise})
                if result.shape != (1, 3, size, size):
                    raise UWCCError("Denoising model returned an invalid image shape.")
                h, w = min(core, height-y), min(core, width-x)
                output[y:y+h, x:x+w] = result[0, :, halo:halo+h, halo:halo+w].transpose(1, 2, 0)
                done += 1
                self.report(done, total)
        return np.clip(output, 0, 1)

    def model_windows(self, frames):
        from .color import bt709_decode, srgb_encode
        window = deque(maxlen=5)
        last = None
        for frame in frames:
            if self.cancel and self.cancel.is_set():
                raise Cancelled("AI denoising canceled.")
            current = srgb_encode(bt709_decode(frame)).astype(np.float32)
            if last is None:
                window.extend([current, current])
            window.append(current)
            last = current
            if len(window) == 5:
                yield list(window)
        if last is not None:
            for _ in range(2):
                window.append(last)
                if len(window) == 5:
                    yield list(window)

    def stream(self, frames, *, prefetch=None, on_stop=None):
        """Prepare one future temporal window while the caller runs inference.

        The source must be read only by the preparation worker. on_stop must
        unblock a blocking source read (FrameReader kills its decoder), so
        cancellation and early generator close can join the worker safely.
        """
        from .color import bt709_encode, srgb_decode
        if prefetch is None:
            prefetch = os.environ.get("UWCC_DENOISE_PREFETCH", "1") != "0"
        windows = self.model_windows(frames)
        stopped = threading.Event()

        def prepare_next():
            if stopped.is_set():
                return None
            if self.cancel and self.cancel.is_set():
                raise Cancelled("AI denoising canceled.")
            window = next(windows, None)
            if window is None or stopped.is_set():
                return None
            prepared = None
            if self.gpu_variant and window[2].shape[0]*window[2].shape[1] <= 1920*1080:
                prepared = self.prepare_full_frame(window)
            return window, prepared

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uwcc-denoise-prepare") if prefetch else None
        future = None
        try:
            if pool:
                future = pool.submit(prepare_next)
            index = 0
            while True:
                if self.cancel and self.cancel.is_set():
                    raise Cancelled("AI denoising canceled.")
                if pool:
                    if not wait([future], timeout=.1).done:
                        continue
                    item = future.result()
                else:
                    item = prepare_next()
                if item is None:
                    break
                if pool:
                    # Only this next window may be prepared ahead of inference.
                    future = pool.submit(prepare_next)
                window, prepared = item
                self.frame_number = index+1
                result = self.predict(window) if prepared is None else self.predict(window, prepared=prepared)
                # CPU fallback ignores prefetched GPU tensors but retains the
                # exact same temporal windows and output ordering.
                output = bt709_encode(srgb_decode(result)).astype(np.float32)
                del item, window, prepared, result
                yield output
                index += 1
                del output
        finally:
            stopped.set()
            if future:
                future.cancel()
            try:
                if on_stop:
                    on_stop()
            finally:
                if pool:
                    pool.shutdown(wait=True, cancel_futures=True)
                windows.close()
                close = getattr(frames, "close", None)
                if close:
                    close()
