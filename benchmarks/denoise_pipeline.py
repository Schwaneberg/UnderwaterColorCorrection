"""Compare serial and prefetched denoise: python -m benchmarks.denoise_pipeline.

Measures decode, transfer conversion, tensor preparation, inference, output
conversion, and SHA-256 consumption. Does not measure full export/encoding.
Input color signaling is not inferred: --clip is interpreted as Rec.709 for
this performance comparison, not a color-accuracy assessment.
"""
import argparse
from contextlib import closing
import hashlib
from itertools import islice
import json
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np

from uwcc.finishing import TemporalDenoiser
from uwcc.media import FrameReader, probe
from uwcc.neural import builtin_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, help="Benchmark the first frames of this clip instead of synthetic pixels")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--width", type=int, default=1920, help="Synthetic frame width")
    parser.add_argument("--height", type=int, default=1080, help="Synthetic frame height")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow a CPU benchmark instead of requiring CUDA")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.width, args.height) < 1:
        parser.error("Frame count, repeats, and dimensions must be positive")
    manifest = builtin_model("denoise")
    if not manifest:
        parser.error("Prepare FastDVDnet models first")
    info = probe(args.clip) if args.clip else None
    width, height = (info.width, info.height) if info else (args.width, args.height)
    model = TemporalDenoiser(manifest, 3)
    if not args.allow_cpu and not model.network.uses_cuda:
        parser.error("An active CUDA runtime is required; use --allow-cpu for CPU measurements")
    print(f"{model.network.execution_label} · {width}×{height} · {args.frames} frames", flush=True)
    warmup = np.full((height, width, 3), .3, np.float32)
    for _ in range(2):
        model.predict([warmup]*5)
    del warmup

    stats = {}
    original_windows, original_prepare, original_run = model.model_windows, model.prepare_full_frame, model.network.run
    def windows(frames):
        source = iter(original_windows(frames))
        while True:
            start = perf_counter()
            item = next(source, None)
            stats["decode_and_transfer_seconds"] += perf_counter()-start
            if item is None:
                break
            yield item
    def prepare(frames):
        start = perf_counter()
        result = original_prepare(frames)
        stats["tensor_preparation_seconds"] += perf_counter()-start
        return result
    def infer(values):
        start = perf_counter()
        result = original_run(values)
        stats["inference_seconds"] += perf_counter()-start
        return result
    model.model_windows, model.prepare_full_frame, model.network.run = windows, prepare, infer

    def synthetic():
        rng = np.random.default_rng(8)
        for _ in range(args.frames):
            yield rng.uniform(.05, .8, (height, width, 3)).astype(np.float32)

    expected = None
    measurements = {"serial": [], "prefetched": []}
    for repeat in range(args.repeats):
        # Reverse order on alternate repeats to reduce warm-up/order bias.
        for parallel in ((False, True) if repeat % 2 == 0 else (True, False)):
            stats = {"decode_and_transfer_seconds": 0., "tensor_preparation_seconds": 0., "inference_seconds": 0.}
            digests = []
            start = perf_counter()
            reader = FrameReader(info) if info else None
            try:
                frames = islice(reader, args.frames) if reader else synthetic()
                with closing(model.stream(frames, prefetch=parallel, on_stop=reader._stop_decode if reader else None)) as stream:
                    for frame in stream:
                        digests.append(hashlib.sha256(memoryview(np.ascontiguousarray(frame))).hexdigest())
            finally:
                if reader:
                    reader.close()
            elapsed = perf_counter()-start
            if not digests:
                parser.error("Source produced no frames")
            if expected is None:
                expected = digests
            if digests != expected:
                raise RuntimeError("Serial and prefetched output checksums differ")
            if not args.allow_cpu and not model.network.uses_cuda:
                raise RuntimeError("GPU fallback invalidated the benchmark")
            name = "prefetched" if parallel else "serial"
            result = {"seconds": elapsed, "frames": len(digests), "seconds_per_frame": elapsed/len(digests), **stats}
            measurements[name].append(result)
            print(f"{name} {repeat+1}: {elapsed:.3f}s · {elapsed/len(digests):.4f}s/frame · "
                  f"preparation {stats['decode_and_transfer_seconds']+stats['tensor_preparation_seconds']:.3f}s · "
                  f"inference {stats['inference_seconds']:.3f}s", flush=True)
    serial = statistics.median(run["seconds"] for run in measurements["serial"])
    parallel = statistics.median(run["seconds"] for run in measurements["prefetched"])
    report = {"backend": model.network.execution_label, "gpu_variant": model.gpu_variant,
              "precision": model.precision, "width": width, "height": height,
              "input": str(args.clip) if args.clip else "synthetic", "measurements": measurements,
              "speedup": serial/parallel, "outputs_identical": True,
              "scope": "decode + denoise + output checksums; not a full video export"}
    print(f"Median speedup: {serial/parallel:.3f}× · all output checksums identical", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
