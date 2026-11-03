"""Validate GPU FP16 accuracy against GPU FP32 before enabling GPU variants."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def validate(root):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from uwcc.errors import UWCCError
    from uwcc.neural import Network, artifact
    import onnxruntime as ort

    if os.environ.get("UWCC_INFERENCE_DEVICE") == "cpu":
        raise UWCCError("CUDA validation requires GPU inference. Set UWCC_INFERENCE_DEVICE=auto and retry.")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise UWCCError(
            f"CUDA validation is running in {sys.executable}, which has no CUDA execution provider. "
            "Install the NVIDIA runtime into this environment, then restart UWCC:\n"
            f'"{sys.executable}" -m pip uninstall -y onnxruntime onnxruntime-directml\n'
            f'"{sys.executable}" -m pip install "onnxruntime-gpu[cuda,cudnn]>=1.21"'
        )
    folder = root/"fastdvdnet"
    networks, paths = {}, {}
    for precision in ("float32", "float16"):
        manifest = folder/f"gpu-{precision}.json"
        data, path = artifact(manifest, "denoise")
        try:
            net = Network(data, path, allow_cpu_fallback=False)
        except UWCCError as exc:
            raise UWCCError(
                "CUDA model initialization failed. Check the NVIDIA driver and the CUDA/cuDNN "
                f"libraries installed in {sys.executable}. Runtime details: {exc}"
            ) from exc
        if not net.uses_cuda:
            raise UWCCError("GPU validation requires an active CUDA execution provider")
        networks[precision], paths[precision] = net, manifest
    rng = np.random.default_rng(8)
    worst, mean = 0., 0.
    for low, high in ((0, 1), (0, .1), (.8, 1)):
        values = {"frames": rng.uniform(low, high, (1, 15, 128, 128)).astype(np.float32),
                  "noise": np.full((1, 1, 128, 128), 5/255, np.float32)}
        a = networks["float32"].run(values)
        b = networks["float16"].run(values)
        error = abs(a-b)
        worst, mean = max(worst, float(error.max())), max(mean, float(error.mean()))
        if not np.isfinite(b).all() or error.max() > .005:
            raise RuntimeError("FP16 exceeded the GPU accuracy tolerance")
    for path in paths.values():
        data = json.loads(path.read_text())
        data.update(gpu_validated=True, gpu_float16_max_error=worst, gpu_float16_max_mean_error=mean)
        path.write_text(json.dumps(data, indent=2)+"\n")
    path = folder/"manifest.json"
    data = json.loads(path.read_text())
    data["gpu_variant_manifest"] = "gpu-float16.json"
    path.write_text(json.dumps(data, indent=2)+"\n")
    print(f"CUDA GPU variants enabled. FP16 maximum error {worst:.6f}, mean {mean:.6f}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from uwcc.errors import UWCCError
    try:
        validate(args.models_dir)
    except (UWCCError, RuntimeError) as exc:
        parser.exit(1, f"CUDA validation failed: {exc}\n")
