"""Prepare dynamic-shape FP32/FP16 FastDVDnet variants; validation enables them."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from convert_models import load_module


def sort_graph(model):
    known = {x.name for x in model.graph.input} | {x.name for x in model.graph.initializer}
    remaining, nodes = list(model.graph.node), []
    while remaining:
        ready = [n for n in remaining if all(not x or x in known for x in n.input)]
        if not ready:
            raise RuntimeError("GPU model graph cannot be topologically sorted")
        for node in ready:
            nodes.append(node)
            known.update(node.output)
            remaining.remove(node)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)


def prepare(root):
    import torch
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16
    torch.set_num_threads(2)
    torch.manual_seed(7)
    folder = root/"fastdvdnet"
    metadata = json.loads((folder/"manifest.json").read_text())
    weights = root/".sources/fastdvdnet.pth"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["weights_sha256"]:
        raise RuntimeError("FastDVDnet source weights failed checksum validation")
    module = load_module("fastdvdnet_gpu_export", root/".sources/fastdvdnet/models.py")
    model = module.FastDVDnet().eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    inputs = (torch.rand(1, 15, 128, 128), torch.full((1, 1, 128, 128), .02))
    path = folder/"model-gpu-float32.onnx"
    with torch.no_grad():
        expected = model(*inputs).numpy()
        torch.onnx.export(model, inputs, str(path), input_names=["frames", "noise"], output_names=["output"],
                          opset_version=17, dynamo=False,
                          dynamic_axes={k: {0: "batch", 2: "height", 3: "width"} for k in ("frames", "noise", "output")})
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"frames": inputs[0].numpy(), "noise": inputs[1].numpy()})[0]
    error = float(abs(actual-expected).max())
    if error > .005 or not np.isfinite(actual).all():
        raise RuntimeError("Dynamic model failed PyTorch parity validation")
    half = convert_float_to_float16(graph, keep_io_types=True)
    sort_graph(half)
    onnx.checker.check_model(half)
    onnx.save(half, folder/"model-gpu-float16.onnx")
    for precision in ("float32", "float16"):
        filename = f"model-gpu-{precision}.onnx"
        data = {k: v for k, v in metadata.items() if k != "gpu_variant_manifest"}
        data.update(file=filename, onnx_sha256=hashlib.sha256((folder/filename).read_bytes()).hexdigest(),
                    runtime="onnxruntime", size=512, precision=precision, gpu_variant=True,
                    gpu_validated=False, dynamic_float32_max_error=error)
        (folder/f"gpu-{precision}.json").write_text(json.dumps(data, indent=2)+"\n")
    print("GPU models prepared. Run tools/validate_gpu_denoise.py with a CUDA runtime to enable them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    prepare(parser.parse_args().models_dir)
