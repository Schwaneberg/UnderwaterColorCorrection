"""Convert pinned, locally downloaded upstream weights to safe runtime ONNX.

Run with the optional conversion environment; see docs/ai-workflow.md.
Only tensor state dicts are loaded (torch.load(weights_only=True)).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def export(kind, source, output):
    import onnx
    import torch
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    if kind == "udnet":
        root = source / "UDnet"
        # Training-only imports/modules are not used by the released inference graph.
        code = (root/"model_utils/UDnet.py").read_text()
        code = code.replace("from torchvision.models.vgg import vgg16", "")
        code = code.replace("from model_utils.colournet import USLN", "")
        code = code.replace("self.colornet = USLN()", "self.colornet = nn.Identity()")
        generated = root/"model_utils/inference_export.py"
        generated.write_text(code)
        sys.path.insert(0, str(root.resolve()))
        module = load_module("udnet_export", generated)
        from argparse import Namespace
        model = module.mynet(Namespace(device="cpu"), with_loss=False)
        weights = source/"UDnet.pth"
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith(("VGG16.", "decoder.physnet.", "decoder.colornet."))}
        model.load_state_dict(state, strict=True)
        class Inference(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = model.decoder
            def forward(self, image):
                return self.decoder(image, image, training=False).clamp(0, 1)
        model = Inference().eval()
        inputs = (torch.rand(1, 3, 256, 256),)
        names = ["image"]
        manifest = {"model_id": "UDnet V1", "model_version": "paper",
                    "task": "enhance", "size": 256, "color_space": "srgb", "layout": "NCHW"}
        license_path = root/"LICENSE"
    else:
        root = source/"fastdvdnet"
        module = load_module("fastdvdnet_export", root/"models.py")
        model = module.FastDVDnet().eval()
        weights = source/"fastdvdnet.pth"
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        inputs = (torch.rand(1, 15, 128, 128), torch.full((1, 1, 128, 128), .02))
        names = ["frames", "noise"]
        manifest = {"model_id": "FastDVDnet", "model_version": "CVPR2020",
                    "task": "denoise", "size": 128, "color_space": "srgb", "layout": "NCHW", "frames": 5}
        license_path = root/"LICENSE"
    path = output/"model.onnx"
    with torch.no_grad():
        expected = model(*inputs).numpy()
        torch.onnx.export(model, inputs, str(path), input_names=names, output_names=["output"],
                          opset_version=17, dynamo=False)
    # Store constant Identity aliases as initializers for OpenCV compatibility.
    import onnx
    graph = onnx.load(path)
    known = {x.name: x for x in graph.graph.initializer}
    for node in list(graph.graph.node):
        if node.op_type == "Identity" and node.input[0] in known:
            tensor = onnx.TensorProto()
            tensor.CopyFrom(known[node.input[0]])
            tensor.name = node.output[0]
            graph.graph.initializer.append(tensor)
            known[tensor.name] = tensor
            graph.graph.node.remove(node)
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    import cv2
    cv2.setNumThreads(2)
    runtime = "opencv"
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
        for name, value in zip(names, inputs):
            net.setInput(value.numpy(), name)
        actual = net.forward()
    except cv2.error:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        actual = session.run(None, {k: v.numpy() for k, v in zip(names, inputs)})[0]
        runtime = "onnxruntime"
    error = float(np.max(abs(actual-expected)))
    if not np.isfinite(actual).all() or error > .005:
        raise RuntimeError(f"ONNX conversion failed parity check: {error}")
    manifest |= {"schema_version": 1, "file": path.name, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "code_license": "MIT", "weights_license": "MIT", "runtime": runtime,
                 "upstream": json.loads((root/"revision.json").read_text()),
                 "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                 "conversion_max_error": error}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    shutil.copyfile(license_path, output/"LICENSE")
    print(kind, "converted", "max error", error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["udnet", "fastdvdnet"])
    parser.add_argument("--sources", type=Path, default=Path("models/.sources"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.kind, args.sources, args.output or Path("models")/args.kind)
