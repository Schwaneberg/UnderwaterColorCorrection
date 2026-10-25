"""Optional offline ONNX reference generator. Never loads pickle checkpoints."""
import numpy as np
from PIL import Image

from .errors import UWCCError
from .neural import ALLOWED, artifact, Network  # ALLOWED retained for integrations

__all__ = ["ALLOWED", "ReferenceModel"]


class ReferenceModel:
    def __init__(self, manifest):
        data, path = artifact(manifest, "enhance")
        self.size = data.get("size", 256)
        self.network = Network(data, path)
        self.name = data.get("model_id", "Local model")+" "+data.get("model_version", "")
        self.input_name = (self.network.input_names or ["image"])[0]
        self.warning = self.network.warning

    def predict(self, rgb):
        image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255)).resize((self.size, self.size), Image.Resampling.BILINEAR)
        tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None]/255
        output = self.network.run({self.input_name: tensor})
        self.warning = self.network.warning
        if output.shape != tensor.shape:
            raise UWCCError("Model returned an invalid reference image.")
        result = Image.fromarray(np.uint8(np.clip(output[0].transpose(1, 2, 0), 0, 1)*255))
        return np.asarray(result.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)/255
