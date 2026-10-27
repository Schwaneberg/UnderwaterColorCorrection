"""Explicitly download pinned RightWayUp ONNX weights from their upstream host."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

REVISION = "69076d99f16fafbc3d7e12958390972b5f291e81"
BASE = f"https://huggingface.co/ortusai/rightwayup/resolve/{REVISION}/"
MODELS = {
    "fast": ("fast-s224-fp32.onnx", "2a61ecc83cf27f5869a4a806c7e87ade81608f7356f2b7f2d6de4c8c38c32a0d"),
    "max": ("max-l280-int8.onnx", "9e358f9c51587ec4472ac9496e52dbcaa428886613ebe338fb0654b10bdc3ccf"),
}


def download(name, destination, checksum=None):
    if checksum and destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == checksum:
        print(f"Verified cached {name}", flush=True)
        return
    partial = destination.with_name(destination.name+".partial")
    digest, size = hashlib.sha256(), 0
    try:
        with urlopen(BASE+name, timeout=120) as response, partial.open("wb") as output:
            while block := response.read(1024*1024):
                size += len(block)
                if size > 350*1024*1024:
                    raise RuntimeError("Orientation model download exceeds size limit")
                digest.update(block)
                output.write(block)
        if checksum and digest.hexdigest() != checksum:
            raise RuntimeError(f"Checksum mismatch: {name}")
        partial.replace(destination)
        print(f"Downloaded {name}: {size/1024/1024:.1f} MiB", flush=True)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, help="Root model directory; defaults to UWCC's user model storage")
    parser.add_argument("--tier", choices=["fast", "max", "both"], default="max")
    args = parser.parse_args()
    if args.models_dir is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from uwcc.project import model_directory
        args.models_dir = model_directory()
    args.models_dir = args.models_dir.expanduser()/"rightwayup"
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "NOTICE", "tiers.json", "SHA256SUMS"):
        download(name, args.models_dir/name)
    chosen = MODELS if args.tier == "both" else {args.tier: MODELS[args.tier]}
    for name, checksum in chosen.values():
        download(name, args.models_dir/name, checksum)
    (args.models_dir/"upstream.json").write_text(json.dumps({
        "repo": "ortusai/rightwayup", "revision": REVISION, "license": "Apache-2.0",
        "files": {name: checksum for name, checksum in chosen.values()},
    }, indent=2)+"\n")
    if "max" in chosen:
        name, checksum = MODELS["max"]
        (args.models_dir/"manifest.json").write_text(json.dumps({
            "schema_version": 1, "model_id": "RightWayUp Max INT8", "task": "orientation",
            "file": name, "onnx_sha256": checksum, "code_license": "Apache-2.0",
            "weights_license": "Apache-2.0", "runtime": "onnxruntime", "size": 280,
            "upstream": {"repo": "ortusai/rightwayup", "revision": REVISION},
        }, indent=2)+"\n")


if __name__ == "__main__":
    main()
