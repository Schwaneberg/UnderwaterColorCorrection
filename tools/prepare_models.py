"""Download pinned MIT model sources/weights and convert them locally to ONNX.

Explicit setup command; the application never downloads models automatically.
Conversion requires torch (CPU suffices), onnx, OpenCV, and onnxruntime.
Use --download-only to fetch upstream files with just the Python standard library.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

SOURCES = {
    "UDnet": {"repo": "alzayats/UDnet", "sha": "d6b196bfef05b91eca0a05925686276a85af5f6c",
              "files": ["LICENSE", "model_utils/UDnet.py", "model_utils/utils.py"]},
    "fastdvdnet": {"repo": "m-tassano/fastdvdnet", "sha": "c8fdf6182a0340e89dd18f5df25b47337cbede6f",
                   "files": ["LICENSE", "models.py"]},
}
WEIGHTS = {
    "udnet": ("UDnet.pth", "https://huggingface.co/Alzayats/UDnet/resolve/3c5816281b881dad1aeb3abbfdf5200da4a2fb27/UDnet.pth",
              "4ab7c8e7a9578c986b583259e3968684e6a00ad179df5bcd0725e282e0519816"),
    "fastdvdnet": ("fastdvdnet.pth", "https://raw.githubusercontent.com/m-tassano/fastdvdnet/"
                   "c8fdf6182a0340e89dd18f5df25b47337cbede6f/model.pth",
                   "9d9d8413c33e3d9d961d07c530237befa1197610b9d60602ff42fd77975d2a17"),
}


def download(url, path, checksum=None):
    if path.is_file() and checksum and hashlib.sha256(path.read_bytes()).hexdigest() == checksum:
        return
    with urlopen(url, timeout=90) as response:
        data = response.read(64*1024*1024+1)
    if len(data) > 64*1024*1024:
        raise RuntimeError("Download exceeds size limit")
    if checksum and hashlib.sha256(data).hexdigest() != checksum:
        raise RuntimeError("Downloaded weights failed checksum validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--download-only", action="store_true",
                        help="Download pinned upstream sources and verified weights without converting")
    args = parser.parse_args(argv)
    source = args.models_dir/".sources"
    for name, metadata in SOURCES.items():
        folder = source/name
        for file in metadata["files"]:
            download(f"https://raw.githubusercontent.com/{metadata['repo']}/{metadata['sha']}/{file}", folder/file)
        (folder/"revision.json").write_text(json.dumps({k: metadata[k] for k in ("repo", "sha")}))
    for name, (file, url, checksum) in WEIGHTS.items():
        download(url, source/file, checksum)
    if args.download_only:
        print(f"Upstream sources and verified weights downloaded to {source}")
        return
    from convert_models import export
    for name in WEIGHTS:
        export(name, source, args.models_dir/name)


if __name__ == "__main__":
    main()
