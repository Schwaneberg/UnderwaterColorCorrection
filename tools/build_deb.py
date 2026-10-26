"""Build unsigned Ubuntu packages from a clean snapshot of the working tree.

Requires Debian build tools; see docs/ubuntu-packaging.md. No network or sudo
is used by this command. Ignored local files and model binaries are excluded.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def source_files(root):
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    for name in sorted(set(names.decode().split("\0")) - {""}):
        path = Path(name)
        if (path.suffix in {".onnx", ".pth", ".pt", ".safetensors", ".so", ".pyd"}
                or any(part in {".sources", ".aws", ".codex", ".agents", "__pycache__"}
                       or part.startswith(".venv") for part in path.parts)):
            continue
        if (root/path).is_file():
            yield path


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true", help="Build a source package for signing and PPA upload")
    parser.add_argument("--output-dir", type=Path, default=root/"dist/debian")
    args = parser.parse_args()
    version = subprocess.check_output(
        ["dpkg-parsechangelog", "--show-field", "Version"], cwd=root, text=True).strip()
    upstream = version.split(":")[-1].rsplit("-", 1)[0]
    with tempfile.TemporaryDirectory(prefix="uwcc-deb-") as temporary:
        work = Path(temporary)
        source = work/f"uwcc-{upstream}"
        source.mkdir()
        for path in source_files(root):
            destination = source/path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/path, destination)
        # Binary builds do not replace upstream tarballs from existing source builds.
        if args.source:
            with tarfile.open(work/f"uwcc_{upstream}.orig.tar.xz", "w:xz") as archive:
                for path in sorted(source.iterdir()):
                    if path.name != "debian":
                        archive.add(path, arcname=f"uwcc-{upstream}/{path.name}")
        command = ["dpkg-buildpackage", "--no-sign", "--build=source" if args.source else "--build=binary"]
        if args.source:
            command.append("-sa")
        subprocess.run(command, cwd=source, check=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for artifact in sorted(work.iterdir()):
            if artifact.is_file():
                destination = args.output_dir/artifact.name
                shutil.copy2(artifact, destination)
                print(destination)


if __name__ == "__main__":
    main()
