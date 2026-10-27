"""Installed entry point for explicit, user-owned model preparation."""
import argparse
from pathlib import Path
import subprocess
import sys

from .project import model_directory


def tool_directory():
    installed = Path("/usr/share/uwcc/model-tools")
    if installed.is_dir():
        return installed
    return Path(__file__).resolve().parent.parent/"tools"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "prepare-gpu", "validate-gpu", "prepare-orientation"])
    parser.add_argument("--models-dir", type=Path, default=model_directory())
    parser.add_argument("--download-only", action="store_true", help="Download without converting (prepare only)")
    args = parser.parse_args(argv)
    if args.download_only and args.command != "prepare":
        parser.error("--download-only is only available with prepare")
    scripts = {"prepare": "prepare_models.py", "prepare-gpu": "prepare_gpu_denoise.py",
               "validate-gpu": "validate_gpu_denoise.py", "prepare-orientation": "prepare_orientation_model.py"}
    script = tool_directory()/scripts[args.command]
    if not script.is_file():
        parser.exit(1, "Install uwcc-model-tools to use model preparation.\n")
    command = [sys.executable, str(script), "--models-dir", str(args.models_dir.expanduser())]
    if args.download_only:
        command.append("--download-only")
    try:
        return subprocess.run(command).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
