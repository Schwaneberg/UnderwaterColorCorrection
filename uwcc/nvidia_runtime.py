"""Explicit private CUDA runtime setup; no changes to system Python or drivers."""
import argparse
import base64
import os
from pathlib import Path
import subprocess
import sys
import venv

from .project import user_directory

ORT_VERSION = "1.31.0"
REQUIREMENT = f"onnxruntime-gpu[cuda,cudnn]=={ORT_VERSION}"
# ONNX opset 17, IR 9: float32[1] y = Add(x, x). No converter dependency.
CUDA_PROBE = "CAk6QwoOCgF4CgF4EgF5IgNBZGQSD3V3Y2MtY3VkYS1jaGVja1oPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEBE="


def environment_path():
    override = os.environ.get("UWCC_NVIDIA_ENV")
    if override:
        return Path(override).expanduser()
    abi = f"{sys.version_info.major}.{sys.version_info.minor}"
    return user_directory(create=False)/"runtimes"/f"nvidia-py{abi}-ort{ORT_VERSION}"


def check_cuda():
    """Require the pinned GPU distribution and a working CUDA inference session."""
    from importlib.metadata import version
    import numpy as np
    import onnxruntime as ort
    ort.disable_telemetry_events()
    if version("onnxruntime-gpu") != ORT_VERSION:
        raise RuntimeError(f"Expected ONNX Runtime GPU {ORT_VERSION}")
    ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(base64.b64decode(CUDA_PROBE), options,
                                   providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA provider did not initialize")
    output = session.run(None, {"x": np.array([2.], dtype=np.float32)})[0]
    if not np.array_equal(output, np.array([4.], dtype=np.float32)):
        raise RuntimeError("CUDA inference validation failed")
    # Validate the installed application and Qt imports in this interpreter too.
    from . import _native, qt
    print(f"CUDA inference validated with ONNX Runtime {ORT_VERSION}.")


def validate_environment(python):
    return subprocess.run([str(python), "-m", "uwcc.nvidia_runtime", "check"],
                          capture_output=True, text=True)


def setup_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Accept the runtime download without an interactive prompt")
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        parser.exit(1, "Run this command as your desktop user, without sudo.\n")
    destination = environment_path()
    python = destination/"bin/python"
    if python.is_file() and validate_environment(python).returncode == 0:
        print(f"Validated NVIDIA runtime already available: {destination}")
        return 0
    print(f"Install {REQUIREMENT} and CUDA/cuDNN libraries from PyPI into:\n{destination}\n"
          "This can download several GB. A compatible NVIDIA GPU and driver are required.")
    if not args.yes:
        try:
            answer = input("Download and set up the NVIDIA runtime? [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return 130
        if answer.strip().lower() not in {"y", "yes"}:
            return 0
    try:
        # Reuse the apt-managed application/Qt/NumPy; GPU wheels live only here.
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(destination)
        subprocess.run([str(python), "-m", "pip", "--isolated", "install", "--only-binary=:all:",
                        "--index-url", "https://pypi.org/simple", REQUIREMENT], check=True)
        result = validate_environment(python)
        if result.returncode:
            print(result.stderr or result.stdout, file=sys.stderr)
            print("NVIDIA runtime validation failed. Check the driver and retry uwcc-nvidia-setup.", file=sys.stderr)
            return 1
        print(result.stdout, end="")
        print("Ready. Launch with uwcc-nvidia; models remain a separate setup step.")
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"NVIDIA runtime setup failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def launch_main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        print("Usage: uwcc-nvidia [UWCC arguments]\n"
              "Launch UWCC in its validated private NVIDIA runtime.\n"
              "Run uwcc-nvidia-setup first. Use uwcc --help for application commands.")
        return 0
    python = environment_path()/"bin/python"
    if not python.is_file():
        print("Run uwcc-nvidia-setup as your desktop user first.", file=sys.stderr)
        return 1
    result = validate_environment(python)
    if result.returncode:
        print(result.stderr or result.stdout, file=sys.stderr)
        print("CUDA validation failed. Run uwcc-nvidia-setup or use uwcc for CPU execution.", file=sys.stderr)
        return 1
    # Model validation must run under the same GPU interpreter as inference.
    module = "uwcc.model_setup" if args and args[0] == "model-tools" else "uwcc"
    if module == "uwcc.model_setup":
        args = args[1:]
    os.execv(str(python), [str(python), "-m", module, *args])


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check_cuda()
    else:
        raise SystemExit(setup_main())
