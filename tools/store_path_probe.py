"""Standalone MSIX AppData visibility experiment; no UWCC imports or models."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from urllib.request import urlopen


def bootstrap_pip(runtime, env):
    """Exercise the embedded bootstrap with physical paths, as in UWCC's fix."""
    python = runtime / "python.exe"
    result = {"python": str(python), "steps": []}

    def run(label, arguments):
        child = subprocess.run([str(python), "-I", *arguments],
            cwd=str(runtime), env=env, capture_output=True, text=True,
            timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
        result["steps"].append({"label": label, "returncode": child.returncode,
                                 "stdout": child.stdout[-12000:], "stderr": child.stderr[-12000:]})
        return child.returncode

    script = runtime / "get-pip.py"
    try:
        result["initial_pip_returncode"] = run("fresh runtime pip check", ["-m", "pip", "--version"])
        if result["initial_pip_returncode"] == 0:
            raise RuntimeError("Expected a fresh embedded runtime without pip")
        for pth in runtime.glob("python*._pth"):
            lines = pth.read_text(encoding="utf-8").splitlines()
            lines = ["import site" if line.strip() == "#import site" else line for line in lines]
            if "import site" not in lines:
                lines.append("import site")
            pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["url"] = "https://bootstrap.pypa.io/get-pip.py"
        with urlopen(result["url"], timeout=120) as response, script.open("wb") as output:
            shutil.copyfileobj(response, output)
        result["downloaded_bytes"] = script.stat().st_size
        result["script"] = str(script)
        if run("install pip using physical script path", [str(script), "--isolated",
               "--disable-pip-version-check", "--no-warn-script-location"]):
            raise RuntimeError("get-pip.py failed; see captured subprocess output")
        result["success"] = run("verify installed pip", ["-m", "pip", "--version"]) == 0
    except Exception:
        result.update(success=False, error=traceback.format_exc())
    finally:
        script.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="Unpackaged control output directory")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    root = args.root or Path(os.environ["LOCALAPPDATA"]) / "UWCC.StorePathProbe"
    logical = root.absolute() / ("run-" + uuid.uuid4().hex)
    logical.mkdir(parents=True)
    physical = logical.resolve(strict=True)
    report = {"probe_version": "1.1.0.0", "logical": str(logical), "physical": str(physical),
              "redirected": logical != physical, "tests": []}
    try:
        family_size = ctypes.c_uint32(0)
        status = ctypes.windll.kernel32.GetCurrentPackageFullName(ctypes.byref(family_size), None)
        if status == 122:
            family = ctypes.create_unicode_buffer(family_size.value)
            status = ctypes.windll.kernel32.GetCurrentPackageFullName(ctypes.byref(family_size), family)
            report["package"] = family.value if status == 0 else f"error {status}"
        else:
            report["package"] = f"no package identity (status {status})"
        bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "probe-runtime"
        shutil.copytree(bundled, logical / "runtime")
        script = logical / "visibility.py"
        script.write_text("import json,sys,pathlib; print(json.dumps({"
                          "'executable':sys.executable,'script':__file__,"
                          "'marker':pathlib.Path(__file__).with_name('marker.txt').read_text()}))",
                          encoding="utf-8")
        (logical / "marker.txt").write_text("UWCC isolated path probe", encoding="utf-8")
        env = os.environ.copy()
        if getattr(sys, "frozen", False):
            # Match UWCC's isolation of external Python from PyInstaller DLLs.
            ctypes.windll.kernel32.SetDllDirectoryW(None)
            excluded = [Path(sys.executable).parent, Path(sys._MEIPASS)]
            env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                if p and not any(Path(p).resolve().is_relative_to(r.resolve()) for r in excluded))
        for label, python, script_path in (
            ("resolved physical paths", physical / "runtime/python.exe", physical / script.name),
        ):
            started = time.perf_counter()
            test = {"label": label, "python": str(python), "script": str(script_path),
                    "parent_sees_script": script_path.is_file()}
            try:
                child = subprocess.run([str(python), "-I", str(script_path)],
                    cwd=str(python.parent), env=env, capture_output=True, text=True,
                    timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
                test.update(returncode=child.returncode, stdout=child.stdout, stderr=child.stderr)
            except Exception:
                test["error"] = traceback.format_exc()
            test["seconds"] = time.perf_counter() - started
            report["tests"].append(test)
        report["pip_bootstrap"] = bootstrap_pip(physical / "runtime", env)
    except Exception:
        report["error"] = traceback.format_exc()
    output = physical / "report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.no_ui:
        ctypes.windll.user32.MessageBoxW(None,
            "Diagnostic finished. No UWCC models or settings were changed.\n\n"
            "The report will open in Notepad. Save a copy to send back.\n\n" + str(output),
            "UWCC isolated Store path probe", 0x40)
        subprocess.Popen(["notepad.exe", str(output)])
    else:
        print(output)


if __name__ == "__main__":
    main()
