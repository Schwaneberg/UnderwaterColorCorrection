"""Build a standalone Windows bundle and Microsoft Store MSIX package.

Bundles UWCC, its optional native C extension, model preparation scripts,
and FFmpeg/ffprobe binaries into a self-contained directory, prepares
Microsoft Store visual assets and AppxManifest.xml, and packs an .msix
archive when makeappx.exe (Windows SDK) is available.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape


EXCLUDED_SUFFIXES = {".onnx", ".pth", ".pt", ".safetensors", ".so", ".pyd", ".dll", ".exe", ".msix", ".ico"}
EXCLUDED_DIRS = {".sources", ".aws", ".codex", ".agents", "__pycache__", "dist", "build"}


def source_files(root: Path):
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    for name in sorted(set(names.decode("utf-8", "replace").split("\0")) - {""}):
        path = Path(name)
        if (
            path.suffix.lower() in EXCLUDED_SUFFIXES
            or any(part in EXCLUDED_DIRS or part.startswith(".venv") for part in path.parts)
        ):
            continue
        if (root / path).is_file():
            yield path


def project_version(root: Path) -> str:
    init_file = root / "uwcc/__init__.py"
    if init_file.is_file():
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_file.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not determine version from pyproject.toml")
    return match.group(1)


def msix_version(version: str) -> str:
    numbers = [ int(part) for part in re.findall(r"\d+", version)[:4] ]
    while len(numbers) < 4:
        numbers.append(0)
    return ".".join(str(n) for n in numbers[:4])


def resolve_ffmpeg(ffmpeg_dir: Path | None = None) -> tuple[Path, Path]:
    search_dir = ffmpeg_dir or (Path(os.environ["UWCC_FFMPEG_DIR"]) if os.environ.get("UWCC_FFMPEG_DIR") else None)
    paths = []
    for name in ("ffmpeg", "ffprobe"):
        if search_dir:
            expanded = Path(search_dir).expanduser()
            found = shutil.which(name, path=str(expanded))
            if not found:
                candidate = expanded / (f"{name}.exe" if os.name == "nt" else name)
                if candidate.is_file():
                    found = str(candidate)
        else:
            found = shutil.which(name)
        if not found:
            raise RuntimeError(
                f"{name} was not found. Install FFmpeg on PATH or pass --ffmpeg-dir / set UWCC_FFMPEG_DIR."
            )
        paths.append(Path(found).resolve())
    return paths[0], paths[1]


def bundle_ffmpeg(destination_bin: Path, ffmpeg_dir: Path | None = None) -> tuple[Path, Path]:
    ffmpeg_path, ffprobe_path = resolve_ffmpeg(ffmpeg_dir)
    destination_bin.mkdir(parents=True, exist_ok=True)
    bundled = []
    for source in (ffmpeg_path, ffprobe_path):
        target_name = source.name if source.suffix.lower() == ".exe" else f"{source.stem}.exe"
        target = destination_bin / target_name
        shutil.copy2(source, target)
        bundled.append(target)
    source_dir = ffmpeg_path.parent
    if ffmpeg_dir or ffmpeg_path.parent == ffprobe_path.parent:
        for extra in sorted(source_dir.iterdir()):
            if extra.is_file() and (
                extra.suffix.lower() == ".dll"
                or extra.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
            ):
                shutil.copy2(extra, destination_bin / extra.name)
    return bundled[0], bundled[1]


def prepare_windows_assets(icon_source: Path, assets_dir: Path, ico_path: Path, app_png_path: Path | None = None):
    from PIL import Image

    assets_dir.mkdir(parents=True, exist_ok=True)
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(icon_source) as raw:
        original = raw.convert("RGBA")
        ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
        original.resize((256, 256), Image.Resampling.LANCZOS).save(ico_path, format="ICO", sizes=ico_sizes)
        for name, size in (
            ("StoreLogo.png", (50, 50)),
            ("Square44x44Logo.png", (44, 44)),
            ("Square150x150Logo.png", (150, 150)),
        ):
            tile = original.resize(size, Image.Resampling.LANCZOS)
            tile.save(assets_dir / name, format="PNG", optimize=True)
        wide = Image.new("RGBA", (310, 150), (20, 30, 42, 255))
        centered = original.resize((130, 130), Image.Resampling.LANCZOS)
        wide.paste(centered, ((310 - 130) // 2, (150 - 130) // 2), centered)
        wide.save(assets_dir / "Wide310x150Logo.png", format="PNG", optimize=True)
        if app_png_path is not None:
            app_png_path.parent.mkdir(parents=True, exist_ok=True)
            original.resize((512, 512), Image.Resampling.LANCZOS).save(app_png_path, format="PNG", optimize=True)


def write_appx_manifest(
    destination: Path,
    *,
    identity_name: str = "OliverSchwaneberg.UnderwaterColorCorrection",
    publisher: str = "CN=OliverSchwaneberg",
    version: str = "0.2.0.0",
    display_name: str = "Underwater Color Correction",
    publisher_display_name: str = "Oliver Schwaneberg",
    architecture: str = "x64",
    executable: str = "uwcc.exe",
) -> Path:
    manifest_version = msix_version(version)
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:uap5="http://schemas.microsoft.com/appx/manifest/uap/windows10/5"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap uap5 rescap">

  <Identity
    Name="{escape(identity_name)}"
    Publisher="{escape(publisher)}"
    Version="{escape(manifest_version)}"
    ProcessorArchitecture="{escape(architecture)}" />

  <Properties>
    <DisplayName>{escape(display_name)}</DisplayName>
    <PublisherDisplayName>{escape(publisher_display_name)}</PublisherDisplayName>
    <Logo>Assets\\StoreLogo.png</Logo>
    <Description>Local underwater video color correction with adaptive white balance, temporal smoothing, stabilization, and optional local ONNX inference.</Description>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0" />
  </Dependencies>

  <Resources>
    <Resource Language="en-us" />
  </Resources>

  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>

  <Applications>
    <Application Id="UWCC" Executable="{escape(executable)}" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="{escape(display_name)}"
        Description="Local underwater video color correction"
        BackgroundColor="#141e2a"
        Square150x150Logo="Assets\\Square150x150Logo.png"
        Square44x44Logo="Assets\\Square44x44Logo.png">
        <uap:DefaultTile Wide310x150Logo="Assets\\Wide310x150Logo.png" />
      </uap:VisualElements>
      <Extensions>
        <uap5:Extension Category="windows.appExecutionAlias">
          <uap5:AppExecutionAlias>
            <uap5:ExecutionAlias Alias="uwcc.exe" />
          </uap5:AppExecutionAlias>
        </uap5:Extension>
        <uap:Extension Category="windows.fileTypeAssociation">
          <uap:FileTypeAssociation Name="uwcc-media">
            <uap:DisplayName>Underwater Video / UWCC Project</uap:DisplayName>
            <uap:SupportedFileTypes>
              <uap:FileType>.mp4</uap:FileType>
              <uap:FileType>.mov</uap:FileType>
              <uap:FileType>.mkv</uap:FileType>
              <uap:FileType>.mts</uap:FileType>
              <uap:FileType>.m2ts</uap:FileType>
              <uap:FileType>.avi</uap:FileType>
              <uap:FileType>.json</uap:FileType>
            </uap:SupportedFileTypes>
          </uap:FileTypeAssociation>
        </uap:Extension>
      </Extensions>
    </Application>
  </Applications>
</Package>
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(xml, encoding="utf-8")
    return destination


def find_makeappx(explicit: Path | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"makeappx executable not found: {path}")
        return path
    found = shutil.which("makeappx") or shutil.which("makeappx.exe")
    if found:
        return Path(found).resolve()
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    sdk_bin = Path(program_files) / "Windows Kits/10/bin"
    if sdk_bin.is_dir():
        candidates = sorted(sdk_bin.glob("10.*/x64/makeappx.exe"))
        if candidates:
            return candidates[-1].resolve()
    return None


def verify_runtime_dependencies(*, allow_missing_vision: bool = False, allow_missing_ml: bool = False) -> list[str]:
    packages = []
    if importlib.util.find_spec("cv2"):
        packages.append("cv2")
    elif not allow_missing_vision:
        raise RuntimeError(
            "OpenCV (cv2) is not installed. Install the 'vision' extra ('pip install .[vision]') "
            "before building the Windows package, or pass --allow-missing-vision."
        )
    if importlib.util.find_spec("onnxruntime"):
        packages.append("onnxruntime")
    elif not allow_missing_ml:
        raise RuntimeError(
            "ONNX Runtime (onnxruntime) is not installed. Install an ML extra ('pip install .[ml-amd]') "
            "before building the Windows package, or pass --allow-missing-ml."
        )
    return packages


def copy_support_files(source_root: Path, bundle_dir: Path):
    tools_dest = bundle_dir / "tools"
    tools_dest.mkdir(parents=True, exist_ok=True)
    for script in sorted((source_root / "tools").glob("*.py")):
        shutil.copy2(script, tools_dest / script.name)
    for item in ("LICENSE", "README.md"):
        src = source_root / item
        if src.is_file():
            shutil.copy2(src, bundle_dir / item)


def build_bundle(
    source_root: Path,
    output_bundle_dir: Path,
    ico_path: Path,
    ffmpeg_dir: Path | None = None,
    *,
    require_native: bool = False,
    collect_packages: list[str] | None = None,
) -> Path:
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=source_root,
        check=require_native,
    )
    if require_native and not list((source_root / "uwcc").glob("_native*.pyd")) and not list((source_root / "uwcc").glob("_native*.so")):
        raise RuntimeError("Native C extension uwcc._native was not produced by build_ext.")
    sep = ";" if os.name == "nt" else ":"
    pyinstaller_dist = output_bundle_dir.parent / "_pyinstaller_dist"
    pyinstaller_work = output_bundle_dir.parent / "_pyinstaller_build"
    spec_dir = output_bundle_dir.parent / "_pyinstaller_spec"
    for folder in (pyinstaller_dist, pyinstaller_work, spec_dir):
        shutil.rmtree(folder, ignore_errors=True)
    entry_script = source_root / "_uwcc_entry.py"
    entry_script.write_text(
        "import sys\nfrom uwcc.__main__ import main\n\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--hide-console",
        "hide-early",
        "--name",
        "uwcc",
        "--icon",
        str(ico_path),
        "--paths",
        str(source_root),
        "--distpath",
        str(pyinstaller_dist),
        "--workpath",
        str(pyinstaller_work),
        "--specpath",
        str(spec_dir),
        "--add-data",
        f"{source_root / 'uwcc/assets'}{sep}uwcc/assets",
        "--collect-all",
        "uwcc",
    ]
    for package in (collect_packages if collect_packages is not None else ["cv2", "onnxruntime"]):
        if importlib.util.find_spec(package):
            cmd.extend(["--collect-all", package])
    cmd.append(str(entry_script))
    subprocess.run(cmd, cwd=source_root, check=True)
    built = pyinstaller_dist / "uwcc"
    if output_bundle_dir.exists():
        shutil.rmtree(output_bundle_dir)
    shutil.move(str(built), str(output_bundle_dir))
    for folder in (pyinstaller_dist, pyinstaller_work, spec_dir):
        shutil.rmtree(folder, ignore_errors=True)
    copy_support_files(source_root, output_bundle_dir)
    bundle_ffmpeg(output_bundle_dir / "bin", ffmpeg_dir=ffmpeg_dir)
    return output_bundle_dir


def stage_msix_layout(
    bundle_dir: Path,
    layout_dir: Path,
    icon_source: Path,
    *,
    identity_name: str,
    publisher: str,
    version: str,
    display_name: str,
    publisher_display_name: str,
    architecture: str,
) -> Path:
    if layout_dir.exists():
        shutil.rmtree(layout_dir)
    shutil.copytree(bundle_dir, layout_dir)
    ico_tmp = layout_dir / "Assets/uwcc.ico"
    prepare_windows_assets(icon_source, layout_dir / "Assets", ico_tmp)
    write_appx_manifest(
        layout_dir / "AppxManifest.xml",
        identity_name=identity_name,
        publisher=publisher,
        version=version,
        display_name=display_name,
        publisher_display_name=publisher_display_name,
        architecture=architecture,
    )
    return layout_dir


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "dist/windows")
    parser.add_argument("--ffmpeg-dir", type=Path, help="Directory containing ffmpeg.exe and ffprobe.exe")
    parser.add_argument("--identity-name", default="OliverSchwaneberg.UnderwaterColorCorrection")
    parser.add_argument("--publisher", default="CN=OliverSchwaneberg",
                        help="Publisher subject from Microsoft Partner Center (e.g. CN=XXXX-XXXX...)")
    parser.add_argument("--publisher-display-name", default="Oliver Schwaneberg")
    parser.add_argument("--display-name", default="Underwater Color Correction")
    parser.add_argument("--architecture", choices=["x64", "arm64"], default="x64")
    parser.add_argument("--makeappx", type=Path, help="Explicit path to Windows SDK makeappx.exe")
    parser.add_argument("--layout-only", action="store_true",
                        help="Prepare the standalone bundle and MSIX layout without invoking makeappx.exe")
    parser.add_argument("--require-native", action="store_true",
                        help="Fail the build if the optional uwcc._native C extension does not compile")
    parser.add_argument("--allow-missing-vision", action="store_true",
                        help="Allow building without OpenCV (cv2) installed")
    parser.add_argument("--allow-missing-ml", action="store_true",
                        help="Allow building without ONNX Runtime installed")
    args = parser.parse_args(argv)

    version = project_version(root)
    resolve_ffmpeg(args.ffmpeg_dir)
    collect_packages = verify_runtime_dependencies(
        allow_missing_vision=args.allow_missing_vision,
        allow_missing_ml=args.allow_missing_ml,
    )
    with tempfile.TemporaryDirectory(prefix="uwcc-win-") as temporary:
        work = Path(temporary)
        source = work / f"uwcc-{version}"
        source.mkdir()
        for path in source_files(root):
            destination = source / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / path, destination)
        icon_source = source / "UWCC Stingray Reef Icon.png"
        ico_path = work / "uwcc.ico"
        prepare_windows_assets(
            icon_source,
            work / "Assets",
            ico_path,
            app_png_path=source / "uwcc/assets/uwcc.png",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir = args.output_dir / "uwcc"
        build_bundle(
            source,
            bundle_dir,
            ico_path,
            ffmpeg_dir=args.ffmpeg_dir,
            require_native=args.require_native,
            collect_packages=collect_packages,
        )
        print(f"Standalone Windows bundle: {bundle_dir}")

        layout_dir = args.output_dir / "msix-layout"
        stage_msix_layout(
            bundle_dir,
            layout_dir,
            icon_source,
            identity_name=args.identity_name,
            publisher=args.publisher,
            version=version,
            display_name=args.display_name,
            publisher_display_name=args.publisher_display_name,
            architecture=args.architecture,
        )
        print(f"MSIX layout staged: {layout_dir}")

        if args.layout_only:
            return 0

        makeappx = find_makeappx(args.makeappx)
        msix_path = args.output_dir / f"uwcc_{version}_{args.architecture}.msix"
        if makeappx:
            subprocess.run(
                [str(makeappx), "pack", "/o", "/d", str(layout_dir), "/p", str(msix_path)],
                check=True,
            )
            print(f"MSIX package built: {msix_path}")
        else:
            print(
                "makeappx.exe not found (install Windows 10/11 SDK to pack .msix automatically).\n"
                f"To pack manually: makeappx pack /o /d \"{layout_dir}\" /p \"{msix_path}\""
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
