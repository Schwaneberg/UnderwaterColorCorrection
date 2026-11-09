"""Build an independent diagnostic MSIX; sign separately with a test certificate."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True,
                        help="Extracted official Windows x64 embedded Python")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = root / "dist/store-path-probe"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--windowed", "--name", "StorePathProbe", "--distpath", str(output),
        "--workpath", str(output / "build"), "--specpath", str(output),
        str(root / "tools/store_path_probe.py")], check=True)
    layout = output / "StorePathProbe"
    runtime = layout / "_internal/probe-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for item in args.runtime.iterdir():
        if item.is_file():
            shutil.copy2(item, runtime / item.name)
    if not (runtime / "python.exe").is_file():
        raise RuntimeError("Runtime must contain python.exe")
    # Never ship the pip packages from the source experiment.
    for pth in runtime.glob("*._pth"):
        pth.write_text("python314.zip\n.\n#import site\n", encoding="utf-8")
    from PIL import Image, ImageDraw
    assets = layout / "Assets"
    assets.mkdir(exist_ok=True)
    for size in (44, 50, 150):
        image = Image.new("RGB", (size, size), "#18475c")
        ImageDraw.Draw(image).text((size // 8, size // 3), "TEST", fill="white")
        image.save(assets / f"Logo{size}.png")
    (layout / "AppxManifest.xml").write_text('''<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
 xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
 xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
 IgnorableNamespaces="uap rescap">
 <Identity Name="UWCC.StorePathProbe" Publisher="CN=UWCC Store Path Probe" Version="1.1.0.0" ProcessorArchitecture="x64" />
 <Properties><DisplayName>UWCC Store Path Probe (diagnostic)</DisplayName>
 <PublisherDisplayName>UWCC diagnostic</PublisherDisplayName><Logo>Assets\\Logo50.png</Logo></Properties>
 <Dependencies><TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.19041.0" MaxVersionTested="10.0.26100.0" /></Dependencies>
 <Resources><Resource Language="en-us" /></Resources>
 <Applications><Application Id="Probe" Executable="StorePathProbe.exe" EntryPoint="Windows.FullTrustApplication">
 <uap:VisualElements DisplayName="UWCC Store Path Probe (diagnostic)" Description="Isolated AppData visibility test"
 BackgroundColor="#18475c" Square150x150Logo="Assets\\Logo150.png" Square44x44Logo="Assets\\Logo44.png" />
 </Application></Applications><Capabilities><rescap:Capability Name="runFullTrust" /></Capabilities>
</Package>''', encoding="utf-8")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10/bin")
    makeappx = sorted(sdk.glob("10.*/x64/makeappx.exe"))[-1]
    subprocess.run([str(makeappx), "pack", "/d", str(layout), "/p",
                    str(output / "UWCC.StorePathProbe.msix"), "/o"], check=True)


if __name__ == "__main__":
    main()
