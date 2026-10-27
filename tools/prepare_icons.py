"""Prepare application and desktop icon sizes from the supplied original PNG."""

from pathlib import Path

from PIL import Image


def main():
    root = Path(__file__).resolve().parent.parent
    with Image.open(root / "UWCC Stingray Reef Icon.png") as original:
        for size in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
            destination = root / "data/icons/hicolor" / f"{size}x{size}/apps/uwcc.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            icon = original.resize((size, size), Image.Resampling.LANCZOS)
            icon.save(destination, optimize=True)
            if size == 512:
                asset = root / "uwcc/assets/uwcc.png"
                asset.parent.mkdir(parents=True, exist_ok=True)
                icon.save(asset, optimize=True)


if __name__ == "__main__":
    main()
