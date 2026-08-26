from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.input_dir.glob("figure*.png"))
    if len(files) != 18:
        raise RuntimeError(f"expected 18 QA renders, found {len(files)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 26)
    for sheet_index in range(3):
        selected = files[sheet_index * 6 : (sheet_index + 1) * 6]
        tiles = []
        for path in selected:
            with Image.open(path) as source:
                image = source.convert("RGB")
            width = 1000
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (width, height + 42), "white")
            tile.paste(image, (0, 42))
            ImageDraw.Draw(tile).text((10, 7), path.stem, fill="#34261F", font=font)
            tiles.append(tile)
        cell_height = max(tile.height for tile in tiles)
        sheet = Image.new("RGB", (2000, cell_height * 3), "#E8DED4")
        for index, tile in enumerate(tiles):
            x = (index % 2) * 1000
            y = (index // 2) * cell_height
            sheet.paste(tile, (x, y))
        sheet.save(args.output_dir / f"contact_sheet_{sheet_index + 1}.png", dpi=(170, 170))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
