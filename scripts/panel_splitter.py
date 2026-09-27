"""
panel_splitter.py -- Split whole-panel EL images into individual solar cells.

The first implementation uses a fixed row/column grid. This is intentionally
simple and predictable for EL images captured with a consistent panel layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitCell:
    source: Path
    path: Path
    row: int
    col: int
    bbox: tuple[int, int, int, int]


def split_panel_image(
    image_path: Path,
    out_dir: Path,
    rows: int,
    cols: int,
    margin_pct: float = 0.0,
    gap_pct: float = 0.0,
    margins: dict[str, int] | None = None,
) -> list[SplitCell]:
    """Split one panel image into row/column cell crops."""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")
    if not 0 <= margin_pct < 40:
        raise ValueError("margin_pct must be between 0 and 40")
    if not 0 <= gap_pct < 40:
        raise ValueError("gap_pct must be between 0 and 40")

    panel_out = out_dir / image_path.stem
    panel_out.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        img = img.convert("L")
        width, height = img.size

        mx = int(width * margin_pct / 100.0)
        my = int(height * margin_pct / 100.0)
        usable_left, usable_top = mx, my
        usable_right, usable_bottom = width - mx, height - my
        if margins is not None:
            if set(margins) != {"left", "top", "right", "bottom"} or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in margins.values()):
                raise ValueError("Margins must contain nonnegative whole pixels for each side.")
            usable_left, usable_top = margins["left"], margins["top"]
            usable_right, usable_bottom = width-margins["right"], height-margins["bottom"]
        usable_w = usable_right - usable_left
        usable_h = usable_bottom - usable_top
        if min(usable_w/cols, usable_h/rows)*(1-gap_pct/100) < 1:
            raise ValueError("Margins and grid leave empty cell crops.")
        cell_w = usable_w / cols
        cell_h = usable_h / rows

        cells: list[SplitCell] = []
        for row in range(rows):
            for col in range(cols):
                left = usable_left + col * cell_w
                top = usable_top + row * cell_h
                right = usable_left + (col + 1) * cell_w
                bottom = usable_top + (row + 1) * cell_h

                gx = (right - left) * gap_pct / 200.0
                gy = (bottom - top) * gap_pct / 200.0
                bbox = (
                    int(round(left + gx)),
                    int(round(top + gy)),
                    int(round(right - gx)),
                    int(round(bottom - gy)),
                )

                crop = img.crop(bbox)
                cell_path = panel_out / f"{image_path.stem}_r{row + 1:02d}_c{col + 1:02d}.png"
                crop.save(cell_path)
                cells.append(
                    SplitCell(
                        source=image_path,
                        path=cell_path,
                        row=row + 1,
                        col=col + 1,
                        bbox=bbox,
                    )
                )

    return cells


def split_panel_folder(
    src_dir: Path,
    out_dir: Path,
    rows: int,
    cols: int,
    margin_pct: float = 0.0,
    gap_pct: float = 0.0,
) -> list[SplitCell]:
    """Split all supported images in a folder into cell crops."""
    src_dir = Path(src_dir)
    images = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    cells: list[SplitCell] = []
    for image_path in images:
        cells.extend(split_panel_image(image_path, out_dir, rows, cols, margin_pct, gap_pct))
    log.info("Split %d panel image(s) into %d cells at %s", len(images), len(cells), out_dir)
    return cells


def split_cells_to_lookup(cells: list[SplitCell]) -> dict[str, SplitCell]:
    """Return a lookup keyed by crop filename for attaching panel metadata."""
    return {cell.path.name: cell for cell in cells}
