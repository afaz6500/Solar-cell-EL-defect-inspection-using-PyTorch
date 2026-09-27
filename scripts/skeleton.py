"""
skeleton.py — Morphological Skeleton Extraction for EL Solar Panel Images

Extracts the crack/defect skeleton from transformed EL images using:
  • Otsu thresholding
  • Morphological skeletonization (Zhang-Suen thinning via OpenCV)
  • Optional overlay generation for visual inspection

Output goes to dataset/skeleton_img/
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from scripts.config import SKELETON_DIR, TRANSFORMED_DIR

# ─── Config ────────────────────────────────────────────────────────────────────
OVERLAY_ALPHA   = 0.5          # blend weight for the overlay image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Core ──────────────────────────────────────────────────────────────────────

def skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen morphological thinning via iterative erosion.
    Input  : binary uint8 image (255 = foreground)
    Output : skeleton uint8 image
    """
    skel   = np.zeros_like(binary)
    elem   = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img    = binary.copy()

    while True:
        eroded   = cv2.erode(img, elem)
        temp     = cv2.dilate(eroded, elem)
        temp     = cv2.subtract(img, temp)
        skel     = cv2.bitwise_or(skel, temp)
        img      = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


def process_image(src: Path, dest_dir: Path, overlay_dir: Path) -> dict:
    """
    Process one image → skeleton + overlay.
    Returns stats dict.
    """
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning("Cannot read %s", src)
        return {}

    # Otsu threshold to isolate defect regions
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological clean-up: remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Skeletonize
    skeleton = skeletonize(binary)

    # Save skeleton
    skel_path = dest_dir / (src.stem + "_skel.png")
    cv2.imwrite(str(skel_path), skeleton)

    # Overlay: skeleton (red) on original gray
    gray_rgb          = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay           = gray_rgb.copy()
    overlay[skeleton > 0] = [0, 0, 255]          # BGR red
    blended = cv2.addWeighted(gray_rgb, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)

    ov_path = overlay_dir / (src.stem + "_overlay.png")
    cv2.imwrite(str(ov_path), blended)

    # Stats
    total_px    = img.shape[0] * img.shape[1]
    defect_px   = int(np.sum(binary > 0))
    skel_px     = int(np.sum(skeleton > 0))
    defect_pct  = defect_px / total_px * 100

    return {
        "file"       : src.name,
        "defect_px"  : defect_px,
        "skel_px"    : skel_px,
        "defect_pct" : round(defect_pct, 2),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def run_skeleton_extraction(src_dir: Path | None = None):
    src      = src_dir or TRANSFORMED_DIR
    dest     = SKELETON_DIR
    overlay  = SKELETON_DIR / "overlays"

    dest.mkdir(parents=True, exist_ok=True)
    overlay.mkdir(parents=True, exist_ok=True)

    exts   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in exts]

    if not images:
        log.warning("No images found in %s", src)
        return []

    log.info("Extracting skeletons from %d images …", len(images))
    stats_list = []

    for img_path in tqdm(images, desc="Skeletonizing"):
        stats = process_image(img_path, dest, overlay)
        if stats:
            stats_list.append(stats)

    # Summary
    if stats_list:
        avg_defect = np.mean([s["defect_pct"] for s in stats_list])
        log.info("─" * 50)
        log.info("Skeleton extraction complete.")
        log.info("  Images processed   : %d", len(stats_list))
        log.info("  Avg defect coverage: %.2f %%", avg_defect)
        log.info("  Skeletons saved to : %s", dest)
        log.info("  Overlays saved to  : %s", overlay)

    return stats_list


if __name__ == "__main__":
    run_skeleton_extraction()
