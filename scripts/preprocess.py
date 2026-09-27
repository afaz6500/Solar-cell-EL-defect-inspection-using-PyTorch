"""
preprocess.py — Solar Panel EL Image Preprocessing Pipeline

Steps:
  1. Load raw grayscale EL images from dataset/raw_img_gray/
  2. Apply histogram equalization + denoising → dataset/transformed_img/
  3. Auto-split into train/val/test by class label (inferred from filename or subfolder)
  4. Generate dataset statistics report
"""

import shutil
import random
import logging
import re
import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from scripts.config import (
    CLASSES, EXCLUDED_CLASSES, CONFLICTS_DIR, IMG_SIZE as CONFIG_IMG_SIZE, LOCKED_TEST_DIR,
    RAW_DIR, SEED, TEST_DIR, TEST_SPLIT, TRAIN_DIR, TRANSFORMED_DIR, VAL_DIR,
    VAL_SPLIT,
)

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = (CONFIG_IMG_SIZE, CONFIG_IMG_SIZE)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Pipeline applied to every raw EL image:
      • Resize to IMG_SIZE
      • CLAHE (contrast-limited adaptive histogram equalization)
      • Gaussian denoising
      • Normalize to [0, 255] uint8
    """
    img = cv2.resize(img, IMG_SIZE)

    # CLAHE on grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:                                   # convert to gray if accidentally RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img)

    # Gaussian blur for denoising
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def infer_class(path: Path) -> str | None:
    """
    Guess class label from the file path.
    Priority: parent folder name → filename keyword.
    Returns None if no match found.
    """
    parent = path.parent.name.lower()
    stem = path.stem.lower()
    text = f"{parent}_{stem}"
    if parent in EXCLUDED_CLASSES:
        return None

    # 1. Exact match on immediate parent folder name (most reliable)
    if parent in CLASSES:
        return parent

    # 2. Keyword matching, longest/most-specific class names first
    #    so "star_crack" is checked before "crack"
    for cls in sorted(CLASSES, key=len, reverse=True):
        if cls in text:
            return cls

    # 3. Common aliases / variations
    aliases = {
        "black_core": ["blackcore", "black-core"],
        "horizontal_dislocation": ["horizontal", "hor_dislocation", "h_dislocation"],
        "short_circuit": ["shortcircuit", "short-circuit", "short_ckt"],
        "star_crack": ["starcrack", "star-crack"],
        "thick_line": ["thickline", "thick-line"],
        "normal": ["good", "ok", "clean", "no_defect", "nodefect", "defect_free"],
    }
    for cls, keys in aliases.items():
        if any(k in text for k in keys):
            return cls

    return None


def gather_images() -> dict[str, list[Path]]:
    """Walk RAW_DIR and build {class_name: [image_paths]} mapping."""
    buckets: dict[str, list[Path]] = {c: [] for c in CLASSES}
    unknown = []
    by_hash: dict[str, list[tuple[str, Path]]] = {}
    locked_names = {
        p.name
        for p in LOCKED_TEST_DIR.rglob("*")
        if LOCKED_TEST_DIR.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    }

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(RAW_DIR.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        if p.name in locked_names:
            continue
        label = infer_class(p)
        if label:
            image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if image is None:
                unknown.append(p)
                continue
            normalized = enhance_image(image)
            digest = hashlib.sha256(normalized.tobytes()).hexdigest()
            by_hash.setdefault(digest, []).append((label, p))
        else:
            unknown.append(p)

    CONFLICTS_DIR.mkdir(parents=True, exist_ok=True)
    conflict_count = 0
    duplicate_count = 0
    for digest, matches in by_hash.items():
        labels = {label for label, _ in matches}
        if len(labels) > 1:
            conflict_count += len(matches)
            for label, path in matches:
                destination = CONFLICTS_DIR / label / f"{digest[:12]}_{path.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            log.warning(
                "Conflicting labels for identical image %s: %s",
                digest[:12],
                sorted(labels),
            )
            continue

        label = matches[0][0]
        representative = sorted((path for _, path in matches), key=str)[0]
        buckets[label].append(representative)
        duplicate_count += len(matches) - 1

    if unknown:
        log.warning("%d images could not be classified and will be skipped:", len(unknown))
        for u in unknown[:10]:
            log.warning("  %s", u)
    if duplicate_count:
        log.info("Removed %d exact duplicate raw images.", duplicate_count)
    if conflict_count:
        log.warning(
            "Excluded %d images with conflicting labels; copies are in %s.",
            conflict_count,
            CONFLICTS_DIR,
        )

    for cls, imgs in buckets.items():
        log.info("Class '%-15s': %d images", cls, len(imgs))

    return buckets


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing():
    random.seed(SEED)

    if TRANSFORMED_DIR.exists():
        shutil.rmtree(TRANSFORMED_DIR)
    if CONFLICTS_DIR.exists():
        shutil.rmtree(CONFLICTS_DIR)
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    for split in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        for cls in CLASSES:
            class_dir = split / cls
            if class_dir.exists():
                shutil.rmtree(class_dir)
            class_dir.mkdir(parents=True, exist_ok=True)

    buckets = gather_images()

    total_train = total_val = total_test = 0

    for cls, paths in buckets.items():
        if not paths:
            log.warning("No images found for class '%s' — skipping.", cls)
            continue

        def get_group_id(path: Path) -> str:
            """
            Keep cells from the same whole-panel source together when the source
            panel is obvious, but do not collapse generic names like cell_0001
            into one giant group.
            """
            filename = path.stem
            panel_crop = re.search(r"(.+?)_r\d+_c\d+", filename, flags=re.IGNORECASE)
            if panel_crop:
                return panel_crop.group(1)
            panel_crop = re.search(r"(.+?\.(?:jpg|jpeg|png|bmp|tif|tiff))_r\d+_c\d+", filename, flags=re.IGNORECASE)
            if panel_crop:
                return panel_crop.group(1)
            return filename

        # Create groupings
        grouped: dict[str, list[Path]] = {}
        for p in paths:
            group_id = get_group_id(p)
            grouped.setdefault(group_id, []).append(p)

        # Shuffle groups
        group_ids = list(grouped.keys())
        random.shuffle(group_ids)

        # Calculate counts
        total_groups = len(group_ids)
        # Keep at least one group for training. Very small classes cannot
        # support all three splits without making the training split empty.
        test_count = int(total_groups * TEST_SPLIT)
        val_count = int(total_groups * VAL_SPLIT)
        if total_groups >= 3:
            test_count = max(1, test_count)
            val_count = max(1, val_count)
        while test_count + val_count >= total_groups:
            if test_count >= val_count and test_count > 0:
                test_count -= 1
            elif val_count > 0:
                val_count -= 1
        test_groups = group_ids[:test_count]
        val_groups = group_ids[test_count:test_count + val_count]
        train_groups = group_ids[test_count + val_count:]

        # Flatten back to image paths
        train_paths = [p for gid in train_groups for p in grouped[gid]]
        val_paths = [p for gid in val_groups for p in grouped[gid]]
        test_paths = [p for gid in test_groups for p in grouped[gid]]

        log.info(
            "Processing class '%s': %d train | %d val | %d test",
            cls,
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )

        for src_paths, dest_root in [(train_paths, TRAIN_DIR), (val_paths, VAL_DIR), (test_paths, TEST_DIR)]:
            dest_dir = dest_root / cls
            for src in tqdm(src_paths, desc=f"{cls} → {dest_root.name}", leave=False):
                img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.warning("Cannot read %s — skipping.", src)
                    continue

                enhanced = enhance_image(img)

                # Save transformed copy
                trans_path = TRANSFORMED_DIR / f"{cls}_{src.stem}_transformed.png"
                cv2.imwrite(str(trans_path), enhanced)

                # Save to split folder
                dest_path = dest_dir / f"{src.stem}.png"
                cv2.imwrite(str(dest_path), enhanced)

        total_train += len(train_paths)
        total_val   += len(val_paths)
        total_test  += len(test_paths)

    log.info("─" * 50)
    log.info("Preprocessing complete.")
    log.info("  Train images : %d", total_train)
    log.info("  Val   images : %d", total_val)
    log.info("  Test  images : %d", total_test)
    log.info("  Transformed  : %s", TRANSFORMED_DIR)
    log.info("  Train dir    : %s", TRAIN_DIR)
    log.info("  Val   dir    : %s", VAL_DIR)
    log.info("  Test  dir    : %s", TEST_DIR)


if __name__ == "__main__":
    run_preprocessing()
