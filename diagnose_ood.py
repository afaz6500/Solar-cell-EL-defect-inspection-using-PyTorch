# -*- coding: utf-8 -*-
"""
diagnose_ood.py -- Find exactly which OOD check is rejecting your real EL images

Run:
    python diagnose_ood.py

What it does:
  1. Measures every image in dataset/train/
  2. Runs each OOD check individually
  3. Reports exactly which check failed and for which images
  4. Suggests the exact threshold fix needed
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from scripts.config import TRAIN_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUT_CSV   = Path("results/ood_diagnosis.csv")


def measure(img_path: Path) -> dict | None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w     = gray.shape
    total_px = h * w
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blurred, 30, 90)
    return {
        "file"            : img_path.name,
        "class"           : img_path.parent.name,
        "mean_brightness" : round(float(np.mean(gray)), 2),
        "color_std"       : round(float(np.std(gray)), 2),
        "dark_ratio"      : round(float(np.sum(gray < 50) / total_px), 4),
        "edge_density"    : round(float(np.sum(edges > 0) / total_px), 4),
        "laplacian_var"   : round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "saturation"      : round(float(np.mean(hsv[:, :, 1])), 2),
        "aspect_ratio"    : round(max(w, h) / max(min(w, h), 1), 3),
    }


def check_all(r: dict) -> list[str]:
    """Return list of which checks this image fails under current thresholds."""
    # Import live thresholds from ood_detector
    from scripts.ood_detector import (
        MIN_COLOR_STD, MAX_COLOR_STD, MAX_SATURATION_MEAN,
        EL_MIN_MEAN_BRIGHTNESS, EL_MAX_MEAN_BRIGHTNESS,
        EL_MIN_DARK_RATIO, EL_MAX_DARK_RATIO,
        EL_MAX_EDGE_DENSITY, EL_MAX_LAPLACIAN_VAR,
        EL_MIN_ASPECT_RATIO, EL_MAX_ASPECT_RATIO,
    )
    fails = []
    if r["color_std"]       < MIN_COLOR_STD:
        fails.append(f"color_std TOO LOW ({r['color_std']} < {MIN_COLOR_STD})")
    if r["color_std"]       > MAX_COLOR_STD:
        fails.append(f"color_std TOO HIGH ({r['color_std']} > {MAX_COLOR_STD})")
    if r["saturation"]      > MAX_SATURATION_MEAN:
        fails.append(f"saturation TOO HIGH ({r['saturation']} > {MAX_SATURATION_MEAN})")
    if r["mean_brightness"] < EL_MIN_MEAN_BRIGHTNESS:
        fails.append(f"brightness TOO LOW ({r['mean_brightness']} < {EL_MIN_MEAN_BRIGHTNESS})")
    if r["mean_brightness"] > EL_MAX_MEAN_BRIGHTNESS:
        fails.append(f"brightness TOO HIGH ({r['mean_brightness']} > {EL_MAX_MEAN_BRIGHTNESS})")
    if r["dark_ratio"]      > EL_MAX_DARK_RATIO:
        fails.append(f"dark_ratio TOO HIGH ({r['dark_ratio']} > {EL_MAX_DARK_RATIO})")
    if r["edge_density"]    > EL_MAX_EDGE_DENSITY:
        fails.append(f"edge_density TOO HIGH ({r['edge_density']} > {EL_MAX_EDGE_DENSITY})")
    if r["laplacian_var"]   > EL_MAX_LAPLACIAN_VAR:
        fails.append(f"laplacian_var TOO HIGH ({r['laplacian_var']} > {EL_MAX_LAPLACIAN_VAR})")
    if r["aspect_ratio"]    < EL_MIN_ASPECT_RATIO:
        fails.append(f"aspect_ratio TOO LOW ({r['aspect_ratio']} < {EL_MIN_ASPECT_RATIO})")
    if r["aspect_ratio"]    > EL_MAX_ASPECT_RATIO:
        fails.append(f"aspect_ratio TOO HIGH ({r['aspect_ratio']} > {EL_MAX_ASPECT_RATIO})")
    return fails


def diagnose():
    exts   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in TRAIN_DIR.rglob("*") if p.suffix.lower() in exts]

    if not images:
        log.error("No images in %s -- run preprocessing first.", TRAIN_DIR)
        return

    log.info("Diagnosing %d images ...", len(images))

    rows        = []
    fail_counts = {}   # check_name -> count
    rejected    = []

    for p in tqdm(images, desc="Diagnosing"):
        r = measure(p)
        if not r:
            continue
        fails = check_all(r)
        r["fails"] = " | ".join(fails) if fails else "OK"
        r["rejected"] = "YES" if fails else "NO"
        rows.append(r)
        if fails:
            rejected.append(r)
            for f in fails:
                key = f.split(" ")[0] + " " + f.split(" ")[1]  # e.g. "edge_density TOO"
                fail_counts[key] = fail_counts.get(key, 0) + 1

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print report
    total = len(rows)
    n_rej = len(rejected)
    print("\n" + "=" * 65)
    print(f"  DIAGNOSIS REPORT  ({n_rej}/{total} EL images rejected = {n_rej/total*100:.1f}%)")
    print("=" * 65)

    if not rejected:
        print("  All EL images pass -- no changes needed.")
        return

    # Which checks cause the most rejections
    print("\n  Checks causing rejections (sorted by frequency):")
    print(f"  {'Check':<40} {'Count':>6} {'% of rejected':>14}")
    print("  " + "-" * 62)
    for check, cnt in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {check:<40} {cnt:>6} {cnt/n_rej*100:>13.1f}%")

    # Suggested fixes
    print("\n  SUGGESTED THRESHOLD FIXES:")
    print("  " + "-" * 62)
    _suggest_fixes(rows, rejected)

    # Show worst offenders
    print(f"\n  Sample rejected EL images (worst {min(10, n_rej)}):")
    for r in rejected[:10]:
        print(f"  [{r['class']}] {r['file']}")
        print(f"    -> {r['fails']}")

    print(f"\n  Full list saved to: {OUT_CSV}")
    print("=" * 65)


def _suggest_fixes(all_rows: list[dict], rejected: list[dict]):
    """For each failing check, suggest the threshold that would accept all EL images."""
    from scripts.ood_detector import (
        MIN_COLOR_STD, MAX_COLOR_STD,
        EL_MIN_MEAN_BRIGHTNESS, EL_MAX_MEAN_BRIGHTNESS,
        EL_MIN_DARK_RATIO, EL_MAX_DARK_RATIO,
        EL_MAX_EDGE_DENSITY, EL_MAX_LAPLACIAN_VAR,
        EL_MIN_ASPECT_RATIO, EL_MAX_ASPECT_RATIO,
    )

    all_vals = {m: [r[m] for r in all_rows] for m in
                ["color_std","mean_brightness","dark_ratio",
                 "edge_density","laplacian_var","aspect_ratio"]}

    suggestions = {}

    for r in rejected:
        fails = check_all(r)
        for f in fails:
            if "color_std TOO LOW"       in f:
                v = round(min(all_vals["color_std"]) * 0.90, 2)
                suggestions["MIN_COLOR_STD"] = v
            if "color_std TOO HIGH"      in f:
                v = round(max(all_vals["color_std"]) * 1.10, 2)
                suggestions["MAX_COLOR_STD"] = v
            if "brightness TOO LOW"      in f:
                v = round(min(all_vals["mean_brightness"]) * 0.90, 2)
                suggestions["EL_MIN_MEAN_BRIGHTNESS"] = v
            if "brightness TOO HIGH"     in f:
                v = round(max(all_vals["mean_brightness"]) * 1.10, 2)
                suggestions["EL_MAX_MEAN_BRIGHTNESS"] = v
            if "dark_ratio TOO HIGH"     in f:
                v = round(max(all_vals["dark_ratio"]) * 1.10, 4)
                suggestions["EL_MAX_DARK_RATIO"] = v
            if "edge_density TOO HIGH"   in f:
                v = round(max(all_vals["edge_density"]) * 1.10, 4)
                suggestions["EL_MAX_EDGE_DENSITY"] = v
            if "laplacian_var TOO HIGH"  in f:
                v = round(max(all_vals["laplacian_var"]) * 1.10, 2)
                suggestions["EL_MAX_LAPLACIAN_VAR"] = v
            if "aspect_ratio TOO LOW"    in f:
                v = round(min(all_vals["aspect_ratio"]) * 0.90, 3)
                suggestions["EL_MIN_ASPECT_RATIO"] = v
            if "aspect_ratio TOO HIGH"   in f:
                v = round(max(all_vals["aspect_ratio"]) * 1.10, 3)
                suggestions["EL_MAX_ASPECT_RATIO"] = v

    if not suggestions:
        print("  No suggestions needed.")
        return

    for var, val in suggestions.items():
        print(f"  Change  {var:<28} = {val}")

    # Ask to auto-apply
    print()
    answer = input("  Auto-apply these fixes to ood_detector.py? [y/N]: ").strip().lower()
    if answer == "y":
        _apply(suggestions)
        print("  Applied. Restart the app: python web_app.py")
    else:
        print("  Not applied. Edit ood_detector.py manually.")


def _apply(suggestions: dict):
    ood_path = Path("scripts/ood_detector.py")
    lines    = ood_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        matched  = False
        for key, val in suggestions.items():
            if stripped.startswith(key) and "=" in stripped and not stripped.startswith("#"):
                orig_key_part = line.split("=")[0].rstrip()
                new_lines.append(f"{orig_key_part} = {val}")
                matched = True
                break
        if not matched:
            new_lines.append(line)
    ood_path.write_text("\n".join(new_lines), encoding="utf-8")
    log.info("Saved %s", ood_path)


if __name__ == "__main__":
    diagnose()
