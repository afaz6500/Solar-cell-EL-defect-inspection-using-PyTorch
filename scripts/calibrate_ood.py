"""
calibrate_ood.py -- Auto-calibrate OOD thresholds from your actual EL dataset

Run this ONCE after training:
    python calibrate_ood.py

What it does:
  1. Scans all images in dataset/train/ (your confirmed EL images)
  2. Measures 6 structural properties per image
  3. Uses p0.5 / p99.5 percentiles + generous safety margin so real EL
     images are never rejected (false negative rate < 1%)
  4. Auto-applies thresholds to ood_detector.py
  5. Saves diagnostic CSV + JSON for inspection
"""

import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from scripts.config import TRAIN_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUT_CSV    = Path("results/ood_calibration.csv")
OUT_JSON   = Path("results/ood_thresholds.json")

# Safety margin applied BEYOND the observed percentile range.
# 25% gives generous buffer so legitimate EL images are never rejected.
MARGIN = 0.25


def measure_image(img_path: Path) -> dict | None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w     = gray.shape
    total_px = h * w

    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    mb       = float(np.mean(gray))
    std      = float(np.std(gray))
    dr       = float(np.sum(gray < 50) / total_px)
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blurred, 30, 90)
    ed       = float(np.sum(edges > 0) / total_px)
    lv       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ar       = max(w, h) / max(min(w, h), 1)

    return {
        "file"            : img_path.name,
        "class"           : img_path.parent.name,
        "width"           : w,
        "height"          : h,
        "mean_brightness" : round(mb,  2),
        "color_std"       : round(std, 2),
        "dark_ratio"      : round(dr,  4),
        "edge_density"    : round(ed,  4),
        "laplacian_var"   : round(lv,  2),
        "saturation"      : round(sat_mean, 2),
        "aspect_ratio"    : round(ar,  3),
    }


def calibrate():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    exts   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in TRAIN_DIR.rglob("*") if p.suffix.lower() in exts]

    if not images:
        log.error("No images found in %s -- run preprocessing first.", TRAIN_DIR)
        return

    log.info("Measuring %d EL images ...", len(images))
    rows = []
    for p in tqdm(images, desc="Calibrating"):
        r = measure_image(p)
        if r:
            rows.append(r)

    if not rows:
        log.error("No images could be read.")
        return

    # Save raw CSV for manual inspection
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info("Diagnostic CSV -> %s", OUT_CSV)

    metrics = [
        "mean_brightness", "color_std", "dark_ratio",
        "edge_density", "laplacian_var", "saturation", "aspect_ratio",
    ]

    stats = {}
    for m in metrics:
        vals = [r[m] for r in rows]
        stats[m] = {
            "min"  : round(float(min(vals)),                      4),
            "max"  : round(float(max(vals)),                      4),
            "mean" : round(float(np.mean(vals)),                  4),
            "p0_5" : round(float(np.percentile(vals,  0.5)),      4),
            "p5"   : round(float(np.percentile(vals,  5)),        4),
            "p95"  : round(float(np.percentile(vals, 95)),        4),
            "p99_5": round(float(np.percentile(vals, 99.5)),      4),
        }

    def lo(metric):
        """Lower bound: p0.5 minus margin (never below 0)."""
        return round(max(0.0, stats[metric]["p0_5"] * (1 - MARGIN)), 4)

    def hi(metric, cap=None):
        """Upper bound: p99.5 plus margin."""
        val = round(stats[metric]["p99_5"] * (1 + MARGIN), 4)
        return val if cap is None else min(val, cap)

    thresholds = {
        # Layer 1 basic
        "MIN_COLOR_STD"          : lo("color_std"),
        "MAX_COLOR_STD"          : hi("color_std"),
        "MAX_SATURATION_MEAN"    : max(10.0, hi("saturation")),  # clamp: never disable colour check
        # Layer 1b EL fingerprint
        "EL_MIN_MEAN_BRIGHTNESS" : lo("mean_brightness"),
        "EL_MAX_MEAN_BRIGHTNESS" : hi("mean_brightness"),
        "EL_MIN_DARK_RATIO"      : lo("dark_ratio"),
        "EL_MAX_DARK_RATIO"      : hi("dark_ratio", cap=0.99),
        "EL_MAX_EDGE_DENSITY"    : hi("edge_density"),
        "EL_MAX_LAPLACIAN_VAR"   : hi("laplacian_var"),
        "EL_MIN_ASPECT_RATIO"    : lo("aspect_ratio"),
        "EL_MAX_ASPECT_RATIO"    : hi("aspect_ratio"),
    }

    OUT_JSON.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    # Print stats table
    print("\n" + "=" * 68)
    print("  EL DATASET STATISTICS  (from your training images)")
    print("=" * 68)
    print(f"  {'Metric':<22} {'Min':>8} {'P0.5':>8} {'P99.5':>8} {'Max':>8}")
    print("-" * 68)
    for m in metrics:
        s = stats[m]
        print(f"  {m:<22} {s['min']:>8.3f} {s['p0_5']:>8.3f} "
              f"{s['p99_5']:>8.3f} {s['max']:>8.3f}")

    print("\n" + "=" * 68)
    print("  CALIBRATED THRESHOLDS  (auto-applied to ood_detector.py)")
    print("=" * 68)
    for k, v in thresholds.items():
        print(f"  {k:<28} = {v}")
    print("=" * 68)

    _apply_to_ood_detector(thresholds)

    # Verify: count how many training images would be rejected
    rejected = _verify(rows, thresholds)
    total    = len(rows)
    pct      = rejected / total * 100
    print(f"\n  Self-check: {rejected}/{total} training images rejected ({pct:.1f}%)")
    if pct > 2.0:
        print("  WARNING: more than 2% of your own EL images are being rejected.")
        print("  Consider increasing MARGIN (currently 25%) and re-running.")
    else:
        print("  OK -- fewer than 2% of real EL images are rejected.")
    print()


def _verify(rows: list[dict], t: dict) -> int:
    """Count how many measured EL images would be rejected by the new thresholds."""
    rejected = 0
    for r in rows:
        if (r["color_std"]       < t["MIN_COLOR_STD"]          or
            r["color_std"]       > t["MAX_COLOR_STD"]           or
            r["saturation"]      > t["MAX_SATURATION_MEAN"]     or
            r["mean_brightness"] < t["EL_MIN_MEAN_BRIGHTNESS"]  or
            r["mean_brightness"] > t["EL_MAX_MEAN_BRIGHTNESS"]  or
            r["dark_ratio"]      < t["EL_MIN_DARK_RATIO"]       or
            r["dark_ratio"]      > t["EL_MAX_DARK_RATIO"]       or
            r["edge_density"]    > t["EL_MAX_EDGE_DENSITY"]     or
            r["laplacian_var"]   > t["EL_MAX_LAPLACIAN_VAR"]    or
            r["aspect_ratio"]    < t["EL_MIN_ASPECT_RATIO"]     or
            r["aspect_ratio"]    > t["EL_MAX_ASPECT_RATIO"]):
            rejected += 1
    return rejected


def _apply_to_ood_detector(thresholds: dict):
    ood_path = Path("scripts/ood_detector.py")
    if not ood_path.exists():
        log.warning("ood_detector.py not found -- cannot auto-apply.")
        return

    lines     = ood_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    applied   = set()

    for line in lines:
        stripped = line.strip()
        matched  = False
        for key, val in thresholds.items():
            # Match the key at the start of the stripped line (handles any spacing)
            if stripped.startswith(key) and "=" in stripped and not stripped.startswith("#"):
                # Preserve original spacing style between key and =
                orig_key_part = line.split("=")[0].rstrip()
                new_lines.append(f"{orig_key_part} = {val}")
                applied.add(key)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    ood_path.write_text("\n".join(new_lines), encoding="utf-8")
    log.info("Auto-applied %d/%d thresholds to %s",
             len(applied), len(thresholds), ood_path)

    missing = set(thresholds) - applied
    if missing:
        log.warning("Could not find these lines to replace: %s", missing)
        log.warning("Paste them manually from the table above.")


if __name__ == "__main__":
    calibrate()
