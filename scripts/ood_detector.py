# -*- coding: utf-8 -*-
"""
ood_detector.py -- Out-of-Distribution (OOD) Detection for Solar Panel EL Images

Three layers, applied in order:

  Layer 1  | Basic sanity gate  (size, blank, colour photo)
  Layer 2  | Confidence threshold  (model must be > 50% confident)
  Layer 3  | Entropy score  (confused model = likely non-EL image)

NOTE: The structural EL fingerprint checks (brightness, edge density, etc.)
were removed because they are dataset-specific and caused false rejections
on valid EL images without a per-dataset calibration step.
Run diagnose_ood.py if you want to re-enable them with your own thresholds.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Layer 1 thresholds ────────────────────────────────────────────────────────
MIN_CONFIDENCE      = 0.35    # below this -> UNCERTAIN
MAX_ENTROPY         = 2.20    # bits; max possible for 10 classes ≈ 3.32
MIN_COLOR_STD       = 3.0     # below this = blank / solid image
MAX_COLOR_STD       = 120.0   # above this = extreme noise
MAX_SATURATION_MEAN = 60.0    # above this = colour photo, not EL
MIN_SIZE            = 16      # px, minimum edge length; split panel cells can be narrow
MAX_SIZE            = 10000   # px, maximum edge length

# ── Structural fingerprint thresholds (DISABLED by default) ───────────────────
# Set ENABLE_STRUCTURAL_CHECKS = True only after running calibrate_ood.py
ENABLE_STRUCTURAL_CHECKS = False

EL_MIN_MEAN_BRIGHTNESS = 20.0
EL_MAX_MEAN_BRIGHTNESS = 245.0
EL_MIN_DARK_RATIO      = 0.0
EL_MAX_DARK_RATIO      = 0.99
EL_MAX_EDGE_DENSITY    = 0.22
EL_MAX_LAPLACIAN_VAR   = 180.0
EL_MIN_ASPECT_RATIO    = 0.65
EL_MAX_ASPECT_RATIO    = 1.35


@dataclass
class OODResult:
    is_valid:        bool
    verdict:         str    # VALID | REJECTED | UNCERTAIN
    reason:          str
    confidence:      float
    entropy:         float
    color_std:       float
    saturation:      float
    layer_failed:    int    # 0=OK 1=sanity 2=structural 3=confidence 4=entropy
    mean_brightness: float = 0.0
    dark_ratio:      float = 0.0
    edge_density:    float = 0.0
    laplacian_var:   float = 0.0
    aspect_ratio:    float = 0.0


def _check_sanity(img_bgr, stats: dict) -> tuple[bool, str]:
    """Layer 1: basic sanity checks that work for any EL dataset."""
    h, w = img_bgr.shape[:2]

    if min(h, w) < MIN_SIZE:
        return False, f"Image too small ({w}x{h}px)"
    if max(h, w) > MAX_SIZE:
        return False, f"Image too large ({w}x{h}px)"

    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    color_std = float(np.std(gray))
    stats["color_std"] = round(color_std, 2)

    if color_std < MIN_COLOR_STD:
        return False, f"Image is blank or solid colour (std={color_std:.1f})"
    if color_std > MAX_COLOR_STD:
        return False, f"Image is extremely noisy (std={color_std:.1f})"

    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    stats["saturation"] = round(sat_mean, 2)

    if sat_mean > MAX_SATURATION_MEAN:
        return False, (
            f"Image appears to be a colour photo, not a grayscale EL image "
            f"(saturation={sat_mean:.1f})"
        )
    return True, "OK"


def _check_structural(img_bgr) -> tuple[bool, str, dict]:
    """Layer 1b: EL-specific structural checks (disabled by default)."""
    fp = dict(mean_brightness=0.0, dark_ratio=0.0,
               edge_density=0.0, laplacian_var=0.0, aspect_ratio=0.0)

    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w     = gray.shape
    total_px = max(h * w, 1)

    mb = float(np.mean(gray))
    fp["mean_brightness"] = round(mb, 2)
    if mb < EL_MIN_MEAN_BRIGHTNESS:
        return False, f"Image too dark for EL (brightness={mb:.1f})", fp
    if mb > EL_MAX_MEAN_BRIGHTNESS:
        return False, f"Image too bright for EL (brightness={mb:.1f})", fp

    dr = float(np.sum(gray < 50) / total_px)
    fp["dark_ratio"] = round(dr, 4)
    if dr > EL_MAX_DARK_RATIO:
        return False, f"Image is almost entirely black (dark_ratio={dr:.2f})", fp

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ed      = float(np.sum(cv2.Canny(blurred, 30, 90) > 0) / total_px)
    fp["edge_density"] = round(ed, 4)
    if ed > EL_MAX_EDGE_DENSITY:
        return False, f"Too many sharp edges for EL (edge_density={ed:.3f})", fp

    lv = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    fp["laplacian_var"] = round(lv, 2)
    if lv > EL_MAX_LAPLACIAN_VAR:
        return False, f"Image too sharp/detailed for EL (laplacian_var={lv:.1f})", fp

    ar = float(max(w, h)) / float(max(min(w, h), 1))
    fp["aspect_ratio"] = round(ar, 3)
    if ar < EL_MIN_ASPECT_RATIO or ar > EL_MAX_ASPECT_RATIO:
        return False, f"Unusual aspect ratio (ratio={ar:.2f})", fp

    return True, "OK", fp


def _compute_entropy(probs) -> float:
    entropy = 0.0
    for p in probs:
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _check_prediction(
    probs: dict,
    min_confidence: float,
    max_entropy: float,
) -> tuple[bool, str, float, float, int]:
    vals     = list(probs.values())
    top_conf = max(vals)
    entropy  = _compute_entropy(vals)

    if top_conf < min_confidence:
        return (False,
                f"Low model confidence ({top_conf*100:.1f}% < {min_confidence*100:.0f}%) "
                f"-- model is uncertain, image may not be an EL panel",
                top_conf, entropy, 3)

    if entropy > max_entropy:
        return (False,
                f"High prediction entropy ({entropy:.3f} bits > {max_entropy:.3f}) "
                f"-- model cannot classify this image as any known panel type",
                top_conf, entropy, 4)

    return True, "OK", top_conf, entropy, 0


def validate_image(
    img_path: Path,
    probs: dict | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    max_entropy: float = MAX_ENTROPY,
) -> OODResult:
    """
    Run OOD validation on an image.

    Args:
        img_path : path to the image
        probs    : {class_name: probability} from the model.
                   Pass None to run Layer 1 only (before inference).
    """
    stats = {"color_std": 0.0, "saturation": 0.0}
    fp    = dict(mean_brightness=0.0, dark_ratio=0.0,
                  edge_density=0.0, laplacian_var=0.0, aspect_ratio=0.0)

    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return OODResult(False, "REJECTED",
                             "Cannot read image (corrupt or unsupported format)",
                             0.0, 0.0, 0.0, 0.0, 1)
    except Exception as e:
        return OODResult(False, "REJECTED", f"Image read error: {e}",
                         0.0, 0.0, 0.0, 0.0, 1)

    # Layer 1: sanity
    ok, reason = _check_sanity(img_bgr, stats)
    if not ok:
        return OODResult(False, "REJECTED", f"[Layer 1] {reason}",
                         0.0, 0.0, stats["color_std"], stats["saturation"], 1)

    # Layer 1b: structural (only if enabled via calibrate_ood.py)
    if ENABLE_STRUCTURAL_CHECKS:
        ok, reason, fp = _check_structural(img_bgr)
        if not ok:
            return OODResult(False, "REJECTED", f"[Layer 1b] {reason}",
                             0.0, 0.0, stats["color_std"], stats["saturation"], 2,
                             fp["mean_brightness"], fp["dark_ratio"],
                             fp["edge_density"], fp["laplacian_var"], fp["aspect_ratio"])
    else:
        # Still compute fp stats for the dashboard display even if not checking
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_px = max(h * w, 1)
        fp["mean_brightness"] = round(float(np.mean(gray)), 2)
        fp["dark_ratio"]      = round(float(np.sum(gray < 50) / total_px), 4)
        fp["aspect_ratio"]    = round(float(max(w,h)) / float(max(min(w,h),1)), 3)

    # Layers 2 & 3: model-based
    if probs is not None:
        ok, reason, confidence, entropy, layer = _check_prediction(
            probs, min_confidence, max_entropy
        )
        if not ok:
            lname = "Confidence" if layer == 3 else "Entropy"
            return OODResult(False, "UNCERTAIN",
                             f"[Layer {layer} - {lname}] {reason}",
                             confidence, entropy,
                             stats["color_std"], stats["saturation"], layer,
                             fp["mean_brightness"], fp["dark_ratio"],
                             fp["edge_density"], fp["laplacian_var"], fp["aspect_ratio"])

        return OODResult(True, "VALID", "OK",
                         confidence, entropy,
                         stats["color_std"], stats["saturation"], 0,
                         fp["mean_brightness"], fp["dark_ratio"],
                         fp["edge_density"], fp["laplacian_var"], fp["aspect_ratio"])

    return OODResult(True, "VALID (pre-check)", "OK",
                     0.0, 0.0, stats["color_std"], stats["saturation"], 0,
                     fp["mean_brightness"], fp["dark_ratio"],
                     fp["edge_density"], fp["laplacian_var"], fp["aspect_ratio"])


def format_ood_result(result: OODResult) -> str:
    if result.is_valid:
        return (f"[VALID] conf={result.confidence*100:.1f}% | "
                f"entropy={result.entropy:.3f} | std={result.color_std}")
    return f"[{result.verdict}] {result.reason}"
