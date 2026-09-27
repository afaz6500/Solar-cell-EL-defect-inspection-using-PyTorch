"""
analyse.py — Deep Batch Analysis for Solar Panel EL Images

Goes beyond prediction by providing:
  • Grad-CAM heatmap overlays (visual explanation of WHERE defects are)
  • Damage Severity Index (DSI) with confidence intervals
  • Skeleton-based defect area quantification
  • Panel fleet statistics + risk tier grouping
  • Full batch HTML report with heatmaps

Usage:
    python -m scripts.analyse                   # uses img_for_analysis/
    python -m scripts.analyse --src /some/dir   # custom folder
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from scripts.config import validate_checkpoint_classes
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scripts.ood_detector import validate_image
from scripts.config import (
    ANALYSIS_INPUT_DIR, ANALYSIS_RESULTS_DIR, CLASSES, DEVICE, IMG_SIZE,
    MODEL_PATH, SEVERITY_WEIGHTS,
)
from scripts.preprocess import enhance_image
from scripts.skeleton import OVERLAY_ALPHA, skeletonize

# ─── Config ────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = ANALYSIS_INPUT_DIR
REPORT_DIR = ANALYSIS_RESULTS_DIR

RISK_TIERS = [
    (0,  20,  "Safe",     "#22c55e"),
    (20, 40,  "Watch",    "#84cc16"),
    (40, 60,  "Caution",  "#f97316"),
    (60, 80,  "Critical", "#ef4444"),
    (80, 101, "Failure",  "#7f1d1d"),
]
NORMAL_DEFECT_AREA_MAX = 18.5

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Transforms ────────────────────────────────────────────────────────────────
INFER_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ for EfficientNet using a higher-resolution feature block.
    """
    def __init__(self, model: torch.nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None

        # features[5] preserves a 14x14 map for finer localization than the
        # final 7x7 feature block.
        target_layer = model.features[5]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        gradients_2 = gradients.pow(2)
        gradients_3 = gradients_2 * gradients
        denominator = (
            2 * gradients_2
            + (activations * gradients_3).sum(dim=(1, 2), keepdim=True)
        )
        denominator = torch.where(
            denominator.abs() > 1e-8,
            denominator,
            torch.ones_like(denominator),
        )
        alpha = gradients_2 / denominator
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        cam = F.relu((activations * weights[:, None, None]).sum(dim=0))
        cam          = cam.cpu().numpy()

        # Normalise
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def structural_anomaly_mask(orig_img: np.ndarray) -> np.ndarray:
    """Find locally dark anomalies while suppressing regular cell busbars."""
    gray = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    darkness = cv2.subtract(background, gray)
    darkness = cv2.GaussianBlur(darkness, (5, 5), 0)
    darkness_limit = max(12.0, float(np.percentile(darkness, 75)))
    intensity_limit = float(np.percentile(gray, 40))
    mask = (
        (darkness >= darkness_limit)
        & (gray <= intensity_limit)
    ).astype(np.uint8) * 255

    # Remove regular busbars and dark image borders identified by their row
    # or column median, while retaining irregular dark regions between them.
    structure_limit = float(np.percentile(gray, 30))
    busbar_rows = np.median(gray, axis=1) < structure_limit
    mask[busbar_rows, :] = 0
    border = max(4, int(IMG_SIZE * 0.035))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    cleaned = np.zeros_like(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for label in range(1, count):
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        thin_line = (
            (width > 8 * max(height, 1) and height <= 8)
            or (height > 8 * max(width, 1) and width <= 8)
        )
        if area >= 40 and not thin_line:
            cleaned[labels == label] = 255
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def crack_skeleton_mask(orig_img: np.ndarray) -> np.ndarray:
    """Extract the defect skeleton using the same pipeline as skeleton.py."""
    gray = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=1
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    return skeletonize(binary)


def skeleton_overlay(
    orig_img: np.ndarray,
    skeleton_mask: np.ndarray,
) -> np.ndarray:
    """Render the skeleton as red lines over the grayscale EL image."""
    gray = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = gray_rgb.copy()
    overlay[skeleton_mask > 0] = (0, 0, 255)
    return cv2.addWeighted(
        gray_rgb,
        1 - OVERLAY_ALPHA,
        overlay,
        OVERLAY_ALPHA,
        0,
    )


def cam_to_heatmap(
    cam: np.ndarray,
    orig_img: np.ndarray,
    structural_mask: np.ndarray,
    skeleton_mask: np.ndarray,
) -> np.ndarray:
    """Overlay broad damage in red and crack-like skeletons in cyan."""
    orig_rgb = cv2.cvtColor(
        cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_GRAY2BGR
    )
    mask_strength = cv2.GaussianBlur(
        structural_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=5,
        sigmaY=5,
    )
    # Grad-CAM++ only modulates intensity inside structurally suspicious areas.
    attention = np.clip(0.65 + 0.35 * cam, 0.0, 1.0)
    alpha = np.clip(mask_strength * attention * 0.8, 0.0, 0.8)[..., None]
    red = np.zeros_like(orig_rgb, dtype=np.float32)
    red[:, :, 2] = 255
    overlay = (
        orig_rgb.astype(np.float32) * (1.0 - alpha)
        + red * alpha
    ).astype(np.uint8)
    skeleton = cv2.dilate(
        skeleton_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    overlay[skeleton > 0] = (255, 255, 0)
    return overlay


def gradcam_overlay(cam: np.ndarray, orig_img: np.ndarray) -> np.ndarray:
    """Create a conventional Grad-CAM color overlay."""
    original = cv2.cvtColor(
        cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_GRAY2BGR
    )
    colored_cam = cv2.applyColorMap(
        np.uint8(np.clip(cam, 0.0, 1.0) * 255),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(original, 0.55, colored_cam, 0.45, 0)


def five_output_panel(
    orig_img: np.ndarray,
    cam_overlay: np.ndarray,
    damage_mask: np.ndarray,
    skeleton_mask: np.ndarray,
    combined: np.ndarray,
) -> np.ndarray:
    """Build a labeled five-view analysis strip."""
    original = cv2.cvtColor(
        cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_GRAY2BGR
    )
    damage_view = np.zeros_like(original)
    damage_view[damage_mask > 0] = (0, 0, 255)
    skeleton_view = skeleton_overlay(orig_img, skeleton_mask)

    views = [
        ("1. ORIGINAL", original),
        ("2. GRAD-CAM", cam_overlay),
        ("3. DAMAGE MASK", damage_view),
        ("4. SKELETON", skeleton_view),
        ("5. COMBINED", combined),
    ]
    title_height = 32
    panels = []
    for title, image in views:
        panel = cv2.copyMakeBorder(
            image, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20)
        )
        cv2.putText(
            panel,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return cv2.hconcat(panels)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def compute_dsi(probs: dict[str, float]) -> float:
    score = sum(probs[c] * SEVERITY_WEIGHTS[c] for c in CLASSES if c in probs)
    return round(score * 100, 2)


def dsi_to_risk(dsi: float) -> tuple[str, str]:
    for lo, hi, label, color in RISK_TIERS:
        if lo <= dsi < hi:
            return label, color
    return "Failure", "#7f1d1d"


def defect_area_pct(img_gray: np.ndarray) -> float:
    """Return % of pixels classified as defect via Otsu threshold."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return round(np.sum(binary > 0) / binary.size * 100, 2)

# ─── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: Path):
    from scripts.train import build_model
    ckpt        = torch.load(str(model_path), map_location=DEVICE)
    class_names = ckpt.get("class_names")
    validate_checkpoint_classes(class_names)
    model       = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Model loaded | val acc %.2f%% | classes: %s", ckpt.get("val_acc", 0), class_names)
    from scripts.confidence_calibration import load_calibration
    calibration = load_calibration(model_path)
    return model, class_names, calibration

# ─── Per-image analysis ────────────────────────────────────────────────────────

def analyse_image(model, grad_cam: GradCAMPlusPlus, class_names: list[str],
                  img_path: Path, heatmap_dir: Path,
                  calibration: dict | None = None,
                  force_process: bool = False) -> dict:
    
    # --- Layer 1: quality gate (before model inference) ---
    ood_pre = validate_image(img_path, probs=None)
    if not ood_pre.is_valid and not force_process:
        log.warning(" [OOD-L1] %s rejected: %s", img_path.name, ood_pre.reason)
        return {
            "file"          : img_path.name,
            "prediction"    : "REJECTED",
            "confidence"    : 0.0,
            "dsi"           : 0.0,
            "risk_tier"     : "Unknown",
            "risk_color"    : "#64748b",
            "defect_area_pct": 0.0,
            "ood_valid"     : False,
            "ood_verdict"   : ood_pre.verdict,
            "ood_reason"    : ood_pre.reason,
        }
    
    try:
        with Image.open(img_path) as source:
            orig_arr = np.array(source.convert("L"))
        pil_img = Image.fromarray(enhance_image(orig_arr))
        tensor   = INFER_TF(pil_img).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)
    except Exception as e:
        return {"file": img_path.name, "error": str(e)}

    with torch.enable_grad():
        logits = model(tensor)
        temperature = (calibration or {}).get("temperature", 1.0)
        probs = F.softmax(logits / temperature, dim=1).squeeze().detach().cpu().numpy()

    prob_dict  = {cls: float(round(probs[i], 4)) for i, cls in enumerate(class_names)}
    pred_idx   = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(round(probs[pred_idx] * 100, 2))
    dsi        = compute_dsi(prob_dict)
    risk, color = dsi_to_risk(dsi)
    area_pct   = defect_area_pct(orig_arr)
    structural_warning = pred_class == "normal" and area_pct > NORMAL_DEFECT_AREA_MAX

    # Grad-CAM heatmap
    heatmap_name = None
    try:
        cam = grad_cam.generate(tensor, pred_idx)
        anomaly_mask = structural_anomaly_mask(orig_arr)
        skeleton_mask = crack_skeleton_mask(orig_arr)
        cam_overlay = gradcam_overlay(cam, orig_arr)
        heatmap = cam_to_heatmap(cam, orig_arr, anomaly_mask, skeleton_mask)
        comparison = five_output_panel(
            orig_arr,
            cam_overlay,
            anomaly_mask,
            skeleton_mask,
            heatmap,
        )
        heatmap_name = img_path.stem + "_localization.jpg"
        cv2.imwrite(str(heatmap_dir / heatmap_name), heatmap)
        cv2.imwrite(
            str(heatmap_dir / (img_path.stem + "_five_outputs.jpg")),
            comparison,
        )
        cv2.imwrite(
            str(heatmap_dir / (img_path.stem + "_gradcam.jpg")),
            cam_overlay,
        )
        cv2.imwrite(
            str(heatmap_dir / (img_path.stem + "_damage_mask.png")),
            anomaly_mask,
        )
        cv2.imwrite(
            str(heatmap_dir / (img_path.stem + "_crack_skeleton.png")),
            skeleton_mask,
        )
        cv2.imwrite(
            str(heatmap_dir / (img_path.stem + "_skeleton_overlay.png")),
            skeleton_overlay(orig_arr, skeleton_mask),
        )
    except Exception as e:
        log.debug("Grad-CAM failed for %s: %s", img_path.name, e)

    # --- Layer 2 & 3: confidence + entropy OOD check ---
    ood = validate_image(
        img_path,
        probs=prob_dict,
        min_confidence=(calibration or {}).get("min_confidence", 0.35),
        max_entropy=(calibration or {}).get("max_entropy", 2.2),
    )
    if not ood.is_valid or structural_warning:
        warning_reason = ood.reason
        if structural_warning:
            warning_reason = (
                f"Structural defect area {area_pct:.1f}% exceeds the validated "
                f"normal-cell limit of {NORMAL_DEFECT_AREA_MAX:.1f}%."
            )
            risk, color = ("Caution", "#f97316") if area_pct >= 35 else ("Watch", "#84cc16")
        log.warning(" [OOD] %s -- %s", img_path.name, warning_reason)
        return {
            "file"        : img_path.name,
            "prediction"  : "UNCERTAIN",
            "confidence"  : confidence,
            "dsi"           : dsi,
            "risk_tier"     : risk,
            "risk_color"    : "#94a3b8",
            "defect_area_pct": area_pct,
            "ood_valid"     : False,
            "ood_verdict"   : "UNCERTAIN",
            "ood_reason"    : warning_reason,
            "ood_entropy"   : ood.entropy,
            "probabilities" : prob_dict,
            "heatmap"       : heatmap_name,
        }

    return {
        "file"          : img_path.name,
        "prediction"    : pred_class,
        "confidence"    : confidence,
        "dsi"           : dsi,
        "risk_tier"     : risk,
        "risk_color"    : color,
        "defect_area_pct": area_pct,
        "ood_valid"     : True,
        "ood_verdict"   : "VALID",
        "ood_reason"    : "Image is valid and within expected distribution.",
        "ood_entropy"   : ood.entropy,
        "probabilities" : prob_dict,
        "heatmap"       : heatmap_name,
    }

# ─── Batch analysis ────────────────────────────────────────────────────────────

def run_batch_analysis(src_dir: Path | None = None,
                       model_path: Path | None = None,
                       split_panel: bool = False,
                       panel_rows: int = 6,
                       panel_cols: int = 10,
                       panel_margin_pct: float = 0.0,
                       panel_gap_pct: float = 0.0) -> list[dict]:
    src   = src_dir   or ANALYSIS_DIR
    mpath = model_path or MODEL_PATH

    heatmap_dir = REPORT_DIR / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    exts   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    cell_lookup = {}
    if split_panel:
        from scripts.panel_splitter import split_cells_to_lookup, split_panel_folder

        split_dir = REPORT_DIR / "split_cells" / datetime.now().strftime("%Y%m%d_%H%M%S")
        cells = split_panel_folder(
            src_dir=src,
            out_dir=split_dir,
            rows=panel_rows,
            cols=panel_cols,
            margin_pct=panel_margin_pct,
            gap_pct=panel_gap_pct,
        )
        cell_lookup = split_cells_to_lookup(cells)
        images = [cell.path for cell in cells]
    else:
        images = sorted([p for p in src.rglob("*") if p.suffix.lower() in exts])

    if not images:
        log.warning("No images found in %s", src)
        return []

    log.info("Analysing %d images …", len(images))
    model, class_names, calibration = load_model(mpath)
    grad_cam           = GradCAMPlusPlus(model)

    results = []
    for img_path in tqdm(images, desc="Analysing"):
        res = analyse_image(
            model, grad_cam, class_names, img_path, heatmap_dir, calibration
        )
        if split_panel and img_path.name in cell_lookup:
            cell = cell_lookup[img_path.name]
            res.update({
                "source_panel": cell.source.name,
                "cell_row": cell.row,
                "cell_col": cell.col,
                "cell_bbox": list(cell.bbox),
            })
        results.append(res)

        if "error" not in res:
            log.info("  %-30s -> %-15s | DSI %5.1f | Risk: %-10s | Defect area: %.1f%%",
                     res["file"], res["prediction"], res["dsi"], res["risk_tier"], res["defect_area_pct"])

    _print_fleet_summary(results)
    _save_analysis(results, heatmap_dir)
    return results


def _print_fleet_summary(results: list[dict]):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    dsi_vals = [r["dsi"] for r in valid]
    tier_counts: dict[str, int] = defaultdict(int)
    for r in valid:
        tier_counts[r["risk_tier"]] += 1

    log.info("═" * 60)
    log.info("FLEET ANALYSIS SUMMARY")
    log.info("  Total panels  : %d", len(valid))
    log.info("  Avg DSI       : %.2f", np.mean(dsi_vals))
    log.info("  Max DSI       : %.2f", max(dsi_vals))
    log.info("  Min DSI       : %.2f", min(dsi_vals))
    log.info("  Risk tiers    :")
    for tier, count in sorted(tier_counts.items()):
        log.info("    %-12s : %d panels (%.1f%%)", tier, count, count / len(valid) * 100)
    log.info("═" * 60)


def _save_analysis(results: list[dict], heatmap_dir: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = REPORT_DIR / f"analysis_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # CSV
    csv_path = REPORT_DIR / f"analysis_{ts}.csv"
    fields = [
        "source_panel", "cell_row", "cell_col", "file",
        "prediction", "confidence", "dsi", "risk_tier", "defect_area_pct",
        "ood_verdict", "ood_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # HTML
    _generate_html_analysis(results, heatmap_dir, REPORT_DIR / f"analysis_{ts}.html")
    log.info("Analysis saved → %s", REPORT_DIR)


def _generate_html_analysis(results: list[dict], heatmap_dir: Path, out_path: Path):
    valid   = [r for r in results if "error" not in r]
    dsi_avg = np.mean([r["dsi"] for r in valid]) if valid else 0
    dsi_max = max((r["dsi"] for r in valid), default=0)

    # Risk distribution
    tier_counts: dict[str, int] = defaultdict(int)
    tier_colors: dict[str, str] = {}
    for r in valid:
        tier_counts[r["risk_tier"]] += 1
        tier_colors[r["risk_tier"]]  = r["risk_color"]

    tier_bars = ""
    total = max(len(valid), 1)
    for tier, cnt in sorted(tier_counts.items()):
        pct   = cnt / total * 100
        color = tier_colors.get(tier, "#94a3b8")
        tier_bars += (
            f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0'>"
            f"<span style='width:110px;font-size:13px'>{tier}</span>"
            f"<div style='background:{color};width:{pct:.1f}%;height:18px;"
            f"border-radius:4px;min-width:4px'></div>"
            f"<span style='font-size:13px'>{cnt} ({pct:.1f}%)</span>"
            f"</div>"
        )

    rows = ""
    for r in results:
        if "error" in r:
            rows += f"<tr><td>{r.get('source_panel', '')}</td><td>{r.get('cell_row', '')}</td><td>{r.get('cell_col', '')}</td><td>{r['file']}</td><td colspan='5' style='color:#f87171'>Error: {r['error']}</td></tr>"
            continue

        color = r["risk_color"]
        dsi   = r["dsi"]
        bar   = f'<div style="background:{color};width:{dsi}%;height:12px;border-radius:4px;min-width:3px"></div>'

        hmap_cell = ""
        if r.get("heatmap"):
            rel = Path("heatmaps") / r["heatmap"]
            hmap_cell = f"<a href='{rel}' target='_blank'>View</a>"

        rows += (
            f"<tr>"
            f"<td>{r.get('source_panel', '')}</td>"
            f"<td>{r.get('cell_row', '')}</td>"
            f"<td>{r.get('cell_col', '')}</td>"
            f"<td>{r['file']}</td>"
            f"<td>{r['prediction'].replace('_',' ').title()}</td>"
            f"<td>{r['confidence']:.1f}%</td>"
            f"<td style='color:{color};font-weight:600'>{r['risk_tier']}</td>"
            f"<td>{dsi:.1f} {bar}</td>"
            f"<td>{r['defect_area_pct']:.1f}%</td>"
            f"<td>{hmap_cell}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Solar Panel EL Analysis</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:'Segoe UI',sans-serif;padding:32px;background:#0f172a;color:#e2e8f0;margin:0}}
  h1{{color:#38bdf8;font-size:28px;margin-bottom:4px}}
  .sub{{color:#94a3b8;margin-bottom:28px}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:32px}}
  .kpi{{background:#1e293b;border-radius:12px;padding:20px;text-align:center}}
  .kpi .val{{font-size:36px;font-weight:700;color:#38bdf8}}
  .kpi .lbl{{font-size:13px;color:#94a3b8;margin-top:4px}}
  .section{{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:24px}}
  h2{{font-size:16px;color:#38bdf8;margin:0 0 16px 0}}
  table{{border-collapse:collapse;width:100%;font-size:14px}}
  th{{padding:10px 12px;text-align:left;color:#94a3b8;border-bottom:1px solid #334155}}
  td{{padding:9px 12px;border-bottom:1px solid #1e293b}}
  tr:hover td{{background:#0f172a}}
  a{{color:#38bdf8}}
</style></head><body>
<h1>☀️ Solar Panel EL Defect Analysis</h1>
<p class='sub'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Panels analysed: {len(results)}</p>
<div class='kpi-grid'>
  <div class='kpi'><div class='val'>{len(valid)}</div><div class='lbl'>Panels Analysed</div></div>
  <div class='kpi'><div class='val'>{dsi_avg:.1f}</div><div class='lbl'>Average DSI</div></div>
  <div class='kpi'><div class='val'>{dsi_max:.1f}</div><div class='lbl'>Peak DSI</div></div>
</div>
<div class='section'>
  <h2>Risk Tier Distribution</h2>
  {tier_bars}
</div>
<div class='section'>
  <h2>Per-Panel Results</h2>
  <table><thead><tr>
    <th>Panel</th><th>Row</th><th>Col</th><th>File</th><th>Prediction</th><th>Confidence</th>
    <th>Risk Tier</th><th>DSI</th><th>Defect Area</th><th>Heatmap</th>
  </tr></thead><tbody>{rows}</tbody></table>
</div>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar panel EL batch analysis")
    parser.add_argument("--src",   type=Path, default=None, help="Source image directory")
    parser.add_argument("--model", type=Path, default=None, help="Model checkpoint path")
    parser.add_argument("--split-panel", action="store_true", help="Split whole-panel images into cells first")
    parser.add_argument("--panel-rows", type=int, default=6, help="Panel grid rows")
    parser.add_argument("--panel-cols", type=int, default=10, help="Panel grid columns")
    parser.add_argument("--panel-margin-pct", type=float, default=0.0, help="Percent margin trimmed from panel edges")
    parser.add_argument("--panel-gap-pct", type=float, default=0.0, help="Percent trimmed inside each cell")
    args = parser.parse_args()
    run_batch_analysis(
        src_dir=args.src,
        model_path=args.model,
        split_panel=args.split_panel,
        panel_rows=args.panel_rows,
        panel_cols=args.panel_cols,
        panel_margin_pct=args.panel_margin_pct,
        panel_gap_pct=args.panel_gap_pct,
    )
