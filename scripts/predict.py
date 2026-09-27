"""
predict.py — Batch Solar Panel Defect Prediction with Damage Severity Scale

Reads images from img_for_prediction/ (or any folder), runs the trained CNN,
and outputs:
  • Class label
  • Confidence score
  • Damage Severity Index (DSI) — a 0–100 continuous scale
  • Per-image JSON result
  • Batch summary CSV + HTML report

Damage Severity Index (DSI) mapping
────────────────────────────────────
  0  – 20  → No damage          (green)
 20  – 40  → Minor damage       (yellow-green)
 40  – 60  → Moderate damage    (orange)
 60  – 80  → Significant damage (orange-red)
 80  – 100 → Severe damage      (red)

The DSI is computed as:
  DSI = weighted_sum(class_probabilities × class_severity_weights) × 100
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scripts.config import validate_checkpoint_classes
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scripts.ood_detector import validate_image
from scripts.config import (
    CLASSES, DEVICE, IMG_SIZE, MODEL_PATH, PREDICTION_INPUT_DIR,
    PREDICTIONS_DIR, SEVERITY_WEIGHTS,
)
from scripts.preprocess import enhance_image

# ─── Config ────────────────────────────────────────────────────────────────────
PREDICT_DIR = PREDICTION_INPUT_DIR
RESULTS_DIR = PREDICTIONS_DIR

# DSI thresholds
DSI_LEVELS = [
    (0,  20,  "No Damage",          "🟢"),
    (20, 40,  "Minor Damage",       "🟡"),
    (40, 60,  "Moderate Damage",    "🟠"),
    (60, 80,  "Significant Damage", "🔴"),
    (80, 101, "Severe Damage",      "🚨"),
]

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

# ─── Helpers ───────────────────────────────────────────────────────────────────

def dsi_to_level(dsi: float) -> tuple[str, str]:
    for lo, hi, label, icon in DSI_LEVELS:
        if lo <= dsi < hi:
            return label, icon
    return "Severe Damage", "🚨"


def compute_dsi(probs: dict[str, float]) -> float:
    """Weighted average of class probabilities × severity weights → [0, 100]."""
    score = sum(probs[c] * SEVERITY_WEIGHTS[c] for c in CLASSES if c in probs)
    return round(score * 100, 2)


def load_model(model_path: Path):
    """Load trained model checkpoint and return (model, class_names)."""
    from scripts.train import build_model   # import here to avoid circular deps

    checkpoint   = torch.load(str(model_path), map_location=DEVICE)
    class_names  = checkpoint.get("class_names")
    validate_checkpoint_classes(class_names)
    model        = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    log.info("Model loaded (val acc %.2f%%) | classes: %s",
             checkpoint.get("val_acc", 0), class_names)
    from scripts.confidence_calibration import load_calibration
    calibration = load_calibration(model_path)
    return model, class_names, calibration


def predict_image(
    model,
    class_names: list[str],
    img_path: Path,
    calibration: dict | None = None,
    force_process: bool = False,
) -> dict:
    """Run inference on a single image and return a result dict."""

    # --- Layer 1: quality gate (before model inference) ---
    ood_pre = validate_image(img_path, probs=None)
    if not ood_pre.is_valid and not force_process:
        return{
            "file"          : img_path.name,
            "prediction"    : "REJECTED",
            "confidence"    : 0.0,
            "dsi"           : 0.0,
            "dsi_level"     : "Unknown",
            "dsi_icon"      : "❓",
            "ood_valid"     : False,
            "ood_verdict"   : ood_pre.verdict,
            "ood_reason"    : ood_pre.reason,
        }
    
    try:
        with Image.open(img_path) as source:
            raw = np.array(source.convert("L"))
            img = Image.fromarray(enhance_image(raw))
            tensor = INFER_TF(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return {"file": img_path.name, "error": str(e)}

    with torch.no_grad():
        logits = model(tensor)
        temperature = (calibration or {}).get("temperature", 1.0)
        probs = F.softmax(logits / temperature, dim=1).squeeze().cpu().numpy()

    prob_dict     = {cls: float(round(probs[i], 4)) for i, cls in enumerate(class_names)}
    pred_idx      = int(np.argmax(probs))
    pred_class    = class_names[pred_idx]
    confidence    = float(round(probs[pred_idx] * 100, 2))
    dsi           = compute_dsi(prob_dict)
    level, icon   = dsi_to_level(dsi)

    # --- Layers 2 & 3: confidence + entropy OOD check ---
    ood = validate_image(
        img_path,
        prob_dict,
        min_confidence=(calibration or {}).get("min_confidence", 0.35),
        max_entropy=(calibration or {}).get("max_entropy", 2.2),
    )
    if not ood.is_valid:
        return {
            "file"        : img_path.name,
            "prediction"  : "UNCERTAIN",
            "confidence"  : confidence,
            "dsi"         : dsi,
            "dsi_level"   : level,
            "dsi_icon"    : icon,
            "ood_valid"   : False,
            "ood_verdict" : ood.verdict,
            "ood_reason"  : ood.reason,
            "ood_entropy" : ood.entropy,
            "probabilities": prob_dict,
        }

    return {
        "file"        : img_path.name,
        "prediction"  : pred_class,
        "confidence"  : confidence,
        "dsi"         : dsi,
        "dsi_level"   : level,
        "dsi_icon"    : icon,
        "ood_valid"   : True,
        "ood_verdict" : "VALID",
        "ood_reason"  : "Image is valid and within expected distribution.",
        "ood_entropy" : ood.entropy,
        "probabilities": prob_dict,
    }


# ─── Batch runner ──────────────────────────────────────────────────────────────

def run_batch_prediction(src_dir: Path | None = None,
                         model_path: Path | None = None,
                         split_panel: bool = False,
                         panel_rows: int = 6,
                         panel_cols: int = 10,
                         panel_margin_pct: float = 0.0,
                         panel_gap_pct: float = 0.0) -> list[dict]:

    src   = src_dir   or PREDICT_DIR
    mpath = model_path or MODEL_PATH
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exts   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    cell_lookup = {}
    if split_panel:
        from scripts.panel_splitter import split_cells_to_lookup, split_panel_folder

        split_dir = RESULTS_DIR / "split_cells" / datetime.now().strftime("%Y%m%d_%H%M%S")
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

    log.info("Running batch prediction on %d images …", len(images))
    model, class_names, calibration = load_model(mpath)

    results = []
    for img_path in tqdm(images, desc="Predicting"):
        res = predict_image(model, class_names, img_path, calibration)
        if split_panel and img_path.name in cell_lookup:
            cell = cell_lookup[img_path.name]
            res.update({
                "source_panel": cell.source.name,
                "cell_row": cell.row,
                "cell_col": cell.col,
                "cell_bbox": list(cell.bbox),
            })
        results.append(res)
        ood_tag = "" if res.get("ood_valid", True) else f" [{res.get('ood_verdict','?')}]"
        log.info("  %s -> %s%s (conf %.1f%% | DSI %.1f | %s)",
                 res["file"],
                 res.get("prediction", "ERROR"),
                 ood_tag,
                 res.get("confidence", 0),
                 res.get("dsi", 0),
                 res.get("dsi_level", ""),
                 )

    _save_results(results)
    return results


def _save_results(results: list[dict]):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = RESULTS_DIR / f"predictions_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # CSV
    csv_path = RESULTS_DIR / f"predictions_{ts}.csv"
    fieldnames = [
        "source_panel", "cell_row", "cell_col", "file",
        "prediction", "confidence", "dsi", "dsi_level", "dsi_icon",
        "ood_verdict", "ood_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # HTML report
    _generate_html_report(results, RESULTS_DIR / f"report_{ts}.html")

    log.info("Results saved → %s | %s", json_path, csv_path)


def _generate_html_report(results: list[dict], out_path: Path):
    dsi_color = {
        "No Damage"          : "#22c55e",
        "Minor Damage"       : "#a3e635",
        "Moderate Damage"    : "#f97316",
        "Significant Damage" : "#ef4444",
        "Severe Damage"      : "#991b1b",
    }
    rows = ""
    for r in results:
        if "error" in r:
            rows += f"<tr><td>{r['file']}</td><td colspan='4' style='color:red'>Error: {r['error']}</td></tr>"
            continue
        color = dsi_color.get(r["dsi_level"], "#94a3b8")
        bar   = f'<div style="background:{color};width:{r["dsi"]}%;height:14px;border-radius:4px"></div>'
        rows += (
            f"<tr>"
            f"<td>{r.get('source_panel', '')}</td>"
            f"<td>{r.get('cell_row', '')}</td>"
            f"<td>{r.get('cell_col', '')}</td>"
            f"<td>{r['file']}</td>"
            f"<td>{r['prediction'].replace('_', ' ').title()}</td>"
            f"<td>{r['confidence']:.1f}%</td>"
            f"<td>{r['dsi_icon']} {r['dsi_level']} ({r['dsi']:.1f})</td>"
            f"<td style='min-width:120px'>{bar}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Solar Panel Defect Report</title>
<style>
  body{{font-family:sans-serif;padding:24px;background:#0f172a;color:#e2e8f0}}
  h1{{color:#38bdf8}} table{{border-collapse:collapse;width:100%}}
  th{{background:#1e293b;padding:10px;text-align:left}}
  td{{padding:8px 10px;border-bottom:1px solid #1e293b}}
  tr:hover{{background:#1e293b}}
</style></head><body>
<h1>☀️ Solar Panel Defect Prediction Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Images: {len(results)}</p>
<table><thead><tr>
  <th>Panel</th><th>Row</th><th>Col</th><th>File</th><th>Prediction</th><th>Confidence</th><th>Damage Level</th><th>DSI Bar</th>
</tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    run_batch_prediction()
