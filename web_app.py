r"""Local EL inspection beta. Run: venv\Scripts\python.exe web_app.py"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from PIL import Image, ImageOps
from scripts import web_store

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
JOBS = ROOT / "results" / "web_jobs"
MODEL = ROOT / "models" / "solar_defect_cnn_10class.pth"
CALIBRATION = ROOT / "models" / "confidence_calibration_10class.json"
MAX_FILE = 20 * 1024 * 1024
MAX_BATCH = 100 * 1024 * 1024
MAX_FILES = 50
MAX_CELLS = 2000
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection")
STATE = {}
ENGINE = None
MODEL_HASH = None


def now():
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid inspection ID.")
    return JOBS / job_id


def save(job):
    web_store.save_job(JOBS, job)
    folder = job_dir(job["id"])
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / "job.tmp"
    temp.write_text(json.dumps(job, indent=2), encoding="utf-8")
    temp.replace(folder / "job.json")


def settings(data):
    mode = data.get("mode", "cells")
    if mode not in ("cells", "panel"):
        raise ValueError("Choose individual cells or whole panels.")
    result = {"mode": mode, "visuals": bool(data.get("visuals", False))}
    for key, default, maximum in (("rows", 6, 24), ("cols", 12, 24)):
        value = float(data.get(key, default))
        if not math.isfinite(value) or not value.is_integer() or not 1 <= value <= maximum:
            raise ValueError(f"{key} must be a whole number from 1 to {maximum}.")
        result[key] = int(value)
    for key in ("margin", "gap"):
        value = float(data.get(key, 0))
        if not math.isfinite(value) or not 0 <= value <= 20:
            raise ValueError(f"{key} must be between 0 and 20 percent.")
        result[key] = value
    if "margins" in data:
        margins = data["margins"]
        if not isinstance(margins, dict) or set(margins) != {"left", "top", "right", "bottom"}:
            raise ValueError("Provide left, top, right, and bottom margins in pixels.")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 10000 for v in margins.values()):
            raise ValueError("Margins must be whole pixels from 0 to 10,000.")
        result["margins"] = margins
    return result


def model_info():
    return {"ready": MODEL.is_file() and CALIBRATION.is_file(), "classes": 10,
            "model": MODEL.name, "calibrated": CALIBRATION.is_file(),
            "limits": {"files": MAX_FILES, "file_mb": 20, "batch_mb": 100, "cells": MAX_CELLS}}


def engine():
    global ENGINE, MODEL_HASH
    if ENGINE is None:
        from scripts.predict import load_model
        from scripts.analyse import GradCAMPlusPlus
        if not MODEL.exists() or not CALIBRATION.exists():
            raise ValueError("Train and calibrate the 10-class model before starting an inspection.")
        MODEL_HASH = hashlib.sha256(MODEL.read_bytes()).hexdigest()
        calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        if calibration.get("checkpoint_sha256") != MODEL_HASH:
            raise ValueError("Calibration does not match the model. Run calibrate-confidence again.")
        model, classes, calibration = load_model(MODEL)
        ENGINE = (model, classes, calibration, GradCAMPlusPlus(model))
    return ENGINE


def process(job_id):
    try:
        from scripts.panel_splitter import split_panel_image
        from scripts.predict import predict_image
        from scripts.analyse import analyse_image
        with LOCK:
            job = STATE[job_id]
            if job["status"] == "cancelled":
                return
            job.update(status="running", stage="Loading calibrated model")
            save(job)
        model, classes, calibration, cam = engine()
        folder = job_dir(job_id)
        heatmaps = folder / "visuals"
        heatmaps.mkdir(exist_ok=True)
        with LOCK:
            job.update(model_sha256=MODEL_HASH, calibration=calibration)
            save(job)
        for source in list(job["sources"]):
            if job.get("cancel_requested"):
                break
            try:
                path = folder / source["path"]
                if job["settings"]["mode"] == "panel":
                    cfg = job["settings"]
                    cells = split_panel_image(path, folder / "crops", cfg["rows"], cfg["cols"], cfg["margin"], cfg["gap"], margins=cfg.get("margins"))
                    entries = [(cell.path, cell.row, cell.col, list(cell.bbox)) for cell in cells]
                else:
                    entries = [(path, None, None, None)]
                for path, row, col, bbox in entries:
                    if job.get("cancel_requested"):
                        break
                    with LOCK:
                        job["stage"] = f"Inspecting {source['name']}" + (f" · R{row} C{col}" if row else "")
                    try:
                        if job["settings"]["visuals"]:
                            result = analyse_image(model, cam, classes, path, heatmaps, calibration)
                        else:
                            result = predict_image(model, classes, path, calibration)
                        if "error" in result:
                            result["prediction"] = "ERROR"
                        if result.get("prediction") in ("REJECTED", "ERROR"):
                            result.update(dsi=None, confidence=None)
                        probabilities = result.get("probabilities", {})
                        result["likely_class"] = max(probabilities, key=probabilities.get) if probabilities else None
                    except Exception as exc:
                        logging.exception("Cell inference failed")
                        result = {"prediction": "ERROR", "error": str(exc), "dsi": None, "confidence": None}
                    result.update(source=source["name"], source_id=source["id"], row=row, col=col, bbox=bbox,
                                  image=path.relative_to(folder).as_posix(), index=len(job["results"]))
                    if result.get("heatmap"):
                        visual = heatmaps / (path.stem + "_five_outputs.jpg")
                        if visual.exists():
                            result["visual"] = visual.relative_to(folder).as_posix()
                    with LOCK:
                        job["results"].append(result)
                        job["done"] += 1
                        save(job)
            except Exception as exc:
                with LOCK:
                    job["errors"].append({"source": source["name"], "message": str(exc)})
                    save(job)
        with LOCK:
            job["status"] = "cancelled" if job.get("cancel_requested") else "completed"
            job.update(stage="Inspection stopped" if job["status"] == "cancelled" else "Inspection complete", finished=now())
            save(job)
    except Exception as exc:
        logging.exception("Inspection failed")
        with LOCK:
            job = STATE[job_id]
            job.update(status="failed", stage=str(exc), finished=now())
            save(job)


def retry_rejected(job_id):
    """Retry rejected crops, retaining original OOD evidence and locations."""
    try:
        from scripts.predict import predict_image
        from scripts.analyse import analyse_image
        with LOCK:
            job = STATE[job_id]
            if job.get("cancel_requested"):
                return
            job.update(status="running", stage="Processing rejected cells with OOD override")
            originals = [r.copy() for r in job["results"] if r.get("prediction") == "REJECTED"]
            save(job)
        model, classes, calibration, cam = engine()
        folder = job_dir(job_id)
        heatmaps = folder / "visuals"
        heatmaps.mkdir(exist_ok=True)
        for original in originals:
            if job.get("cancel_requested"):
                break
            path = folder / original["image"]
            try:
                if job["settings"]["visuals"]:
                    result = analyse_image(model, cam, classes, path, heatmaps, calibration, force_process=True)
                else:
                    result = predict_image(model, classes, path, calibration, force_process=True)
                if "error" in result:
                    result.update(prediction="ERROR", confidence=None, dsi=None)
                probabilities = result.get("probabilities", {})
                result["likely_class"] = max(probabilities, key=probabilities.get) if probabilities else None
                if result.get("heatmap"):
                    visual = heatmaps / (path.stem + "_five_outputs.jpg")
                    if visual.exists():
                        result["visual"] = visual.relative_to(folder).as_posix()
            except Exception as exc:
                result = dict(prediction="ERROR", error=str(exc), confidence=None, dsi=None)
            result.update({k: original[k] for k in ("source", "source_id", "row", "col", "bbox", "image", "index") if k in original})
            result.update(ood_override=True, ood_valid=False, original_ood_reason=original.get("ood_reason", ""), override_at=now())
            result["ood_reason"] = "Processed with OOD override. Original rejection: " + original.get("ood_reason", "") + " | " + result.get("ood_reason", "")
            with LOCK:
                job["results"][original["index"]] = result
                save(job)
        with LOCK:
            job.update(status="cancelled" if job.get("cancel_requested") else "completed",
                       stage="Inspection stopped" if job.get("cancel_requested") else "Rejected-cell processing complete", finished=now())
            save(job)
    except Exception as exc:
        logging.exception("Rejected-cell processing failed")
        with LOCK:
            job = STATE[job_id]
            job.update(status="failed", stage=str(exc), finished=now())
            save(job)


def restore():
    JOBS.mkdir(parents=True, exist_ok=True)
    for job in web_store.load_jobs(JOBS):
        STATE[job['id']] = job
    for path in JOBS.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job['id'] not in STATE:
                job.setdefault('profile', web_store.profile(JOBS))
                web_store.save_job(JOBS, job)
                STATE[job["id"]] = job
        except (ValueError, KeyError, OSError):
            logging.warning("Could not restore %s", path)
    for job in STATE.values():
        if job['status'] in ('queued', 'running', 'uploading'):
            job.update(status='interrupted', stage='Server stopped. Start a new inspection to retry.')
            save(job)


class Handler(BaseHTTPRequestHandler):
    def respond(self, data, code=200, content_type="application/json", filename=None):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def body(self, limit=65536):
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= limit:
            raise ValueError("Upload is empty or exceeds the size limit.")
        self.connection.settimeout(30)
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("Upload was interrupted.")
        return data

    def dispatch(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed or (self.headers.get("Origin") and self.headers["Origin"] != f"http://{host}"):
            return self.respond({"error": "Only same-origin local requests are allowed."}, 403)
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        if self.command == "GET" and url.path in ("/", "/app.js", "/style.css", "/favicon.svg"):
            filename = "index.html" if url.path == "/" else parts[0]
            types = {"html": "text/html; charset=utf-8", "js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8", "svg": "image/svg+xml"}
            return self.respond((WEB / filename).read_bytes(), content_type=types[filename.split('.')[-1]])
        if url.path == "/api/status" and self.command == "GET":
            return self.respond(model_info())
        if url.path == '/api/dashboard' and self.command == 'GET':
            with LOCK:
                return self.respond(web_store.dashboard(JOBS))
        if url.path == '/api/profile':
            with LOCK:
                if self.command == 'GET':
                    return self.respond(web_store.profile(JOBS))
                if self.command == 'POST':
                    return self.respond(web_store.update_profile(JOBS, json.loads(self.body())))
        if url.path == "/api/jobs":
            with LOCK:
                if self.command == "GET":
                    jobs = [{k: v for k, v in job.items() if k not in ("results", "calibration")} for job in STATE.values()]
                    return self.respond(sorted(jobs, key=lambda j: j["created"], reverse=True))
                if self.command == "POST":
                    if sum(j["status"] in ("uploading", "queued", "running") for j in STATE.values()) >= 3:
                        raise ValueError("Three inspections are already active. Finish or delete one first.")
                    data = json.loads(self.body())
                    cfg = settings(data)
                    title = str(data.get('title', '')).strip()[:120]
                    job_id = uuid.uuid4().hex
                    job = dict(id=job_id, created=now(), status="uploading", stage="Uploading images", settings=cfg,
                               sources=[], results=[], errors=[], done=0, total=0, bytes=0, model=MODEL.name,
                               title=title, profile=web_store.profile(JOBS), user_id=1)
                    STATE[job_id] = job
                    save(job)
                    return self.respond(job, 201)
        if len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
            job_id = parts[2]
            folder = job_dir(job_id)
            with LOCK:
                if job_id not in STATE:
                    return self.respond({"error": "Inspection not found."}, 404)
                job = STATE[job_id]
                action = parts[3] if len(parts) > 3 else ""
                if self.command == "GET" and not action:
                    return self.respond(job)
                if self.command == "DELETE" and not action:
                    if job["status"] in ("running", "queued"):
                        raise ValueError("Stop the inspection before deleting its files.")
                    # folder is constrained to JOBS by the strict UUID validator.
                    shutil.rmtree(folder)
                    web_store.delete_job(JOBS, job_id)
                    del STATE[job_id]
                    return self.respond({"deleted": True})
                if self.command == "POST" and action == "cancel":
                    job["cancel_requested"] = True
                    if job["status"] in ("queued", "uploading"):
                        job.update(status="cancelled", stage="Inspection stopped")
                    save(job)
                    return self.respond({"ok": True})
                if self.command == "POST" and action == "retry-rejected":
                    if job["status"] not in ("completed", "cancelled", "failed", "interrupted"):
                        raise ValueError("Wait for the inspection to stop before retrying rejected cells.")
                    if not any(r.get("prediction") == "REJECTED" for r in job["results"]):
                        raise ValueError("This inspection has no rejected cells to retry.")
                    if not model_info()["ready"]:
                        raise ValueError("The model or calibration file is missing.")
                    job.update(status="queued", stage="Waiting to process rejected cells", cancel_requested=False)
                    save(job)
                    POOL.submit(retry_rejected, job_id)
                    return self.respond({"ok": True})
                if self.command == "POST" and action == "start":
                    if job["status"] != "uploading" or not job["sources"]:
                        raise ValueError("Upload at least one image before starting.")
                    if not model_info()["ready"]:
                        raise ValueError("The 10-class model or calibration file is missing.")
                    job.update(status="queued", stage="Waiting for inspection worker")
                    save(job)
                    POOL.submit(process, job_id)
                    return self.respond({"ok": True})
                if self.command == "PUT" and action == "files":
                    if job["status"] != "uploading":
                        raise ValueError("This inspection has already started.")
                    cfg = job["settings"]
                    per_image = cfg["rows"] * cfg["cols"] if cfg["mode"] == "panel" else 1
                    if len(job["sources"]) >= MAX_FILES or job["total"] + per_image > MAX_CELLS:
                        raise ValueError("Batch exceeds 50 images or 2,000 cells.")
                    data = self.body(MAX_FILE)
                    if job["bytes"] + len(data) > MAX_BATCH:
                        raise ValueError("Batch exceeds 100 MB.")
                    name = parse_qs(url.query).get("name", ["image"])[0]
                    name = name.replace("\\", "/").split("/")[-1][:120]
                    name = "".join(c for c in name if ord(c) >= 32) or "image"
                    with Image.open(io.BytesIO(data)) as source:
                        if source.format not in ("PNG", "JPEG", "BMP", "TIFF"):
                            raise ValueError("Use PNG, JPEG, BMP, or TIFF images.")
                        w, h = source.size
                        if w * h > 20_000_000 or max(w, h) > 10000 or min(w, h) < 16:
                            raise ValueError("Images must be at least 16 pixels per side, at most 10,000 per side and 20 megapixels.")
                        img = ImageOps.exif_transpose(source).convert("RGB")
                        w, h = img.size
                        cell_w = w * (1 - 2 * cfg["margin"] / 100) / cfg["cols"] * (1 - cfg["gap"] / 100)
                        cell_h = h * (1 - 2 * cfg["margin"] / 100) / cfg["rows"] * (1 - cfg["gap"] / 100)
                        if "margins" in cfg:
                            m = cfg["margins"]
                            cell_w = (w-m["left"]-m["right"])/cfg["cols"]*(1-cfg["gap"]/100)
                            cell_h = (h-m["top"]-m["bottom"])/cfg["rows"]*(1-cfg["gap"]/100)
                        if cfg["mode"] == "panel" and min(cell_w, cell_h) < 16:
                            raise ValueError("This grid produces cells smaller than 16 pixels. Reduce the grid size or use a larger image.")
                        source_id = uuid.uuid4().hex[:12]
                        relative = f"uploads/{source_id}.png"
                        (folder / "uploads").mkdir(exist_ok=True)
                        img.save(folder / relative)
                    job["sources"].append(dict(id=source_id, name=name, path=relative, width=w, height=h))
                    job["bytes"] += len(data)
                    job["total"] += per_image
                    save(job)
                    return self.respond({"ok": True}, 201)
                if self.command == "GET" and action == "asset":
                    relative = parse_qs(url.query).get("path", [""])[0]
                    path = (folder / relative).resolve()
                    if not path.is_relative_to(folder.resolve()) or path.suffix.lower() not in (".png", ".jpg") or not path.is_file():
                        return self.respond({"error": "Image not found."}, 404)
                    return self.respond(path.read_bytes(), content_type="image/png" if path.suffix == ".png" else "image/jpeg")
                if self.command == "GET" and action == "export":
                    fmt = parse_qs(url.query).get("format", ["json"])[0]
                    if fmt not in ('pdf','csv','json'):
                        raise ValueError('Choose PDF, CSV, or JSON export.')
                    if fmt == 'pdf':
                        if job['status'] not in ('completed','cancelled','failed','interrupted') or not job['results']:
                            raise ValueError('Wait for inspection results before downloading a PDF.')
                        from scripts.inspection_pdf import report_pdf
                        pdf = report_pdf(job, folder)
                        web_store.record_event(JOBS, job_id, 'report_downloaded', 'PDF inspection report generated')
                        return self.respond(pdf, content_type='application/pdf', filename=f"solar-inspection-{job_id[:8]}.pdf")
                    if fmt == "json":
                        return self.respond(json.dumps(job, indent=2).encode(), filename=f"inspection-{job_id[:8]}.json")
                    fields = ["source", "row", "col", "prediction", "likely_class", "confidence", "dsi", "dsi_level", "risk_tier", "ood_reason", "ood_override", "original_ood_reason", "override_at", "error"]
                    output = io.StringIO(newline="")
                    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    for result in job["results"]:
                        safe = {k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v) for k, v in result.items()}
                        writer.writerow(safe)
                    return self.respond(output.getvalue().encode("utf-8-sig"), content_type="text/csv; charset=utf-8", filename=f"inspection-{job_id[:8]}.csv")
        self.respond({"error": "Not found."}, 404)

    def handle_request(self):
        try:
            self.dispatch()
        except (ValueError, KeyError, OSError) as exc:
            self.respond({"error": str(exc)}, 400)
        except Exception:
            logging.exception("Request failed")
            self.respond({"error": "Unexpected server error. Check the terminal log."}, 500)

    do_GET = handle_request
    do_POST = handle_request
    do_PUT = handle_request
    do_DELETE = handle_request

    def log_message(self, fmt, *args):
        logging.debug(fmt, *args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local EL inspection web beta")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    restore()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Solar Inspection: http://127.0.0.1:{args.port}\nKeep this terminal open. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with LOCK:
            for job in STATE.values():
                if job["status"] in ("queued", "running"):
                    job["cancel_requested"] = True
    finally:
        server.server_close()
        POOL.shutdown(wait=True, cancel_futures=True)
