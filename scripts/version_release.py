"""Create a traceable manifest for a trained model and its dataset."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.config import (
    CLASSES,
    CONFIDENCE_CALIBRATION_PATH,
    EVALUATION_RESULTS_DIR,
    MODEL_PATH,
    MODELS_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    TRAINING_LOG_PATH,
    VAL_DIR,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(split_dirs: dict[str, Path]) -> dict:
    digest = hashlib.sha256()
    splits = {}
    total = 0
    for split_name, split_dir in split_dirs.items():
        class_counts = {}
        split_total = 0
        for class_name in sorted(CLASSES):
            class_dir = split_dir / class_name
            files = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ) if class_dir.is_dir() else []
            class_counts[class_name] = len(files)
            split_total += len(files)
            for path in files:
                relative = path.relative_to(split_dir.parent).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(bytes.fromhex(sha256_file(path)))
        splits[split_name] = {"total": split_total, "classes": class_counts}
        total += split_total
    return {"sha256": digest.hexdigest(), "total_images": total, "splits": splits}


def _latest_evaluation() -> Path | None:
    reports = sorted(
        EVALUATION_RESULTS_DIR.glob("evaluation_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def create_release_manifest(name: str | None = None) -> Path:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    created = datetime.now(timezone.utc)
    release_name = name or created.strftime("model_%Y%m%d_%H%M%S")
    release_dir = MODELS_DIR / "releases" / release_name
    if release_dir.exists():
        raise FileExistsError(f"Release already exists: {release_dir}")
    release_dir.mkdir(parents=True)

    evaluation_path = _latest_evaluation()
    calibration = (
        json.loads(CONFIDENCE_CALIBRATION_PATH.read_text(encoding="utf-8"))
        if CONFIDENCE_CALIBRATION_PATH.is_file() else None
    )
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path else None
    )

    artifacts = {}
    for label, path in {
        "model": MODEL_PATH,
        "training_log": TRAINING_LOG_PATH,
        "calibration": CONFIDENCE_CALIBRATION_PATH,
        "evaluation": evaluation_path,
        "requirements": Path("requirements.txt"),
    }.items():
        if path and path.is_file():
            artifacts[label] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }

    manifest = {
        "release": release_name,
        "created_utc": created.isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "classes": CLASSES,
        "dataset": dataset_fingerprint({
            "train": TRAIN_DIR,
            "val": VAL_DIR,
            "test": TEST_DIR,
        }),
        "metrics": {
            key: evaluation.get(key)
            for key in (
                "samples", "accuracy", "macro_precision", "macro_recall", "macro_f1"
            )
        } if evaluation else None,
        "calibration": {
            key: calibration.get(key)
            for key in (
                "temperature", "min_confidence", "max_entropy",
                "raw_ece", "calibrated_ece",
            )
        } if calibration else None,
        "artifacts": artifacts,
    }
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
