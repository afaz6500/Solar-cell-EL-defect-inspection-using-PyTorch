"""Validate and freeze an independently sourced external test set."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from scripts.config import CLASSES, DATASET_DIR, TEST_DIR, TRAIN_DIR, VAL_DIR

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXTERNAL_TEST_DIR = DATASET_DIR / "external_test"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if directory.is_dir() else []


def assess_external_test(
    external_dir: Path = EXTERNAL_TEST_DIR,
    minimum_per_class: int = 30,
) -> dict:
    external_dir = Path(external_dir)
    counts = {
        class_name: len(_images(external_dir / class_name))
        for class_name in CLASSES
    }
    unexpected_classes = sorted(
        path.name for path in external_dir.iterdir()
        if path.is_dir() and path.name not in CLASSES
    ) if external_dir.is_dir() else []

    internal_hashes = {}
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        for path in _images(split_dir):
            internal_hashes.setdefault(_sha256(path), str(path))

    unreadable = []
    overlaps = []
    records = []
    for class_name in CLASSES:
        for path in _images(external_dir / class_name):
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                unreadable.append({"path": str(path), "error": str(exc)})
                continue
            digest = _sha256(path)
            if digest in internal_hashes:
                overlaps.append({
                    "external": str(path),
                    "internal": internal_hashes[digest],
                })
            records.append({
                "path": path.relative_to(external_dir).as_posix(),
                "class": class_name,
                "sha256": digest,
            })

    deficient = {
        class_name: minimum_per_class - count
        for class_name, count in counts.items()
        if count < minimum_per_class
    }
    ready = not deficient and not unexpected_classes and not unreadable and not overlaps
    return {
        "external_dir": str(external_dir),
        "minimum_per_class": minimum_per_class,
        "ready_to_freeze": ready,
        "total_images": sum(counts.values()),
        "class_counts": counts,
        "images_needed": deficient,
        "unexpected_classes": unexpected_classes,
        "unreadable_images": unreadable,
        "internal_exact_overlaps": overlaps,
        "records": records,
    }


def freeze_external_test(
    external_dir: Path = EXTERNAL_TEST_DIR,
    minimum_per_class: int = 30,
) -> Path:
    report = assess_external_test(external_dir, minimum_per_class)
    if not report["ready_to_freeze"]:
        raise ValueError(
            "External test set is not ready to freeze. "
            "Run external-readiness and resolve the reported issues."
        )

    fingerprint = hashlib.sha256()
    for record in report["records"]:
        fingerprint.update(record["path"].encode("utf-8"))
        fingerprint.update(bytes.fromhex(record["sha256"]))
    report["dataset_sha256"] = fingerprint.hexdigest()
    report["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = Path(external_dir) / "external_test_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return manifest_path
