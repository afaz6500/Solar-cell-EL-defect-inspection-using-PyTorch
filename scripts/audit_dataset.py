"""Audit dataset splits for duplicate images and data leakage."""

import csv
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DATASET_DIR = Path("dataset")
SPLITS = ("train", "val", "test")
RESULTS_DIR = Path("results/dataset_audit")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

log = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path, width: int = 16, height: int = 16) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((width + 1, height), Image.Resampling.LANCZOS)
        pixels = list(grayscale.getdata())
    value = 0
    row_width = width + 1
    for row in range(height):
        offset = row * row_width
        for col in range(width):
            value = (value << 1) | (pixels[offset + col] > pixels[offset + col + 1])
    return value


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _cross_split_pairs(records: list[dict]) -> list[dict]:
    pairs = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            if left["split"] == right["split"]:
                continue
            pairs.append({
                "left": left["path"],
                "left_split": left["split"],
                "left_class": left["class"],
                "right": right["path"],
                "right_split": right["split"],
                "right_class": right["class"],
            })
    return pairs


def audit_dataset(dataset_dir: Path | None = None, near_distance: int = 2) -> dict:
    root = Path(dataset_dir) if dataset_dir else DATASET_DIR
    records = []
    unreadable = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")
        for path in sorted(split_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                records.append({
                    "path_obj": path,
                    "path": str(path),
                    "split": split,
                    "class": path.parent.name,
                    "name": path.name.lower(),
                })

    exact_groups: dict[str, list[dict]] = defaultdict(list)
    name_groups: dict[str, list[dict]] = defaultdict(list)
    perceptual_records = []
    for record in tqdm(records, desc="Hashing images"):
        try:
            record["sha256"] = _file_hash(record["path_obj"])
            record["dhash"] = _difference_hash(record["path_obj"])
        except (OSError, UnidentifiedImageError) as exc:
            unreadable.append({"path": record["path"], "error": str(exc)})
            continue
        exact_groups[record["sha256"]].append(record)
        name_groups[record["name"]].append(record)
        perceptual_records.append(record)

    exact_leaks = []
    for group in exact_groups.values():
        exact_leaks.extend(_cross_split_pairs(group))

    same_name_leaks = []
    for group in name_groups.values():
        same_name_leaks.extend(_cross_split_pairs(group))

    # Compare only cross-split records. A 256-bit dHash distance <= 2 is a
    # conservative near-duplicate threshold.
    near_leaks = []
    for left_index, left in enumerate(tqdm(perceptual_records, desc="Comparing hashes")):
        for right in perceptual_records[left_index + 1:]:
            if left["split"] == right["split"] or left["sha256"] == right["sha256"]:
                continue
            distance = _hamming_distance(left["dhash"], right["dhash"])
            if distance <= near_distance:
                near_leaks.append({
                    "left": left["path"],
                    "left_split": left["split"],
                    "left_class": left["class"],
                    "right": right["path"],
                    "right_split": right["split"],
                    "right_class": right["class"],
                    "distance": distance,
                })

    within_split_duplicates = []
    for group in exact_groups.values():
        grouped_by_split: dict[str, list[dict]] = defaultdict(list)
        for record in group:
            grouped_by_split[record["split"]].append(record)
        for split, split_records in grouped_by_split.items():
            if len(split_records) > 1:
                within_split_duplicates.append({
                    "split": split,
                    "files": [record["path"] for record in split_records],
                })

    summary = {
        "dataset": str(root),
        "images_scanned": len(records),
        "unreadable_images": unreadable,
        "exact_cross_split_leaks": exact_leaks,
        "near_cross_split_leaks": near_leaks,
        "same_name_cross_split_pairs": same_name_leaks,
        "within_split_exact_duplicate_groups": within_split_duplicates,
        "counts": {
            "unreadable": len(unreadable),
            "exact_cross_split": len(exact_leaks),
            "near_cross_split": len(near_leaks),
            "same_name_cross_split": len(same_name_leaks),
            "within_split_exact_groups": len(within_split_duplicates),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"dataset_audit_{timestamp}.json"
    csv_path = RESULTS_DIR / f"cross_split_leaks_{timestamp}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = [
        {"type": "exact", **pair, "distance": 0}
        for pair in exact_leaks
    ] + [
        {"type": "near", **pair}
        for pair in near_leaks
    ]
    fieldnames = [
        "type", "left", "left_split", "left_class",
        "right", "right_split", "right_class", "distance",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Images scanned: %d", len(records))
    log.info("Exact cross-split duplicate pairs: %d", len(exact_leaks))
    log.info("Near cross-split duplicate pairs: %d", len(near_leaks))
    log.info("Same-name cross-split pairs: %d", len(same_name_leaks))
    log.info("Within-split exact duplicate groups: %d", len(within_split_duplicates))
    log.info("Unreadable images: %d", len(unreadable))
    log.info("Audit results saved to %s", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    audit_dataset()
