"""Report whether each class has enough unique data for model development."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

DATASET_DIR = Path("dataset")
COLLECTION_DIR = DATASET_DIR / "collection_needed"
RESULTS_DIR = Path("results/data_readiness")
MIN_TRAIN = 50
MIN_VAL = 10
MIN_TEST = 20

log = logging.getLogger(__name__)


def _status(train: int, val: int, test: int) -> tuple[str, str]:
    if train < 10:
        return "BLOCKED", "Too few unique training images; do not rely on this class."
    if train < MIN_TRAIN or val < MIN_VAL or test < MIN_TEST:
        return "NEEDS_DATA", "Collect more unique images before production evaluation."
    return "READY_FOR_EXPERIMENT", "Enough data for experimentation, not certification."


def assess_data_readiness(dataset_dir: Path | None = None) -> dict:
    root = Path(dataset_dir) if dataset_dir else DATASET_DIR
    split_dirs = {split: root / split for split in ("train", "val", "test")}
    for split, path in split_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {path}")

    classes = sorted(path.name for path in split_dirs["train"].iterdir() if path.is_dir())
    rows = []
    for class_name in classes:
        counts = {
            split: sum(1 for path in (directory / class_name).glob("*") if path.is_file())
            for split, directory in split_dirs.items()
        }
        status, action = _status(counts["train"], counts["val"], counts["test"])
        rows.append({
            "class": class_name,
            **counts,
            "train_needed": max(0, MIN_TRAIN - counts["train"]),
            "val_needed": max(0, MIN_VAL - counts["val"]),
            "test_needed": max(0, MIN_TEST - counts["test"]),
            "status": status,
            "action": action,
        })
        if status != "READY_FOR_EXPERIMENT":
            (COLLECTION_DIR / class_name).mkdir(parents=True, exist_ok=True)

    summary = {
        "minimum_targets": {"train": MIN_TRAIN, "validation": MIN_VAL, "test": MIN_TEST},
        "classes": rows,
        "blocked_classes": [row["class"] for row in rows if row["status"] == "BLOCKED"],
        "classes_needing_data": [
            row["class"] for row in rows if row["status"] != "READY_FOR_EXPERIMENT"
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"data_readiness_{timestamp}.json"
    csv_path = RESULTS_DIR / f"data_readiness_{timestamp}.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        log.info(
            "%-24s train=%-4d val=%-3d test=%-3d %s",
            row["class"], row["train"], row["val"], row["test"], row["status"],
        )
    log.info("Collection folders: %s", COLLECTION_DIR)
    log.info("Readiness reports: %s", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    assess_data_readiness()
