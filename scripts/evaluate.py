"""Evaluate a trained classifier on a labeled ImageFolder dataset."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.config import validate_checkpoint_classes
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from scripts.predict import MODEL_PATH
from scripts.train import DEVICE, VAL_TF, build_model
from scripts.config import BATCH_SIZE, EVALUATION_RESULTS_DIR, TEST_DIR, VAL_DIR

RESULTS_DIR = EVALUATION_RESULTS_DIR

log = logging.getLogger(__name__)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _calculate_metrics(confusion: np.ndarray, class_names: list[str]) -> list[dict]:
    rows = []
    for index, class_name in enumerate(class_names):
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        rows.append({
            "class": class_name,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def _save_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    size = max(10, len(class_names) * 0.8)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = confusion.max() / 2 if confusion.size else 0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(
                col,
                row,
                str(confusion[row, col]),
                ha="center",
                va="center",
                color="white" if confusion[row, col] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    dataset_dir: Path | None = None,
    model_path: Path | None = None,
) -> dict:
    dataset_path = Path(dataset_dir) if dataset_dir else (
        TEST_DIR if TEST_DIR.exists() else VAL_DIR
    )
    checkpoint_path = Path(model_path) if model_path else MODEL_PATH

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
    class_names = checkpoint.get("class_names")
    validate_checkpoint_classes(class_names)
    if not class_names:
        raise ValueError("Checkpoint does not contain class_names metadata.")

    dataset = datasets.ImageFolder(str(dataset_path), transform=VAL_TF, allow_empty=True)
    if dataset.classes != class_names:
        raise ValueError(
            "Dataset class order does not match the checkpoint. "
            f"Checkpoint: {class_names}; dataset: {dataset.classes}"
        )
    if not dataset.samples:
        raise ValueError(f"No evaluation images found in {dataset_path}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    predictions_log = []
    sample_offset = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            probabilities = F.softmax(model(images.to(DEVICE)), dim=1).cpu()
            confidences, predictions = probabilities.max(dim=1)
            for batch_index, (true_label, predicted_label, confidence) in enumerate(
                zip(labels, predictions, confidences)
            ):
                confusion[int(true_label), int(predicted_label)] += 1
                image_path = dataset.samples[sample_offset + batch_index][0]
                predictions_log.append({
                    "file": str(Path(image_path)),
                    "true_class": class_names[int(true_label)],
                    "predicted_class": class_names[int(predicted_label)],
                    "confidence": round(float(confidence), 4),
                    "correct": int(true_label) == int(predicted_label),
                })
            sample_offset += len(labels)

    per_class = _calculate_metrics(confusion, class_names)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = _safe_divide(correct, total)
    macro_precision = float(np.mean([row["precision"] for row in per_class]))
    macro_recall = float(np.mean([row["recall"] for row in per_class]))
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))

    summary = {
        "dataset": str(dataset_path),
        "model": str(checkpoint_path),
        "samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions_log,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"evaluation_{timestamp}.json"
    csv_path = RESULTS_DIR / f"per_class_{timestamp}.csv"
    predictions_path = RESULTS_DIR / f"predictions_{timestamp}.csv"
    matrix_path = RESULTS_DIR / f"confusion_matrix_{timestamp}.png"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions_log[0].keys())
        writer.writeheader()
        writer.writerows(predictions_log)
    _save_confusion_matrix(confusion, class_names, matrix_path)

    log.info("Accuracy: %.2f%% (%d/%d)", accuracy * 100, correct, total)
    log.info("Macro precision: %.2f%%", macro_precision * 100)
    log.info("Macro recall: %.2f%%", macro_recall * 100)
    log.info("Macro F1: %.2f%%", macro_f1 * 100)
    log.info("Evaluation results saved to %s", RESULTS_DIR)
    return summary


if __name__ == "__main__":
    evaluate()
