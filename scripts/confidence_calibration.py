"""Calibrate model confidence and uncertainty thresholds on validation data."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scripts.config import validate_checkpoint_classes
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from scripts.predict import MODEL_PATH
from scripts.train import DEVICE, VAL_DIR, VAL_TF, build_model
from scripts.config import BATCH_SIZE, CONFIDENCE_CALIBRATION_PATH

CALIBRATION_PATH = CONFIDENCE_CALIBRATION_PATH
log = logging.getLogger(__name__)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 15,
) -> float:
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    error = torch.tensor(0.0)
    boundaries = torch.linspace(0, 1, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            accuracy = correct[mask].float().mean()
            average_confidence = confidence[mask].mean()
            error += mask.float().mean() * torch.abs(average_confidence - accuracy)
    return float(error)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log2(probabilities.clamp_min(1e-9))).sum(dim=1)


def calibrate_confidence(
    dataset_dir: Path | None = None,
    model_path: Path | None = None,
) -> dict:
    validation_dir = Path(dataset_dir) if dataset_dir else VAL_DIR
    checkpoint_path = Path(model_path) if model_path else MODEL_PATH
    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
    class_names = checkpoint["class_names"]
    validate_checkpoint_classes(class_names)

    dataset = datasets.ImageFolder(
        str(validation_dir),
        transform=VAL_TF,
        allow_empty=True,
    )
    if dataset.classes != class_names:
        raise ValueError("Validation classes do not match checkpoint classes.")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits_parts = []
    label_parts = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting validation logits"):
            logits_parts.append(model(images.to(DEVICE)).cpu())
            label_parts.append(labels)
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp().detach().clamp(0.05, 10.0))
    raw_probabilities = F.softmax(logits, dim=1)
    calibrated_probabilities = F.softmax(logits / temperature, dim=1)

    confidence, predictions = calibrated_probabilities.max(dim=1)
    correct = predictions.eq(labels)
    correct_confidence = confidence[correct].numpy()
    correct_entropy = _entropy(calibrated_probabilities[correct]).numpy()
    min_confidence = float(np.quantile(correct_confidence, 0.05))
    max_entropy = float(np.quantile(correct_entropy, 0.95))

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "validation_dataset": str(validation_dir),
        "validation_samples": len(dataset),
        "temperature": round(temperature, 6),
        "min_confidence": round(min_confidence, 6),
        "max_entropy": round(max_entropy, 6),
        "raw_nll": round(float(F.cross_entropy(logits, labels)), 6),
        "calibrated_nll": round(float(F.cross_entropy(logits / temperature, labels)), 6),
        "raw_ece": round(expected_calibration_error(raw_probabilities, labels), 6),
        "calibrated_ece": round(
            expected_calibration_error(calibrated_probabilities, labels), 6
        ),
    }
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Temperature: %.3f", temperature)
    log.info("ECE: %.4f -> %.4f", result["raw_ece"], result["calibrated_ece"])
    log.info("Confidence threshold: %.3f", min_confidence)
    log.info("Entropy threshold: %.3f", max_entropy)
    log.info("Calibration saved to %s", CALIBRATION_PATH)
    return result


def load_calibration(model_path: Path) -> dict:
    defaults = {"temperature": 1.0, "min_confidence": 0.35, "max_entropy": 2.2}
    if not CALIBRATION_PATH.is_file():
        return defaults
    try:
        calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        if calibration.get("checkpoint_sha256") != checkpoint_sha256(Path(model_path)):
            log.warning("Ignoring calibration because it belongs to a different checkpoint.")
            return defaults
        return {**defaults, **calibration}
    except (OSError, ValueError, KeyError) as exc:
        log.warning("Cannot load confidence calibration: %s", exc)
        return defaults


if __name__ == "__main__":
    calibrate_confidence()
