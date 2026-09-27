"""Shared project configuration."""

from pathlib import Path

import torch

DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "raw_img_gray"
TRANSFORMED_DIR = DATASET_DIR / "transformed_img"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
LOCKED_TEST_DIR = DATASET_DIR / "test_locked"
CONFLICTS_DIR = DATASET_DIR / "label_conflicts"
SKELETON_DIR = DATASET_DIR / "skeleton_img"

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "solar_defect_cnn_10class.pth"
TRAINING_LOG_PATH = MODELS_DIR / "training_log_10class.json"
CONFIDENCE_CALIBRATION_PATH = MODELS_DIR / "confidence_calibration_10class.json"

RESULTS_DIR = Path("results")
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"

PREDICTION_INPUT_DIR = Path("img_for_prediction")
ANALYSIS_INPUT_DIR = Path("img_for_analysis")

CLASSES = [
    "normal", "black_core", "corner", "crack", "finger",
    "horizontal_dislocation", "scratch", "short_circuit",
    "star_crack", "thick_line",
]
EXCLUDED_CLASSES = ("fragment", "printing_error", "vertical_dislocation")


def validate_checkpoint_classes(class_names):
    """Reject legacy checkpoints before inference or confidence calibration."""
    if not class_names or sorted(class_names) != sorted(CLASSES):
        raise ValueError(
            "Checkpoint classes do not match the active 10-class configuration. "
            "Train a new model with python main.py train, then calibrate confidence."
        )


NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
EARLY_STOPPING_PATIENCE = 8

MIN_CONFIDENCE = 0.35
MAX_ENTROPY = 2.20

SEVERITY_WEIGHTS = {
    "normal": 0.0,
    "black_core": 0.33,
    "corner": 0.33,
    "crack": 0.66,
    "finger": 0.66,
    "horizontal_dislocation": 0.66,
    "scratch": 0.66,
    "short_circuit": 1.0,
    "star_crack": 1.0,
    "thick_line": 0.66,
}
