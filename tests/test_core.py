import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

import main
from scripts import config
from scripts.confidence_calibration import (
    expected_calibration_error,
    load_calibration,
)
from scripts.data_readiness import _status
from scripts.evaluate import _calculate_metrics
from scripts.external_test import assess_external_test
from scripts.ood_detector import _check_prediction, validate_image
from scripts.panel_splitter import split_panel_image
from scripts.predict import compute_dsi, dsi_to_level
from scripts.preprocess import enhance_image, infer_class
from scripts.version_release import dataset_fingerprint
from scripts.analyse import (
    NORMAL_DEFECT_AREA_MAX,
    crack_skeleton_mask,
    structural_anomaly_mask,
)


class PreprocessTests(unittest.TestCase):
    def test_excluded_classes_are_not_relabelled(self):
        for name in config.EXCLUDED_CLASSES:
            self.assertIsNone(infer_class(Path("root") / name / "crack_1.png"))
        self.assertIsNone(infer_class(Path("root/misc/ver_dislocation_1.png")))
        self.assertIsNone(infer_class(Path("root/misc/print_error_1.png")))

    def test_legacy_checkpoint_classes_are_rejected(self):
        config.validate_checkpoint_classes(sorted(config.CLASSES))
        with self.assertRaisesRegex(ValueError, "Train a new model"):
            config.validate_checkpoint_classes(config.CLASSES + list(config.EXCLUDED_CLASSES))
        with self.assertRaises(ValueError):
            config.validate_checkpoint_classes(None)

    def test_infer_class_from_parent_and_alias(self):
        self.assertEqual(infer_class(Path("root/crack/image.jpg")), "crack")
        self.assertEqual(infer_class(Path("root/misc/short-circuit_1.jpg")), "short_circuit")
        self.assertIsNone(infer_class(Path("root/misc/unknown.jpg")))

    def test_enhance_image_shape_and_type(self):
        image = np.arange(120 * 80, dtype=np.uint8).reshape(120, 80)
        enhanced = enhance_image(image)
        self.assertEqual(enhanced.shape, (224, 224))
        self.assertEqual(enhanced.dtype, np.uint8)


class PredictionTests(unittest.TestCase):
    def test_dsi_weighting_and_levels(self):
        self.assertEqual(compute_dsi({"normal": 1.0}), 0.0)
        self.assertEqual(compute_dsi({"short_circuit": 1.0}), 100.0)
        self.assertEqual(compute_dsi({"normal": 0.5, "crack": 0.5}), 33.0)
        self.assertEqual(dsi_to_level(40.0)[0], "Moderate Damage")

    def test_normal_structural_limit_is_conservative(self):
        self.assertEqual(NORMAL_DEFECT_AREA_MAX, 18.5)

    def test_structural_mask_finds_dark_blob(self):
        image = np.full((224, 224), 180, dtype=np.uint8)
        cv2.circle(image, (112, 112), 35, 40, -1)
        mask = structural_anomaly_mask(image)
        self.assertGreater(np.count_nonzero(mask), 500)
        self.assertGreater(mask[112, 112], 0)

    def test_crack_skeleton_finds_thin_dark_line(self):
        image = np.full((224, 224), 180, dtype=np.uint8)
        cv2.line(image, (45, 40), (175, 185), 35, 2)
        skeleton = crack_skeleton_mask(image)
        self.assertGreater(np.count_nonzero(skeleton), 40)

    def test_prediction_thresholds(self):
        valid = _check_prediction({"a": 0.8, "b": 0.2}, 0.5, 1.0)
        uncertain = _check_prediction({"a": 0.4, "b": 0.35, "c": 0.25}, 0.5, 2.0)
        self.assertTrue(valid[0])
        self.assertFalse(uncertain[0])
        self.assertEqual(uncertain[4], 3)


class OODTests(unittest.TestCase):
    def test_blank_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            cv2.imwrite(str(path), np.full((64, 64), 128, dtype=np.uint8))
            result = validate_image(path)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.layer_failed, 1)


class PanelSplitterTests(unittest.TestCase):
    def test_split_panel_grid_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "panel.png"
            Image.new("L", (100, 60), color=128).save(source)
            cells = split_panel_image(source, root / "out", rows=2, cols=5)
            self.assertEqual(len(cells), 10)
            self.assertEqual(cells[0].row, 1)
            self.assertEqual(cells[-1].col, 5)
            self.assertTrue(all(cell.path.is_file() for cell in cells))
            with self.assertRaises(ValueError):
                split_panel_image(source, root / "bad", rows=0, cols=5)


class MetricsTests(unittest.TestCase):
    def test_per_class_metrics(self):
        confusion = np.array([[8, 2], [1, 9]])
        metrics = _calculate_metrics(confusion, ["a", "b"])
        self.assertEqual(metrics[0]["support"], 10)
        self.assertEqual(metrics[0]["precision"], 0.8889)
        self.assertEqual(metrics[0]["recall"], 0.8)

    def test_expected_calibration_error(self):
        probabilities = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        labels = torch.tensor([0, 1])
        error = expected_calibration_error(probabilities, labels, bins=10)
        self.assertAlmostEqual(error, 0.15, places=5)


class ReadinessTests(unittest.TestCase):
    def test_readiness_statuses(self):
        self.assertEqual(_status(5, 1, 1)[0], "BLOCKED")
        self.assertEqual(_status(30, 5, 5)[0], "NEEDS_DATA")
        self.assertEqual(_status(50, 10, 20)[0], "READY_FOR_EXPERIMENT")


class VersionReleaseTests(unittest.TestCase):
    def test_dataset_fingerprint_is_stable_and_counts_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "train" / "normal"
            image_dir.mkdir(parents=True)
            cv2.imwrite(str(image_dir / "sample.png"), np.full((8, 8), 128, np.uint8))
            first = dataset_fingerprint({"train": root / "train"})
            second = dataset_fingerprint({"train": root / "train"})
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["total_images"], 1)
        self.assertEqual(first["splits"]["train"]["classes"]["normal"], 1)


class ExternalTestTests(unittest.TestCase):
    def test_empty_external_set_reports_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            report = assess_external_test(Path(directory), minimum_per_class=2)
        self.assertFalse(report["ready_to_freeze"])
        self.assertEqual(report["total_images"], 0)
        self.assertTrue(all(value == 2 for value in report["images_needed"].values()))


class CalibrationTests(unittest.TestCase):
    def test_missing_calibration_uses_defaults(self):
        with patch(
            "scripts.confidence_calibration.CALIBRATION_PATH",
            Path("missing-calibration-file.json"),
        ):
            calibration = load_calibration(Path("missing-model.pth"))
        self.assertEqual(calibration["temperature"], 1.0)
        self.assertEqual(calibration["min_confidence"], 0.35)

    def test_matching_calibration_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pth"
            model.write_bytes(b"checkpoint")
            from scripts.confidence_calibration import checkpoint_sha256

            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps({
                    "checkpoint_sha256": checkpoint_sha256(model),
                    "temperature": 1.5,
                    "min_confidence": 0.6,
                    "max_entropy": 1.4,
                }),
                encoding="utf-8",
            )
            with patch(
                "scripts.confidence_calibration.CALIBRATION_PATH",
                calibration_path,
            ):
                calibration = load_calibration(model)
        self.assertEqual(calibration["temperature"], 1.5)


class CLITests(unittest.TestCase):
    def test_key_commands_parse(self):
        parser = main.build_parser()
        for command in (
            "predict",
            "evaluate",
            "audit-dataset",
            "data-readiness",
            "calibrate-confidence",
            "external-readiness",
            "version-release",
        ):
            self.assertEqual(parser.parse_args([command]).command, command)


class ConfigTests(unittest.TestCase):
    def test_shared_configuration_is_consistent(self):
        self.assertEqual(config.NUM_CLASSES, len(config.CLASSES))
        self.assertEqual(set(config.CLASSES), set(config.SEVERITY_WEIGHTS))
        self.assertEqual(config.MODEL_PATH.parent, config.MODELS_DIR)


if __name__ == "__main__":
    unittest.main()
