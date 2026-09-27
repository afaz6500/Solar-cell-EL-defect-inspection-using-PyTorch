"""
main.py — Solar Panel EL Defect Detection · Unified Entry Point

Usage examples
──────────────
  python main.py preprocess               # prepare dataset
  python main.py skeleton                 # extract crack skeletons
  python main.py train                    # train CNN (target > 80 % acc)
  python main.py predict                  # batch predict  (img_for_prediction/)
  python main.py predict --src /my/imgs   # batch predict  (custom folder)
  python main.py analyse                  # deep analysis  (img_for_analysis/)
  python main.py analyse --src /my/imgs   # deep analysis  (custom folder)
  python main.py pipeline                 # run full pipeline end-to-end

Global flags
────────────
  --model  PATH   override model checkpoint  (default: models/solar_defect_cnn_10class.pth)
  --src    PATH   override image source dir  (predict / analyse / skeleton)
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


def _configure_console_encoding():
    """Use UTF-8 for Unicode report/summary characters on Windows consoles."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BANNER = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║   ☀️  Solar Panel EL Defect Detection System            ║
  ║      EfficientNet-B0 · DSI Scale · Grad-CAM Analysis     ║
  ╚══════════════════════════════════════════════════════════╝
"""

# ─── Command handlers ──────────────────────────────────────────────────────────

def cmd_preprocess(args):
    """Preprocess raw EL images → transformed + train/val split."""
    log.info("▶  STEP: Preprocessing")
    from scripts.preprocess import run_preprocessing
    run_preprocessing()


def cmd_skeleton(args):
    """Extract morphological crack skeletons from transformed images."""
    log.info("▶  STEP: Skeleton Extraction")
    from scripts.skeleton import run_skeleton_extraction
    src = Path(args.src) if args.src else None
    run_skeleton_extraction(src_dir=src)


def cmd_train(args):
    """Fine-tune EfficientNet-B0 on the prepared dataset."""
    log.info("▶  STEP: Training")
    from scripts.train import train
    train()


def cmd_predict(args):
    """Batch-predict solar panel images and output DSI scores."""
    log.info("▶  STEP: Batch Prediction")
    from scripts.predict import run_batch_prediction
    src   = Path(args.src)   if args.src   else None
    model = Path(args.model) if args.model else None
    results = run_batch_prediction(
        src_dir=src,
        model_path=model,
        split_panel=args.split_panel,
        panel_rows=args.panel_rows,
        panel_cols=args.panel_cols,
        panel_margin_pct=args.panel_margin_pct,
        panel_gap_pct=args.panel_gap_pct,
    )
    _print_prediction_summary(results)


def cmd_analyse(args):
    """Deep batch analysis: Grad-CAM heatmaps + fleet statistics."""
    log.info("▶  STEP: Deep Analysis")
    from scripts.analyse import run_batch_analysis
    src   = Path(args.src)   if args.src   else None
    model = Path(args.model) if args.model else None
    results = run_batch_analysis(
        src_dir=src,
        model_path=model,
        split_panel=args.split_panel,
        panel_rows=args.panel_rows,
        panel_cols=args.panel_cols,
        panel_margin_pct=args.panel_margin_pct,
        panel_gap_pct=args.panel_gap_pct,
    )
    _print_analysis_summary(results)


def cmd_evaluate(args):
    """Evaluate the model on a labeled test or validation dataset."""
    log.info("▶  STEP: Evaluation")
    from scripts.evaluate import evaluate

    result = evaluate(
        dataset_dir=Path(args.dataset) if args.dataset else None,
        model_path=Path(args.model) if args.model else None,
    )
    print(
        f"\nAccuracy: {result['accuracy'] * 100:.2f}%"
        f"\nMacro precision: {result['macro_precision'] * 100:.2f}%"
        f"\nMacro recall: {result['macro_recall'] * 100:.2f}%"
        f"\nMacro F1: {result['macro_f1'] * 100:.2f}%\n"
    )


def cmd_audit_dataset(args):
    """Check dataset splits for exact and near-duplicate leakage."""
    log.info("▶  STEP: Dataset Audit")
    from scripts.audit_dataset import audit_dataset

    result = audit_dataset(
        dataset_dir=Path(args.dataset) if args.dataset else None,
        near_distance=args.near_distance,
    )
    counts = result["counts"]
    print(
        f"\nImages scanned: {result['images_scanned']}"
        f"\nExact cross-split pairs: {counts['exact_cross_split']}"
        f"\nNear cross-split pairs: {counts['near_cross_split']}"
        f"\nSame-name cross-split pairs: {counts['same_name_cross_split']}"
        f"\nUnreadable images: {counts['unreadable']}\n"
    )


def cmd_data_readiness(args):
    """Report class support and create folders for needed data collection."""
    log.info("▶  STEP: Data Readiness")
    from scripts.data_readiness import assess_data_readiness

    result = assess_data_readiness(
        dataset_dir=Path(args.dataset) if args.dataset else None,
    )
    print(
        f"\nBlocked classes: {', '.join(result['blocked_classes']) or 'none'}"
        f"\nClasses needing data: "
        f"{', '.join(result['classes_needing_data']) or 'none'}\n"
    )


def cmd_calibrate_confidence(args):
    """Calibrate probabilities and uncertainty thresholds on validation data."""
    log.info("▶  STEP: Confidence Calibration")
    from scripts.confidence_calibration import calibrate_confidence

    result = calibrate_confidence(
        dataset_dir=Path(args.dataset) if args.dataset else None,
        model_path=Path(args.model) if args.model else None,
    )
    print(
        f"\nTemperature: {result['temperature']:.3f}"
        f"\nECE: {result['raw_ece']:.4f} -> {result['calibrated_ece']:.4f}"
        f"\nConfidence threshold: {result['min_confidence']:.3f}"
        f"\nEntropy threshold: {result['max_entropy']:.3f}\n"
    )


def cmd_test(args):
    """Run the automated unit and integration test suite."""
    log.info("▶  STEP: Automated Tests")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"Test suite failed with exit code {completed.returncode}")


def cmd_version_release(args):
    """Fingerprint the current model, dataset, calibration, and evaluation."""
    log.info("▶  STEP: Version Release")
    from scripts.version_release import create_release_manifest

    manifest = create_release_manifest(args.name)
    print(f"\nRelease manifest saved to: {manifest}\n")


def cmd_external_readiness(args):
    """Validate coverage and independence of the external test set."""
    log.info("▶  STEP: External Test Readiness")
    from scripts.external_test import assess_external_test, freeze_external_test

    dataset = Path(args.dataset) if args.dataset else Path("dataset/external_test")
    result = assess_external_test(dataset, args.minimum_per_class)
    print(
        f"\nExternal images: {result['total_images']}"
        f"\nReady to freeze: {'yes' if result['ready_to_freeze'] else 'no'}"
        f"\nExact overlaps with internal data: "
        f"{len(result['internal_exact_overlaps'])}"
        f"\nUnreadable images: {len(result['unreadable_images'])}"
    )
    if result["images_needed"]:
        print("\nImages still needed:")
        for class_name, needed in result["images_needed"].items():
            print(f"  {class_name:<24} {needed}")
    if result["unexpected_classes"]:
        print("\nUnexpected class folders: " + ", ".join(result["unexpected_classes"]))
    if args.freeze:
        manifest = freeze_external_test(dataset, args.minimum_per_class)
        print(f"\nFrozen external manifest: {manifest}")
    print()


def cmd_pipeline(args):
    """Run the full pipeline: preprocess → skeleton → train → predict → analyse."""
    steps = [
        ("Preprocess",          cmd_preprocess),
        ("Skeleton Extraction", cmd_skeleton),
        ("Train",               cmd_train),
        ("Batch Predict",       cmd_predict),
        ("Deep Analyse",        cmd_analyse),
    
    ]

    log.info("═" * 60)
    log.info("FULL PIPELINE — %d steps", len(steps))
    log.info("═" * 60)

    pipeline_start = time.time()
    for i, (name, fn) in enumerate(steps, 1):
        log.info("")
        log.info("── Step %d/%d: %s ──", i, len(steps), name)
        t0 = time.time()
        try:
            fn(args)
        except Exception as exc:
            log.error("Step '%s' failed: %s", name, exc)
            log.error("Pipeline aborted. Fix the error above and re-run.")
            sys.exit(1)
        log.info("   Step '%s' completed in %.1fs", name, time.time() - t0)

    log.info("")
    log.info("═" * 60)
    log.info("✓ Full pipeline complete in %.1fs", time.time() - pipeline_start)
    log.info("═" * 60)


# ─── Summary helpers ───────────────────────────────────────────────────────────

def _print_prediction_summary(results: list[dict]):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    from collections import Counter
    counts = Counter(r["prediction"] for r in valid)
    dsis   = [r["dsi"] for r in valid]

    print("\n" + "─" * 50)
    print(f"  PREDICTION SUMMARY  ({len(valid)} images)")
    print("─" * 50)
    for cls, cnt in sorted(counts.items()):
        bar = "█" * int(cnt / len(valid) * 30)
        print(f"  {cls:<20} {cnt:>4}  {bar}")
    print(f"\n  Avg DSI : {sum(dsis)/len(dsis):.1f}")
    print(f"  Max DSI : {max(dsis):.1f}")
    print("─" * 50 + "\n")


def _print_analysis_summary(results: list[dict]):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    from collections import Counter
    tiers = Counter(r["risk_tier"] for r in valid)
    dsis  = [r["dsi"] for r in valid]
    areas = [r["defect_area_pct"] for r in valid]

    print("\n" + "═" * 50)
    print(f"  ANALYSIS SUMMARY  ({len(valid)} panels)")
    print("═" * 50)
    print("  Risk Tier Distribution:")
    for tier, cnt in sorted(tiers.items()):
        bar = "█" * int(cnt / len(valid) * 25)
        print(f"    {tier:<14} {cnt:>4}  {bar}")
    print(f"\n  DSI   avg / max : {sum(dsis)/len(dsis):.1f} / {max(dsis):.1f}")
    print(f"  Defect area avg : {sum(areas)/len(areas):.1f}%")
    print("═" * 50 + "\n")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Solar Panel EL Defect Detection — Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global flags
    parser.add_argument(
        "--model", metavar="PATH", default=None,
        help="Path to model checkpoint (default: models/solar_defect_cnn_10class.pth)",
    )
    parser.add_argument(
        "--src", metavar="PATH", default=None,
        help="Source image directory (overrides default for predict/analyse/skeleton)",
    )
    parser.add_argument(
        "--split-panel", action="store_true",
        help="Split whole-panel images into cells before predict/analyse",
    )
    parser.add_argument(
        "--panel-rows", type=int, default=6,
        help="Panel grid rows used with --split-panel (default: 6)",
    )
    parser.add_argument(
        "--panel-cols", type=int, default=10,
        help="Panel grid columns used with --split-panel (default: 10)",
    )
    parser.add_argument(
        "--panel-margin-pct", type=float, default=0.0,
        help="Percent margin trimmed from panel edges before splitting",
    )
    parser.add_argument(
        "--panel-gap-pct", type=float, default=0.0,
        help="Percent trimmed inside each cell to avoid grid-line borders",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )

    def add_inference_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument(
            "--src", metavar="PATH", default=argparse.SUPPRESS,
            help="Source image directory",
        )
        cmd.add_argument(
            "--model", metavar="PATH", default=argparse.SUPPRESS,
            help="Path to model checkpoint",
        )
        cmd.add_argument(
            "--split-panel", action="store_true", default=argparse.SUPPRESS,
            help="Split whole-panel images into cells before inference",
        )
        cmd.add_argument("--panel-rows", type=int, default=argparse.SUPPRESS, help="Panel grid rows")
        cmd.add_argument("--panel-cols", type=int, default=argparse.SUPPRESS, help="Panel grid columns")
        cmd.add_argument(
            "--panel-margin-pct", type=float, default=argparse.SUPPRESS,
            help="Percent margin trimmed from panel edges before splitting",
        )
        cmd.add_argument(
            "--panel-gap-pct", type=float, default=argparse.SUPPRESS,
            help="Percent trimmed inside each cell to avoid grid-line borders",
        )

    # Sub-commands
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    sub.add_parser("preprocess", help="Preprocess raw EL images into train/val/test splits")
    sub.add_parser("skeleton",   help="Extract morphological crack skeletons")
    sub.add_parser("train",      help="Train the EfficientNet-B0 defect classifier")
    predict_cmd = sub.add_parser("predict", help="Batch-predict images with DSI damage scale")
    add_inference_args(predict_cmd)
    analyse_cmd = sub.add_parser("analyse", help="Deep analysis: Grad-CAM + fleet statistics")
    add_inference_args(analyse_cmd)
    eval_cmd = sub.add_parser("evaluate", help="Evaluate model metrics on labeled images")
    eval_cmd.add_argument(
        "--dataset", metavar="PATH", default=None,
        help="Labeled dataset directory (default: dataset/test, then dataset/val)",
    )
    eval_cmd.add_argument(
        "--model", metavar="PATH", default=argparse.SUPPRESS,
        help="Path to model checkpoint",
    )
    audit_cmd = sub.add_parser("audit-dataset", help="Check dataset splits for duplicate leakage")
    audit_cmd.add_argument(
        "--dataset", metavar="PATH", default=None,
        help="Dataset root containing train, val, and test folders",
    )
    audit_cmd.add_argument(
        "--near-distance", type=int, default=2,
        help="Maximum 256-bit perceptual-hash distance for near duplicates (default: 2)",
    )
    readiness_cmd = sub.add_parser("data-readiness", help="Report underrepresented classes")
    readiness_cmd.add_argument(
        "--dataset", metavar="PATH", default=None,
        help="Dataset root containing train, val, and test folders",
    )
    calibration_cmd = sub.add_parser(
        "calibrate-confidence",
        help="Calibrate confidence and uncertainty thresholds",
    )
    calibration_cmd.add_argument(
        "--dataset", metavar="PATH", default=None,
        help="Validation dataset directory (default: dataset/val)",
    )
    calibration_cmd.add_argument(
        "--model", metavar="PATH", default=argparse.SUPPRESS,
        help="Path to model checkpoint",
    )
    sub.add_parser("test", help="Run automated tests")
    release_cmd = sub.add_parser(
        "version-release",
        help="Create a versioned model and dataset manifest",
    )
    release_cmd.add_argument(
        "--name",
        default=None,
        help="Release name (default: timestamped model release)",
    )
    external_cmd = sub.add_parser(
        "external-readiness",
        help="Validate or freeze the independent external test set",
    )
    external_cmd.add_argument(
        "--dataset",
        default=None,
        help="External ImageFolder root (default: dataset/external_test)",
    )
    external_cmd.add_argument(
        "--minimum-per-class",
        type=int,
        default=30,
        help="Minimum independent images required per class (default: 30)",
    )
    external_cmd.add_argument(
        "--freeze",
        action="store_true",
        help="Write an immutable checksum manifest when all checks pass",
    )
    sub.add_parser("pipeline",   help="Run the complete end-to-end pipeline")

    return parser


COMMAND_MAP = {
    "preprocess" : cmd_preprocess,
    "skeleton"   : cmd_skeleton,
    "train"      : cmd_train,
    "predict"    : cmd_predict,
    "analyse"    : cmd_analyse,
    "evaluate"   : cmd_evaluate,
    "audit-dataset": cmd_audit_dataset,
    "data-readiness": cmd_data_readiness,
    "calibrate-confidence": cmd_calibrate_confidence,
    "test": cmd_test,
    "version-release": cmd_version_release,
    "external-readiness": cmd_external_readiness,
    "pipeline"   : cmd_pipeline,
}


def main():
    print(BANNER)
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    t0 = time.time()
    try:
        handler(args)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)

    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
