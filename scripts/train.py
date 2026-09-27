"""
train.py — Solar Panel Defect Detection CNN Trainer

Architecture: EfficientNet-B0 (pretrained ImageNet) fine-tuned for 10-class
classification.  Targets > 90 % validation accuracy with:
  • Aggressive data augmentation
  • Weighted cross-entropy for class imbalance
  • Cosine-annealing LR schedule
  • Early stopping + best-model checkpoint

Classes: normal | black_core | corner | crack | finger | horizontal_dislocation | scratch | short_circuit | star_crack | thick_line
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from scripts.config import (
    BATCH_SIZE, CLASSES, DEVICE, EARLY_STOPPING_PATIENCE, IMG_SIZE,
    LEARNING_RATE, MODEL_PATH, NUM_CLASSES, NUM_EPOCHS, TRAINING_LOG_PATH,
    TRAIN_DIR, VAL_DIR,
)

# ─── Config ────────────────────────────────────────────────────────────────────
LOG_PATH = TRAINING_LOG_PATH
LR = LEARNING_RATE
PATIENCE = EARLY_STOPPING_PATIENCE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # EL images are grayscale
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """EfficientNet-B0 with a custom classification head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze early backbone layers, fine-tune later ones
    for name, param in model.named_parameters():
        if "features.0" in name or "features.1" in name:
            param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ─── Dataset helpers ───────────────────────────────────────────────────────────

def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Over-sample minority classes so each batch is roughly balanced."""
    targets   = [s[1] for s in dataset.samples]
    if not targets:
        raise ValueError(f"No training images found in {TRAIN_DIR}")
    counts    = np.bincount(targets)
    weights   = 1.0 / counts[targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def load_datasets():
    # Fall back gracefully if val dir has same structure as train
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=TRAIN_TF, allow_empty=True)
    val_ds   = datasets.ImageFolder(str(VAL_DIR),   transform=VAL_TF, allow_empty=True)

    log.info("Train classes : %s", train_ds.classes)
    log.info("Val   classes : %s", val_ds.classes)
    log.info("Train samples : %d", len(train_ds))
    log.info("Val   samples : %d", len(val_ds))

    expected_classes = sorted(CLASSES)
    if train_ds.classes != expected_classes:
        raise ValueError(
            "Training class folders do not match the configured class order. "
            f"Expected {expected_classes}, found {train_ds.classes}."
        )
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError("Training and validation class folders do not match.")
    if not val_ds.samples:
        raise ValueError(f"No validation images found in {VAL_DIR}")
    train_counts = np.bincount(
        [target for _, target in train_ds.samples],
        minlength=len(train_ds.classes),
    )
    for index, count in enumerate(train_counts):
        if count < 10:
            log.warning(
                "Class '%s' has only %d unique training image(s); "
                "its predictions are not reliable.",
                train_ds.classes[index],
                count,
            )

    sampler     = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


# ─── Training loop ─────────────────────────────────────────────────────────────

def train(model_path: Path | None = None,
          log_path: Path | None = None,
          num_epochs: int | None = None):
    model_path = Path(model_path) if model_path else MODEL_PATH
    log_path = Path(log_path) if log_path else LOG_PATH
    epochs = int(num_epochs) if num_epochs else NUM_EPOCHS

    model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Device: %s", DEVICE)

    train_loader, val_loader, class_names = load_datasets()

    num_classes = len(class_names)
    model     = build_model(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc    = 0.0
    patience_count  = 0
    history         = []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = train_correct = train_total = 0
        t0 = time.time()

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d} train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = val_correct = val_total = 0
        class_correct = np.zeros(num_classes)
        class_total   = np.zeros(num_classes)

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d} val  ", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs  = model(imgs)
                loss     = criterion(outputs, labels)

                val_loss    += loss.item() * imgs.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

                for c in range(num_classes):
                    mask = (labels == c)
                    class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    class_total[c]   += mask.sum().item()

        epoch_train_acc = train_correct / train_total * 100
        epoch_val_acc   = val_correct   / val_total   * 100
        epoch_train_loss = train_loss   / train_total
        epoch_val_loss   = val_loss     / val_total
        elapsed          = time.time() - t0

        per_class = {
            class_names[i]: f"{class_correct[i]/max(class_total[i],1)*100:.1f}%"
            for i in range(num_classes)
        }

        log.info(
            "Epoch %02d/%02d | train %.2f%% (loss %.4f) | val %.2f%% (loss %.4f) | %.1fs",
            epoch, epochs,
            epoch_train_acc, epoch_train_loss,
            epoch_val_acc,   epoch_val_loss, elapsed,
        )
        log.info("  Per-class val acc: %s", per_class)

        record = {
            "epoch"      : epoch,
            "train_acc"  : round(epoch_train_acc, 4),
            "val_acc"    : round(epoch_val_acc,   4),
            "train_loss" : round(epoch_train_loss, 6),
            "val_loss"   : round(epoch_val_loss,  6),
            "lr"         : scheduler.get_last_lr()[0],
            "per_class"  : per_class,
        }
        history.append(record)

        # ── Checkpoint ──
        if epoch_val_acc > best_val_acc:
            best_val_acc   = epoch_val_acc
            patience_count = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_acc"     : best_val_acc,
                "class_names" : class_names,
            }, model_path)
            log.info("  ✓ Saved best model (val acc %.2f%%)", best_val_acc)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, PATIENCE)
                break

    # Save training log
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info("─" * 60)
    log.info("Training complete. Best val accuracy: %.2f%%", best_val_acc)
    log.info("Model saved to : %s", model_path)
    log.info("Training log   : %s", log_path)

    if best_val_acc < 80.0:
        log.warning(
            "Best val accuracy %.2f%% is below 80%%. Consider:\n"
            "  • Adding more training images per class (≥ 200 recommended)\n"
            "  • Running more epochs (increase NUM_EPOCHS)\n"
            "  • Tuning LR or augmentation settings",
            best_val_acc,
        )

    return history


if __name__ == "__main__":
    train()
