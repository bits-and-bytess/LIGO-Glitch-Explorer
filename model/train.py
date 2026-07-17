"""
Fine-tune EfficientNet-B0 on the Gravity Spy dataset.

Usage:
    python model/train.py --data data/processed --epochs 20 \
        --out model/weights/efficientnet_gravityspy.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from model.dataset import get_dataloaders
from model.model import GlitchClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data/processed")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="model/weights/efficientnet_gravityspy.pt")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=3,
                     help="train only the classifier head for this many epochs first")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data, batch_size=args.batch_size
    )
    print(f"Classes ({len(classes)}): {classes}")

    model = GlitchClassifier(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        # Simple two-phase schedule: freeze backbone briefly so the new
        # classifier head doesn't get swamped by noisy early gradients,
        # then unfreeze for full fine-tuning.
        for p in model.features.parameters():
            p.requires_grad = epoch >= args.freeze_backbone_epochs

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(f"epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)")
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
                             val_loss=val_loss, val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "classes": classes, "val_acc": val_acc},
                args.out,
            )
            print(f"  -> saved new best checkpoint ({val_acc:.4f}) to {args.out}")

    # Final test-set report using the best checkpoint
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=classes, zero_division=0))

    with open(Path(args.out).with_suffix(".history.json"), "w") as f:
        json.dump({"history": history, "test_acc": test_acc,
                   "confusion_matrix": confusion_matrix(labels, preds).tolist(),
                   "classes": classes}, f, indent=2)


if __name__ == "__main__":
    main()
