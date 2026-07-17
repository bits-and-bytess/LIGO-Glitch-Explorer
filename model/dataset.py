"""
Dataset loader for preprocessed Gravity Spy Q-transform images.

Expects a directory structure produced by preprocessing/qtransform.py's
batch mode (see scripts/build_dataset.py), or any directory laid out as:

    data/processed/
        train/<ClassName>/*.png
        val/<ClassName>/*.png
        test/<ClassName>/*.png

This mirrors torchvision.datasets.ImageFolder conventions on purpose so
we get correct-by-construction label indices for free.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model.model import GRAVITY_SPY_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.0),  # NOTE: spectrograms are time/freq oriented;
            # flipping can invert meaning, so keep this off by default. Left here
            # as a documented no-op rather than silently omitted.
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 4):
    root = Path(data_root)
    train_ds = ImageFolder(root / "train", transform=build_transforms(train=True))
    val_ds = ImageFolder(root / "val", transform=build_transforms(train=False))
    test_ds = ImageFolder(root / "test", transform=build_transforms(train=False))

    # Sanity check: class ordering must match model.GRAVITY_SPY_CLASSES so
    # label indices returned by the model line up with human-readable names.
    if list(train_ds.classes) != sorted(GRAVITY_SPY_CLASSES):
        raise ValueError(
            "Class folders under data/processed/train don't match "
            "GRAVITY_SPY_CLASSES in model/model.py. Either update that "
            "list or fix your data layout -- label indices must agree "
            "between training and inference."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.classes
