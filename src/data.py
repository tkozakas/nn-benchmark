"""Data loading, transforms, and augmentation utilities."""

import os
from pathlib import Path

import kagglehub
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    VAL_SPLIT, DATASET_NAME, DATASET_SUBDIR,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
)


def get_transforms(augment=True):
    """Get transforms for TinyImageNet (64x64 images)."""
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomCrop(IMAGE_SIZE, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_num_classes(dataset):
    """Get number of classes from dataset."""
    if hasattr(dataset, 'classes'):
        return len(dataset.classes)
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
        return len(dataset.dataset.classes)
    raise ValueError("Cannot determine number of classes from dataset")


def get_subsample(dataset, subsample_size):
    """Optionally subsample the dataset."""
    if subsample_size:
        dataset, _ = torch.utils.data.random_split(
            dataset, [subsample_size, len(dataset) - subsample_size]
        )
    return dataset


def load_dataset(transform=None):
    """Load TinyImageNet dataset, downloading if necessary."""
    project_root = Path(__file__).parent.parent.resolve()
    local_base = project_root / 'dataset'

    if os.path.isdir(local_base):
        path = local_base
        print(f"Using existing dataset cache: {path}")
    else:
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"Downloaded dataset to: {path}")

    if transform is None:
        transform = get_transforms()

    data_root = os.path.join(path, DATASET_SUBDIR)
    return datasets.ImageFolder(root=data_root, transform=transform)


def create_data_loaders(dataset, train_idx, val_idx, batch_size, num_workers, include_test=False):
    """Create train/val (and optionally test) DataLoaders from indices."""
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    use_pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_subset, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_subset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    if include_test:
        return train_loader, val_loader, val_loader  # test = val for k-fold
    return train_loader, val_loader


def create_fold_loaders(dataset, train_idx, test_idx, batch_size, num_workers):
    """Create train/val/test DataLoaders for k-fold CV (splits train into train+val)."""
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    val_size = int(len(train_subset) * VAL_SPLIT)
    train_size = len(train_subset) - val_size
    train_data, val_data = torch.utils.data.random_split(
        train_subset, [train_size, val_size]
    )

    use_pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_subset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    return train_loader, val_loader, test_loader


# Mixup/CutMix augmentation functions
def mixup_data(x, y, alpha=0.2):
    """Apply Mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]

    w, h = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return x_cut, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for Mixup/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


AUGMENT_MAP = {
    'none': None,
    'mixup': mixup_data,
    'cutmix': cutmix_data,
}
