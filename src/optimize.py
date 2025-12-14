__doc__ = r"""
Usage:
    optimize.py [--architecture=ARCH]
                [--device=DEVICE]
                [--cpu-workers=NUM]
                [--epochs=N]
                [--patience=P]
                [--n-trials=T]

Options:
    -h --help               Show this help message.
    --device=DEVICE         Device to use for training (cpu or cuda) [default: cuda].
    --cpu-workers=NUM       Number of CPU workers for data loading [default: 6].
    --architecture=ARCH     Model architecture [default: DenseNet121].
    --epochs=N              Max epochs per trial             [default: 20].
    --patience=P            Early-stop patience             [default: 5].
    --n-trials=T            Number of Optuna trials         [default: 30].
"""

import os
import time
import warnings

import kagglehub
import numpy as np
import optuna
import pandas as pd
import psutil
import torch
from docopt import docopt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from pathlib import Path

from config import train_config
from model import get_model
from train import get_gpu_usage_percent
from utility import get_transforms
from visualise import plot_optimization_comparison

os.makedirs('../test_data', exist_ok=True)
warnings.filterwarnings("ignore", message=".*GoogleNet.*", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")


def _num_classes(ds):
    if hasattr(ds, 'classes'):
        return len(ds.classes)
    if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'classes'):
        return len(ds.dataset.classes)
    raise ValueError("Cannot determine number of classes from dataset")


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

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x_cut = x.clone()
    x_cut[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x_cut, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for Mixup/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


AUGMENT_MAP = {
    'none': None,
    'mixup': mixup_data,
    'cutmix': cutmix_data,
}


def evaluate(model, loader, criterion, device):
    """Run evaluation and compute loss, accuracy and metrics."""
    model.eval()
    running_loss = correct = total = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    loss = running_loss / total
    acc = correct / total
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return loss, acc, prec, rec, f1


def train_one_epoch(model, loader, criterion, optimizer, device, augment_fn=None):
    """Train model for one epoch with optional Mixup/CutMix augmentation."""
    model.train()
    running_loss = correct = total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if augment_fn is not None:
            imgs_aug, labels_a, labels_b, lam = augment_fn(imgs, labels)
            outputs = model(imgs_aug)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            preds = outputs.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


def get_data_loaders(dataset, train_idx, val_idx, batch_size, num_workers):
    """Split dataset into train/val and return DataLoaders."""
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    loader_args = dict(
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        num_workers=num_workers,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_subset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_args)
    return train_loader, val_loader


def train_single_fold(
    architecture,
    dataset,
    train_idx,
    val_idx,
    pretrained=True,
    augment_fn=None,
    epochs=20,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-4,
    optimizer_name='adam',
    early_stopping_patience=5,
    device='cuda',
    cpu_workers=4,
    verbose=True
):
    """Train model on a single fold and return validation F1."""
    device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    n_classes = _num_classes(dataset)

    train_loader, val_loader = get_data_loaders(
        dataset, train_idx, val_idx, batch_size, cpu_workers
    )

    model = get_model(architecture, num_classes=n_classes, pretrained=pretrained).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Select optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        tloss, tacc = train_one_epoch(model, train_loader, criterion, optimizer, device, augment_fn)
        vloss, vacc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)

        if f1 > best_val_f1:
            best_val_f1 = f1

        if early_stopping_patience is not None:
            if vloss < best_val_loss:
                best_val_loss, patience_cnt = vloss, 0
            else:
                patience_cnt += 1
                if patience_cnt >= early_stopping_patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch}")
                    break

        if verbose:
            print(f"    Epoch {epoch}/{epochs} | Train Loss: {tloss:.3f} | Val F1: {f1:.3f}")

    return best_val_f1


def create_objective(architecture, dataset, epochs, patience, cpu_workers, device):
    """Create Optuna objective function."""
    
    # Create a fixed train/val split for consistent evaluation
    n = len(dataset)
    n_val = int(n * 0.2)
    n_train = n - n_val
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    def objective(trial):
        # Hyperparameters to optimize
        pretrained = trial.suggest_categorical('pretrained', [True, False])
        augmentation = trial.suggest_categorical('augmentation', ['none', 'mixup', 'cutmix'])
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])

        augment_fn = AUGMENT_MAP[augmentation]

        config_str = (f"pretrained={pretrained}, aug={augmentation}, "
                      f"lr={learning_rate:.1e}, bs={batch_size}, wd={weight_decay:.1e}, opt={optimizer_name}")
        print(f"\n  Trial {trial.number}: {config_str}")

        try:
            val_f1 = train_single_fold(
                architecture=architecture,
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                pretrained=pretrained,
                augment_fn=augment_fn,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer_name=optimizer_name,
                early_stopping_patience=patience,
                device=device,
                cpu_workers=cpu_workers,
                verbose=True
            )
        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0

        print(f"  Trial {trial.number} finished with F1: {val_f1:.4f}")
        return val_f1

    return objective


def run_final_evaluation(architecture, dataset, best_params, epochs, patience, cpu_workers, device):
    """Run final evaluation with best parameters using 3-fold CV."""
    print("\n=== Final Evaluation with Best Parameters (3-fold CV) ===")
    print(f"Best params: {best_params}")

    augment_fn = AUGMENT_MAP[best_params['augmentation']]

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
        print(f"\n  Fold {fold_idx}/3")
        f1 = train_single_fold(
            architecture=architecture,
            dataset=dataset,
            train_idx=list(train_idx),
            val_idx=list(val_idx),
            pretrained=best_params['pretrained'],
            augment_fn=augment_fn,
            epochs=epochs,
            batch_size=best_params['batch_size'],
            learning_rate=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            optimizer_name=best_params['optimizer'],
            early_stopping_patience=patience,
            device=device,
            cpu_workers=cpu_workers,
            verbose=True
        )
        fold_f1s.append(f1)
        print(f"  Fold {fold_idx} F1: {f1:.4f}")

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"\n  Final F1: {mean_f1:.4f} ± {std_f1:.4f}")

    return mean_f1, std_f1, fold_f1s


def main():
    project_root = Path(__file__).parent.parent.resolve()
    local_base = project_root / 'dataset'
    if os.path.isdir(local_base):
        path = local_base
        print("Using existing dataset cache:", path)
    else:
        path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
        print("Downloaded dataset to:", path)

    data_root = os.path.join(path, 'tiny-imagenet-200', 'train')
    transform = get_transforms()
    full_ds = datasets.ImageFolder(root=data_root, transform=transform)

    args = docopt(__doc__)
    ARCHITECTURE = args['--architecture']
    N = int(args['--epochs'])
    CPU_WORKERS = int(args['--cpu-workers'])
    DEVICE = args['--device']
    PAT = int(args['--patience']) if args['--patience'] else 5
    N_TRIALS = int(args['--n-trials']) if args['--n-trials'] else 30

    print(f"=== Optuna Optimization ===")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Device: {DEVICE}")
    print(f"Epochs per trial: {N}")
    print(f"Patience: {PAT}")
    print(f"Number of trials: {N_TRIALS}")

    # Create and run Optuna study
    study = optuna.create_study(direction='maximize', study_name=f'{ARCHITECTURE}_optimization')

    objective = create_objective(
        architecture=ARCHITECTURE,
        dataset=full_ds,
        epochs=N,
        patience=PAT,
        cpu_workers=CPU_WORKERS,
        device=DEVICE
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 60)
    print("=== Optimization Results ===")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study results
    results_df = study.trials_dataframe()
    results_df.to_csv(f'../test_data/optuna_{ARCHITECTURE.lower()}.csv', index=False)
    print(f"\nTrial results saved to ../test_data/optuna_{ARCHITECTURE.lower()}.csv")

    # Run final evaluation with best params
    mean_f1, std_f1, fold_f1s = run_final_evaluation(
        architecture=ARCHITECTURE,
        dataset=full_ds,
        best_params=study.best_params,
        epochs=N,
        patience=PAT,
        cpu_workers=CPU_WORKERS,
        device=DEVICE
    )

    # Save final summary
    summary = {
        'architecture': ARCHITECTURE,
        'best_params': study.best_params,
        'optuna_best_f1': study.best_value,
        'final_mean_f1': mean_f1,
        'final_std_f1': std_f1,
        'fold_f1s': fold_f1s,
    }

    summary_df = pd.DataFrame([{
        'architecture': ARCHITECTURE,
        'pretrained': study.best_params['pretrained'],
        'augmentation': study.best_params['augmentation'],
        'lr': study.best_params['lr'],
        'batch_size': study.best_params['batch_size'],
        'weight_decay': study.best_params['weight_decay'],
        'optimizer': study.best_params['optimizer'],
        'optuna_best_f1': study.best_value,
        'final_mean_f1': mean_f1,
        'final_std_f1': std_f1,
    }])
    summary_df.to_csv(f'../test_data/optuna_{ARCHITECTURE.lower()}_best.csv', index=False)

    print("\n" + "=" * 60)
    print("=== Final Summary ===")
    print("=" * 60)
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Best configuration:")
    print(f"  - Pretrained: {study.best_params['pretrained']}")
    print(f"  - Augmentation: {study.best_params['augmentation']}")
    print(f"  - Learning rate: {study.best_params['lr']:.1e}")
    print(f"  - Batch size: {study.best_params['batch_size']}")
    print(f"  - Weight decay: {study.best_params['weight_decay']:.1e}")
    print(f"  - Optimizer: {study.best_params['optimizer']}")
    print(f"\nFinal F1 (3-fold CV): {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()
