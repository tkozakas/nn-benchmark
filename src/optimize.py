"""Optuna hyperparameter optimization."""

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
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from docopt import docopt
from sklearn.model_selection import KFold
from torch import nn, optim

from config import RANDOM_STATE
from data import (
    load_dataset, get_transforms, get_num_classes,
    create_data_loaders, mixup_criterion, AUGMENT_MAP
)
from metrics import evaluate
from model import get_model

os.makedirs('../test_data', exist_ok=True)
warnings.filterwarnings("ignore", message=".*GoogleNet.*", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")


def train_one_epoch(model, loader, criterion, optimizer, device, augment_fn=None):
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


def train_single_fold(architecture, dataset, train_idx, val_idx, pretrained=True,
                      augment_fn=None, epochs=20, batch_size=128, learning_rate=1e-3,
                      weight_decay=1e-4, optimizer_name='adam', early_stopping_patience=5,
                      device='cuda', cpu_workers=4, verbose=True):
    """Train on one fold and return best validation F1 score."""
    device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    n_classes = get_num_classes(dataset)

    train_loader, val_loader = create_data_loaders(
        dataset, train_idx, val_idx, batch_size, cpu_workers
    )

    model = get_model(architecture, num_classes=n_classes, pretrained=pretrained).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if optimizer_name == 'adam':
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        tloss, _ = train_one_epoch(model, train_loader, criterion, opt, device, augment_fn)
        vloss, _, _, _, f1 = evaluate(model, val_loader, criterion, device)

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
    """Create Optuna objective with fixed train/val split for fair comparison."""
    n = len(dataset)
    n_val = int(n * 0.2)
    n_train = n - n_val
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(RANDOM_STATE)).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    def objective(trial):
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
                architecture=architecture, dataset=dataset,
                train_idx=train_idx, val_idx=val_idx,
                pretrained=pretrained, augment_fn=augment_fn,
                epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                optimizer_name=optimizer_name, early_stopping_patience=patience,
                device=device, cpu_workers=cpu_workers, verbose=True
            )
        except Exception as e:
            print(f"    Trial failed: {e}")
            return 0.0

        print(f"  Trial {trial.number} finished with F1: {val_f1:.4f}")
        return val_f1

    return objective


def run_final_evaluation(architecture, dataset, best_params, epochs, patience, cpu_workers, device):
    """Run 3-fold CV with best params for reliable performance estimate."""
    print("\n=== Final Evaluation with Best Parameters (3-fold CV) ===")
    print(f"Best params: {best_params}")

    augment_fn = AUGMENT_MAP[best_params['augmentation']]
    kfold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
        print(f"\n  Fold {fold_idx}/3")
        f1 = train_single_fold(
            architecture=architecture, dataset=dataset,
            train_idx=list(train_idx), val_idx=list(val_idx),
            pretrained=best_params['pretrained'], augment_fn=augment_fn,
            epochs=epochs, batch_size=best_params['batch_size'],
            learning_rate=best_params['lr'], weight_decay=best_params['weight_decay'],
            optimizer_name=best_params['optimizer'], early_stopping_patience=patience,
            device=device, cpu_workers=cpu_workers, verbose=True
        )
        fold_f1s.append(f1)
        print(f"  Fold {fold_idx} F1: {f1:.4f}")

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"\n  Final F1: {mean_f1:.4f} ± {std_f1:.4f}")

    return mean_f1, std_f1, fold_f1s


def main():
    transform = get_transforms()
    full_ds = load_dataset(transform)

    args = docopt(__doc__)
    architecture = args['--architecture']
    epochs = int(args['--epochs'])
    cpu_workers = int(args['--cpu-workers'])
    device = args['--device']
    patience = int(args['--patience']) if args['--patience'] else 5
    n_trials = int(args['--n-trials']) if args['--n-trials'] else 30

    print("=== Optuna Optimization ===")
    print(f"Architecture: {architecture}")
    print(f"Device: {device}")
    print(f"Epochs per trial: {epochs}")
    print(f"Patience: {patience}")
    print(f"Number of trials: {n_trials}")

    study = optuna.create_study(direction='maximize', study_name=f'{architecture}_optimization')
    objective = create_objective(architecture, full_ds, epochs, patience, cpu_workers, device)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("=== Optimization Results ===")
    print("=" * 60)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 score: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    results_df = study.trials_dataframe()
    results_df.to_csv(f'../test_data/optuna_{architecture.lower()}.csv', index=False)
    print(f"\nTrial results saved to ../test_data/optuna_{architecture.lower()}.csv")

    mean_f1, std_f1, _ = run_final_evaluation(
        architecture, full_ds, study.best_params, epochs, patience, cpu_workers, device
    )

    summary_df = pd.DataFrame([{
        'architecture': architecture,
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
    summary_df.to_csv(f'../test_data/optuna_{architecture.lower()}_best.csv', index=False)

    print("\n" + "=" * 60)
    print("=== Final Summary ===")
    print("=" * 60)
    print(f"Architecture: {architecture}")
    print("Best configuration:")
    print(f"  - Pretrained: {study.best_params['pretrained']}")
    print(f"  - Augmentation: {study.best_params['augmentation']}")
    print(f"  - Learning rate: {study.best_params['lr']:.1e}")
    print(f"  - Batch size: {study.best_params['batch_size']}")
    print(f"  - Weight decay: {study.best_params['weight_decay']:.1e}")
    print(f"  - Optimizer: {study.best_params['optimizer']}")
    print(f"\nFinal F1 (3-fold CV): {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()
