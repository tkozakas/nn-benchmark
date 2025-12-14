__doc__ = r"""
Usage:
    optimize.py [--architecture=ARCH]
                [--device=DEVICE]
                [--cpu-workers=NUM]
                [--k-folds=K]
                [--epochs=N]
                [--batch-size=B]
                [--lr=LR]
                [--patience=P]

Options:
    -h --help               Show this help message.
    --device=DEVICE         Device to use for training (cpu or cuda) [default: cuda].
    --cpu-workers=NUM       Number of CPU workers for data loading [default: 6].
    --architecture=ARCH     Model architecture [default: DenseNet121].
    --k-folds=K             Number of CV folds              [default: 3].
    --epochs=N              Max epochs per fold             [default: 20].
    --batch-size=B          Training batch size             [default: 128].
    --lr=LR                 Learning rate                   [default: 0.001].
    --patience=P            Early-stop patience             [default: 5].
"""

import os
import time
import warnings

import kagglehub
import numpy as np
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


def train_one_epoch_augmented(model, loader, criterion, optimizer, device, augment_fn=None):
    """Train model for one epoch with optional Mixup/CutMix augmentation."""
    model.train()
    running_loss = correct = total = 0
    max_cpu = max_gpu = 0.0

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

        cpu = psutil.cpu_percent(interval=None)
        gpu = get_gpu_usage_percent()
        max_cpu = max(max_cpu, cpu)
        max_gpu = max(max_gpu, gpu)

        running_loss += loss.item() * imgs.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total, max_cpu, max_gpu


def get_data_loaders(dataset, train_idx, test_idx, batch_size, num_workers):
    """Split dataset into train/val/test and return DataLoaders."""
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    val_size = int(len(train_subset) * train_config["val_split"])
    train_size = len(train_subset) - val_size
    train_data, val_data = torch.utils.data.random_split(
        train_subset, [train_size, val_size]
    )
    loader_args = dict(
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        num_workers=num_workers,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


def train_with_augmentation(
    architecture,
    dataset,
    pretrained=True,
    augment_fn=None,
    k_folds=3,
    epochs=20,
    batch_size=128,
    learning_rate=1e-3,
    early_stopping_patience=5,
    device='cuda',
    cpu_workers=4,
    random_state=42
):
    """Train model with optional augmentation and pretrained weights."""
    device = torch.device(device)
    criterion = nn.CrossEntropyLoss()
    n_classes = _num_classes(dataset)

    if k_folds is None or k_folds < 2:
        val_frac = train_config["val_split"]
        n = len(dataset)
        n_val = int(n * val_frac)
        n_train = n - n_val
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(random_state)
        )
        folds = [(train_subset.indices, val_subset.indices)]
    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        folds = list(kfold.split(dataset))

    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        print(f"  Fold {fold_idx}/{len(folds)}")

        train_loader, val_loader, test_loader = get_data_loaders(
            dataset, train_idx, test_idx, batch_size, cpu_workers
        )
        num_train_samples = len(train_idx)

        model = get_model(architecture, num_classes=n_classes, pretrained=pretrained).to(device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        best_val_loss = float('inf')
        patience_cnt = 0

        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'f1_score': [], 'precision': [], 'recall': [],
            'epoch_time': [], 'cpu_usage': [], 'gpu_usage': [],
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            tloss, tacc, cpu_peak, gpu_peak = train_one_epoch_augmented(
                model, train_loader, criterion, optimizer, device, augment_fn
            )
            epoch_duration = time.time() - t0

            vloss, vacc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)

            history['train_loss'].append(tloss)
            history['train_accuracy'].append(tacc)
            history['val_loss'].append(vloss)
            history['val_accuracy'].append(vacc)
            history['f1_score'].append(f1)
            history['precision'].append(prec)
            history['recall'].append(rec)
            history['epoch_time'].append(epoch_duration)
            history['cpu_usage'].append(cpu_peak)
            history['gpu_usage'].append(gpu_peak)

            if early_stopping_patience is not None:
                if vloss < best_val_loss:
                    best_val_loss, patience_cnt = vloss, 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= early_stopping_patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break

            print(f"    Epoch {epoch}/{epochs} | "
                  f"Train Loss: {tloss:.3f} | Val F1: {f1:.3f}")

        # Test evaluation
        tloss, tacc, prec, rec, f1 = evaluate(model, test_loader, criterion, device)

        all_results.append({
            'fold': fold_idx,
            'param_count': param_count,
            'test_loss': tloss,
            'test_accuracy': tacc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1_score': f1,
            **history
        })

    return all_results


def run_optimization_experiment(name, architecture, dataset, pretrained=True, augment_fn=None, **kwargs):
    """Run one experimental configuration and collect metrics."""
    folds_data = train_with_augmentation(
        architecture=architecture,
        dataset=dataset,
        pretrained=pretrained,
        augment_fn=augment_fn,
        **kwargs
    )

    test_accs = [f['test_accuracy'] for f in folds_data]
    test_f1s = [f['test_f1_score'] for f in folds_data]
    test_losses = [f['test_loss'] for f in folds_data]

    avg_test_accuracy = sum(test_accs) / len(test_accs)
    avg_test_f1 = sum(test_f1s) / len(test_f1s)
    avg_test_loss = sum(test_losses) / len(test_losses)

    training_times = [sum(f['epoch_time']) for f in folds_data]
    avg_training_time = sum(training_times) / len(training_times)

    first_fold = folds_data[0]

    return {
        'name': name,
        'pretrained': pretrained,
        'augmentation': augment_fn.__name__ if augment_fn else 'None',
        'test_accuracy': avg_test_accuracy,
        'test_f1_score': avg_test_f1,
        'test_loss': avg_test_loss,
        'training_time': avg_training_time,
        'train_loss_curve': first_fold['train_loss'],
        'val_loss_curve': first_fold['val_loss'],
        'f1_score_curve': first_fold['f1_score'],
        'folds_data': folds_data,
    }


def save_optimization_results(runs, filename):
    """Save optimization results to CSV."""
    df = pd.DataFrame([{
        'name': r['name'],
        'pretrained': r['pretrained'],
        'augmentation': r['augmentation'],
        'test_accuracy': r['test_accuracy'],
        'test_f1_score': r['test_f1_score'],
        'test_loss': r['test_loss'],
        'training_time': r['training_time'],
    } for r in runs])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


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
    B = int(args['--batch-size'])
    LR = float(args['--lr'])
    CPU_WORKERS = int(args['--cpu-workers'])
    DEVICE = args['--device']
    K = None if args['--k-folds'] is None or str(args['--k-folds']).lower() == 'none' else int(args['--k-folds'])
    PAT = None if args['--patience'] is None or str(args['--patience']).lower() == 'none' else int(args['--patience'])

    print(f"=== Optimization Experiment ===")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Device: {DEVICE}")
    print(f"K-Folds: {K}, Epochs: {N}, Batch Size: {B}, LR: {LR}, Patience: {PAT}")

    common_kwargs = dict(
        k_folds=K,
        epochs=N,
        batch_size=B,
        learning_rate=LR,
        early_stopping_patience=PAT,
        cpu_workers=CPU_WORKERS,
        device=DEVICE
    )

    runs = []

    # 1) From scratch (baseline)
    print(f"\n[1/6] {ARCHITECTURE} from scratch (baseline)...")
    runs.append(run_optimization_experiment(
        f"{ARCHITECTURE} (scratch)",
        ARCHITECTURE, full_ds,
        pretrained=False,
        augment_fn=None,
        **common_kwargs
    ))

    # 2) From scratch + Mixup
    print(f"\n[2/6] {ARCHITECTURE} from scratch + Mixup...")
    runs.append(run_optimization_experiment(
        f"{ARCHITECTURE} (scratch + Mixup)",
        ARCHITECTURE, full_ds,
        pretrained=False,
        augment_fn=mixup_data,
        **common_kwargs
    ))

    # 3) From scratch + CutMix
    print(f"\n[3/6] {ARCHITECTURE} from scratch + CutMix...")
    runs.append(run_optimization_experiment(
        f"{ARCHITECTURE} (scratch + CutMix)",
        ARCHITECTURE, full_ds,
        pretrained=False,
        augment_fn=cutmix_data,
        **common_kwargs
    ))

    # 4) Pretrained (transfer learning)
    print(f"\n[4/6] {ARCHITECTURE} pretrained...")
    runs.append(run_optimization_experiment(
        f"{ARCHITECTURE} (pretrained)",
        ARCHITECTURE, full_ds,
        pretrained=True,
        augment_fn=None,
        **common_kwargs
    ))

    # 5) Pretrained + Mixup
    print(f"\n[5/6] {ARCHITECTURE} pretrained + Mixup...")
    runs.append(run_optimization_experiment(
        f"{ARCHITECTURE} (pretrained + Mixup)",
        ARCHITECTURE, full_ds,
        pretrained=True,
        augment_fn=mixup_data,
        **common_kwargs
    ))

    # 6) Pretrained + CutMix
    print(f"\n[6/6] {ARCHITECTURE} pretrained + CutMix...")
    runs.append(run_optimization_experiment(
        f"{ARCHITECTURE} (pretrained + CutMix)",
        ARCHITECTURE, full_ds,
        pretrained=True,
        augment_fn=cutmix_data,
        **common_kwargs
    ))

    # Save results
    save_optimization_results(runs, f'../test_data/optimization_{ARCHITECTURE.lower()}.csv')

    # Plot comparison
    plot_optimization_comparison(runs, f'optimization_{ARCHITECTURE.lower()}')

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Configuration':<40} {'Accuracy':>10} {'F1 Score':>10} {'Time (s)':>10}")
    print("-" * 72)
    for r in runs:
        print(f"{r['name']:<40} {r['test_accuracy']:>10.4f} {r['test_f1_score']:>10.4f} {r['training_time']:>10.1f}")

    best = max(runs, key=lambda r: r['test_f1_score'])
    print(f"\nBest configuration: {best['name']} (F1: {best['test_f1_score']:.4f})")


if __name__ == "__main__":
    main()
