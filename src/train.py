"""Training loop and utilities."""

__doc__ = r"""
Usage:
    train.py [--architecture=ARCH]
             [--device=DEVICE]
             [--cpu-workers=NUM]
             [--subsample-size=S]
             [--k-folds=K]
             [--epochs=N]
             [--batch-size=B]
             [--lr=LR]
             [--weight-decay=WD]
             [--patience=P]

Options:
    -h --help               Show this help message.
    --architecture=ARCH     Model architecture [default: ResNet18].
    --device=DEVICE         Device to use (cpu or cuda) [default: cpu].
    --cpu-workers=NUM       Number of CPU workers for data loading [default: 4].
    --subsample-size=S      Subsample size for training set [default: None].
    --k-folds=K             Number of CV folds [default: 5].
    --epochs=N              Max epochs per fold [default: 20].
    --batch-size=B          Training batch size [default: 128].
    --lr=LR                 Learning rate [default: 0.001].
    --patience=P            Early-stop patience [default: 5].
"""

import re
import subprocess
import time
import warnings

import psutil
import torch
from docopt import docopt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import VAL_SPLIT, RANDOM_STATE
from data import load_dataset, get_subsample, create_fold_loaders, get_transforms
from metrics import evaluate, compute_inference_latency
from model import get_model, save_model, load_model
from utility import parse_args
from visualise import plot_confusion_matrix

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*hipBLASLt.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")
cudnn.benchmark = True


def get_gpu_usage_percent():
    """Return GPU utilization percent for NVIDIA or AMD/ROCm."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        pynvml.nvmlShutdown()
        return float(util)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            m = re.search(r"GPU use\s*\(%\)\s*:\s*(\d+)", line)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch, returning loss, acc, peak CPU/GPU usage."""
    model.train()
    running_loss = correct = total = 0
    max_cpu = max_gpu = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Resource sampling
        cpu = psutil.cpu_percent(interval=None)
        gpu = get_gpu_usage_percent()
        max_cpu = max(max_cpu, cpu)
        max_gpu = max(max_gpu, gpu)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total, max_cpu, max_gpu


def init_model_optimizer_scheduler(model_fn, learning_rate, weight_decay,
                                   optimizer_fn, scheduler_fn, device):
    """Instantiate model, optimizer, and scheduler."""
    model = model_fn().to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optimizer_fn(model.parameters()) if optimizer_fn else optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = scheduler_fn(optimizer) if scheduler_fn else None
    return model, optimizer, scheduler


def train(architecture, dataset, model_fn, k_folds=None, epochs=10, batch_size=128,
          learning_rate=1e-3, weight_decay=1e-4, optimizer_fn=None, scheduler_fn=None,
          criterion=None, early_stopping_patience=None, device='cuda', cpu_workers=4,
          random_state=RANDOM_STATE):
    """Train model with optional k-fold cross-validation."""
    device = torch.device(device)
    criterion = criterion or nn.CrossEntropyLoss()

    if k_folds is None:
        n = len(dataset)
        n_val = int(n * VAL_SPLIT)
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
    total_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        num_train_samples = len(train_idx)
        train_loader, val_loader, test_loader = create_fold_loaders(
            dataset, train_idx, test_idx, batch_size, cpu_workers
        )

        model, optimizer, scheduler = init_model_optimizer_scheduler(
            model_fn, learning_rate, weight_decay, optimizer_fn, scheduler_fn, device
        )

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best_val_loss = float('inf')
        patience_cnt = 0

        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'f1_score': [], 'precision': [], 'recall': [],
            'lr': [], 'epoch_time': [],
            'cpu_usage': [], 'gpu_usage': [],
            'samples_per_sec': []
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tloss, tacc, cpu_peak, gpu_peak = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            epoch_duration = time.time() - t0

            vloss, vacc, prec, rec, f1 = evaluate(model, val_loader, criterion, device)
            throughput = num_train_samples / epoch_duration

            history['train_loss'].append(tloss)
            history['train_accuracy'].append(tacc)
            history['val_loss'].append(vloss)
            history['val_accuracy'].append(vacc)
            history['f1_score'].append(f1)
            history['precision'].append(prec)
            history['recall'].append(rec)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_duration)
            history['samples_per_sec'].append(throughput)
            history['cpu_usage'].append(cpu_peak)
            history['gpu_usage'].append(gpu_peak)

            if scheduler:
                scheduler.step()

            if early_stopping_patience is not None:
                if vloss < best_val_loss:
                    best_val_loss, patience_cnt = vloss, 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            print(f"Fold {fold_idx} | Epoch {epoch}/{epochs} | "
                  f"Train Loss: {tloss:.3f} | Train Acc: {tacc:.3f} | "
                  f"Val Loss: {vloss:.3f} | Val Acc: {vacc:.3f} | "
                  f"F1: {f1:.3f} | CPU: {cpu_peak:.1f}% | GPU: {gpu_peak:.1f}% | "
                  f"Time: {epoch_duration:.2f}s")

        # Evaluate on test set
        tloss, tacc, prec, rec, f1 = evaluate(model, test_loader, criterion, device)
        print(f"Test | Loss: {tloss:.3f} | Acc: {tacc:.3f} | "
              f"Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        inf_latency = compute_inference_latency(model, test_loader, device)

        history.update({
            'test_loss': tloss,
            'test_accuracy': tacc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1_score': f1,
        })

        save_model(model, f"{architecture}_fold{fold_idx}.pth")
        all_results.append({
            'fold': fold_idx,
            'param_count': param_count,
            'inference_latency': inf_latency,
            **history
        })

    print(f"Total training time: {time.time() - total_start:.2f}s")
    return all_results


def main():
    print("Starting training...")
    args = parse_args(docopt(__doc__))
    architecture, batch_size, cpu_workers, device, k_folds, lr, epochs, patience, subsample_size = args

    transform = get_transforms()
    full = load_dataset(transform)
    ds = get_subsample(full, subsample_size)
    num_classes = len(full.classes)

    results = train(
        device=device,
        architecture=architecture,
        dataset=ds,
        model_fn=lambda: get_model(architecture, num_classes=num_classes),
        k_folds=k_folds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        early_stopping_patience=patience,
        cpu_workers=cpu_workers,
        optimizer_fn=optim.Adam,
        scheduler_fn=None,
    )

    # Final confusion matrix
    fold_to_load = results[-1]['fold']
    test_loader = DataLoader(
        ds, batch_size=batch_size, num_workers=cpu_workers,
        shuffle=False, pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    model = get_model(architecture, num_classes=num_classes).to(device)
    model = load_model(model, f"{architecture}_fold{fold_to_load}.pth")
    plot_confusion_matrix(model, test_loader, device, classes=list(range(num_classes)))

    test_loss, test_acc, precision, recall, f1_val = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device
    )
    print("\nTest results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_val:.4f}")


if __name__ == "__main__":
    main()
