__doc__ = r"""
Usage:
    experiment.py [--architecture=ARCH]
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
    --device=DEVICE         Device to use for training (cpu or cuda) [default: cuda].
    --cpu-workers=NUM       Number of CPU workers for data loading [default: 6].
    --architecture=ARCH     Starting model architecture [default: ResNet18].
    --subsample-size=S      Subsample size for training set [default: None].
    --k-folds=K             Number of CV folds              [default: None].
    --epochs=N              Max epochs per fold             [default: 10].
    --batch-size=B          Training batch size             [default: 128].
    --lr=LR                 Learning rate                   [default: 0.001].
    --weight-decay=WD       Weight decay (L2)               [default: 0.0001].
    --patience=P            Early-stop patience             [default: None].
"""

import os
import warnings

import kagglehub
import pandas as pd
import torch
from docopt import docopt
from torchvision import datasets

from config import supported_architectures
from model import get_model
from train import train, transform
from utility import parse_args, get_subsample
from visualise import (
    plot_architecture_comparison, plot_optimizer_comparison, plot_scheduler_comparison,
    plot_regularization_comparison, plot_batch_size_comparison, plot_architecture_by_fold, plot_learning_rate_comparison
)

os.makedirs('../test_data', exist_ok=True)
warnings.filterwarnings("ignore", message=".*GoogleNet.*", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")


def _num_classes(ds):
    if hasattr(ds, 'classes'):
        return len(ds.classes)
    if hasattr(ds, 'dataset') and hasattr(ds.dataset, 'classes'):
        return len(ds.dataset.classes)
    raise ValueError("Cannot determine number of classes from dataset")


def run_experiment(name, architecture, dataset, **kwargs):
    """Run one experimental configuration and collect metrics using k-fold CV on a single dataset."""
    n_classes = _num_classes(dataset)
    folds_data = train(
        architecture=architecture,
        dataset=dataset,
        model_fn=lambda arch=architecture: get_model(arch, num_classes=n_classes),
        **kwargs
    )

    # Compute averaged curves
    per_fold_curves = {
        'train_loss': [f['train_loss'] for f in folds_data],
        'val_loss': [f['val_loss'] for f in folds_data],
        'train_accuracy': [f['train_accuracy'] for f in folds_data],
        'val_accuracy': [f['val_accuracy'] for f in folds_data],
        'f1_score': [f['f1_score'] for f in folds_data],
        'precision': [f['precision'] for f in folds_data],
        'recall': [f['recall'] for f in folds_data],
        'lr': [f['lr'] for f in folds_data],
        'epoch_time': [f['epoch_time'] for f in folds_data],
        'cpu_usage': [f['cpu_usage'] for f in folds_data],
        'gpu_usage': [f['gpu_usage'] for f in folds_data],
        'samples_per_sec': [f['samples_per_sec'] for f in folds_data],
    }
    avg_curve = {
        k: [sum(vals) / len(vals) for vals in zip(*v)]
        for k, v in per_fold_curves.items()
    }

    # Aggregate test metrics
    test_accs = [f['test_accuracy'] for f in folds_data]
    test_losses = [f['test_loss'] for f in folds_data]
    test_precisions = [f['test_precision'] for f in folds_data]
    test_recalls = [f['test_recall'] for f in folds_data]
    test_f1s = [f['test_f1_score'] for f in folds_data]

    avg_test_accuracy = sum(test_accs) / len(test_accs)
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_precision = sum(test_precisions) / len(test_precisions)
    avg_test_recall = sum(test_recalls) / len(test_recalls)
    avg_test_f1_score = sum(test_f1s) / len(test_f1s)

    # Aggregate performance metrics
    training_time = sum(avg_curve['epoch_time'])
    avg_cpu_usage = sum(avg_curve['cpu_usage']) / len(avg_curve['cpu_usage'])
    avg_gpu_usage = sum(avg_curve['gpu_usage']) / len(avg_curve['gpu_usage'])
    avg_samples_per_sec = sum(avg_curve['samples_per_sec']) / len(avg_curve['samples_per_sec'])

    # Aggregate architecture metrics
    param_counts = [f['param_count'] for f in folds_data]
    inference_latencies = [f['inference_latency'] for f in folds_data]
    avg_param_count = sum(param_counts) / len(param_counts)
    avg_inference_latency = sum(inference_latencies) / len(inference_latencies)

    return {
        'folds_data': folds_data,
        'name': name,
        'batch_size': kwargs.get('batch_size'),
        'param_count': avg_param_count,
        'inference_latency': avg_inference_latency,
        # averaged epoch-wise curves
        'train_loss_curve': avg_curve['train_loss'],
        'val_loss_curve': avg_curve['val_loss'],
        'train_accuracy_curve': avg_curve['train_accuracy'],
        'val_accuracy_curve': avg_curve['val_accuracy'],
        'f1_score_curve': avg_curve['f1_score'],
        'precision_curve': avg_curve['precision'],
        'recall_curve': avg_curve['recall'],
        'lr_curve': avg_curve['lr'],
        'epoch_time_curve': avg_curve['epoch_time'],
        'cpu_usage_curve': avg_curve['cpu_usage'],
        'gpu_usage_curve': avg_curve['gpu_usage'],
        'samples_per_sec_curve': avg_curve['samples_per_sec'],
        # aggregated performance metrics
        'training_time': training_time,
        'avg_cpu_usage': avg_cpu_usage,
        'avg_gpu_usage': avg_gpu_usage,
        'avg_samples_per_sec': avg_samples_per_sec,
        # aggregated test metrics
        'test_accuracy': avg_test_accuracy,
        'test_loss': avg_test_loss,
        'test_precision': avg_test_precision,
        'test_recall': avg_test_recall,
        'test_f1_score': avg_test_f1_score,
    }


def save_test_data(data, filename):
    """Save the metrics to CSV."""
    df = pd.DataFrame(data)
    df.to_csv(
        filename,
        index=False,
        columns=[
            'name',
            'batch_size',
            'param_count',
            'inference_latency',
            'training_time',
            'avg_cpu_usage',
            'avg_gpu_usage',
            'avg_samples_per_sec',
            'test_accuracy',
            'test_loss',
            'test_precision',
            'test_recall',
            'test_f1_score'
        ]
    )


def main():
    cached_path = os.path.expanduser("~/.cache/kagglehub/datasets/akash2sharma/tiny-imagenet/versions/1")
    if os.path.isdir(cached_path):
        path = cached_path
        print("Using existing dataset cache:", path)
    else:
        path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
        print("Downloaded dataset to:", path)

    data_root = os.path.join(path, 'tiny-imagenet-200', 'train')
    full_ds = datasets.ImageFolder(root=data_root, transform=transform)

    print("Running experiments...")
    ARCHITECTURE, B, CPU_WORKERS, DEVICE, K, LR, N, PAT, SUBSAMPLE_SIZE, WD = parse_args(docopt(__doc__))
    ds = get_subsample(full_ds, SUBSAMPLE_SIZE)

    # 1) Learning rate comparison
    print("Running learning rate comparison...")
    lr_map = {
        '1e-5': 1e-5,
        '1e-4': 1e-4,
        '1e-3': 1e-3,
        '1e-2': 1e-2,
        '1e-1': 1e-1,
    }
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=lr,
            optimizer_fn=None,
            weight_decay=WD,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for name, lr in lr_map.items()
    ]
    plot_learning_rate_comparison(runs, 'learning_rate_comparison')

    # 2) Optimizer Comparison
    print("Running optimizer comparison...")
    optim_map = {
        'Adam': lambda p: torch.optim.Adam(p, lr=LR, weight_decay=WD),
        'AdamW': lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=WD),
        'SGD': lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9, weight_decay=WD),
        'RMSprop': lambda p: torch.optim.RMSprop(p, lr=LR, weight_decay=WD)
    }
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=LR, optimizer_fn=opt_fn,
            weight_decay=WD,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for name, opt_fn in optim_map.items()
    ]
    save_test_data(runs, '../test_data/optimizer_comparison.csv')
    plot_optimizer_comparison(runs, 'optimizer_comparison')
    best_opt = max(runs, key=lambda r: r['test_f1_score'])['name']
    best_opt_fn = optim_map[best_opt]
    print(f"Best optimizer: {best_opt}")

    # 3) Scheduler Comparison
    print("Running scheduler comparison...")
    sched_map = {
        'None': None,
        'StepLR': lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=10, gamma=0.1),
        'CosineAnnealing': lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=N),
        'OneCycle': lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=LR, total_steps=N)
    }
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=LR, optimizer_fn=best_opt_fn, scheduler_fn=scheduler,
            weight_decay=WD,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for name, scheduler in sched_map.items()
    ]
    save_test_data(runs, '../test_data/scheduler_comparison.csv')
    plot_scheduler_comparison(runs, 'scheduler_comparison')
    best_sched = max(runs, key=lambda r: r['test_f1_score'])['name']
    best_sched_fn = sched_map[best_sched]
    print(f"Best scheduler: {best_sched}")

    # 4) Regularization Comparison
    print("Running regularization comparison...")
    reg_map = {
        'No WD': 0.0,
        'WD=1e-6': 1e-8,
        'WD=1e-4': 1e-4,
        'WD=1e-2': 1e-2,
        'WD=1e-1': 1e-1,
    }
    runs = [
        run_experiment(
            name, ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=B,
            learning_rate=LR, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for name, wd in reg_map.items()
    ]
    save_test_data(runs, '../test_data/regularization_comparison.csv')
    plot_regularization_comparison(runs, 'regularization_comparison')
    best_reg = max(runs, key=lambda r: r['test_f1_score'])['name']
    best_wd = reg_map[best_reg]
    print(f"Best weight decay: {best_reg}")

    # 5) Batch Size Comparison
    print("Running batch size comparison...")
    batch_sizes = [64, 128, 256, 512]
    runs_bs = [
        run_experiment(
            f"BS={b}", ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=b,
            learning_rate=LR, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=best_wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for b in batch_sizes
    ]
    save_test_data(runs_bs, '../test_data/batch_size_comparison.csv')
    plot_batch_size_comparison(runs_bs, 'batch_size_comparison')
    best_bs = max(runs_bs, key=lambda r: r['test_f1_score'])['batch_size']
    print(f"Best batch size: {best_bs}")

    # 6) Architecture Comparison (use supported_architectures)
    print("Running final architecture comparison...")
    archs = list(supported_architectures.keys())
    runs = [
        run_experiment(
            arch, arch, ds,
            k_folds=K, epochs=N, batch_size=best_bs,
            learning_rate=LR, optimizer_fn=best_opt_fn,
            scheduler_fn=best_sched_fn,
            weight_decay=best_wd,
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS,
            device=DEVICE
        )
        for arch in archs
    ]
    save_test_data(runs, '../test_data/architecture_comparison.csv')
    plot_architecture_comparison(runs, 'architecture_comparison')
    best_run = sorted(
        runs,
        key=lambda r: (
            r['param_count'],  # first: smallest model
            -r['test_f1_score']  # second: highest F1
        )
    )[0]

    # Plot of the best architecture by fold F1 score
    plot_architecture_by_fold(
        best_run['folds_data'], best_run['name'].lower(),
    )

    # 7) Auto-tune core hyperparameters with Optuna (no activation search)
    import optuna

    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        wd = trial.suggest_loguniform("wd", 1e-6, 1e-2)
        bs = trial.suggest_categorical("batch_size", [64, 128, 256])
        opt_name = trial.suggest_categorical("optimizer", list(optim_map.keys()))
        sched_name = trial.suggest_categorical("scheduler", list(sched_map.keys()))

        opt_fn = optim_map[opt_name]
        sched_fn = sched_map[sched_name]

        run = run_experiment(
            f"HPO_trial_{trial.number}",
            ARCHITECTURE, ds,
            k_folds=K, epochs=N, batch_size=bs,
            learning_rate=lr, optimizer_fn=opt_fn,
            scheduler_fn=sched_fn,
            weight_decay=wd, early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS, device=DEVICE
        )
        return run['test_f1_score']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    print("⇒  Best HPO params:", study.best_params)

    # 8) Re-run architecture comparison with best HPO settings
    best = study.best_params
    runs = [
        run_experiment(
            arch, arch, ds,
            k_folds=K, epochs=N, batch_size=best["batch_size"],
            learning_rate=best["lr"],
            optimizer_fn=optim_map[best["optimizer"]],
            scheduler_fn=sched_map[best["scheduler"]],
            weight_decay=best["wd"],
            early_stopping_patience=PAT,
            cpu_workers=CPU_WORKERS, device=DEVICE
        )
        for arch in archs
    ]
    plot_architecture_comparison(runs, 'architecture_comparison_hpo')
    best_run = sorted(
        runs,
        key=lambda r: (r['param_count'], -r['test_f1_score'])
    )[0]
    plot_architecture_by_fold(
        best_run['folds_data'], best_run['name'].lower(),
    )

    print("Best configuration summary → "
          f"Architecture: {best_run['name']}, Optimizer: {best_opt}, Scheduler: {best_sched}, "
          f"Weight Decay: {best_reg}, Batch Size: {best_bs}, LR: {LR}, Epochs: {N}, Patience: {PAT}, Subsample: {SUBSAMPLE_SIZE}")


if __name__ == "__main__":
    main()
