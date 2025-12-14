"""Experiment runner for hyperparameter comparison."""

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
    --patience=P            Early-stop patience             [default: None].
"""

import os
import warnings

import optuna
import pandas as pd
import torch
from docopt import docopt

from config import SUPPORTED_ARCHITECTURES
from data import load_dataset, get_transforms, get_subsample, get_num_classes
from model import get_model
from train import train
from utility import parse_args
from visualise import (
    plot_architecture_comparison, plot_optimizer_comparison, plot_scheduler_comparison,
    plot_batch_size_comparison, plot_architecture_by_fold, plot_learning_rate_comparison
)

os.makedirs('../test_data', exist_ok=True)
warnings.filterwarnings("ignore", message=".*GoogleNet.*", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")


def run_experiment(name, architecture, dataset, **kwargs):
    """Run one experimental configuration and collect metrics using k-fold CV."""
    n_classes = get_num_classes(dataset)
    folds_data = train(
        architecture=architecture,
        dataset=dataset,
        model_fn=lambda arch=architecture: get_model(arch, num_classes=n_classes),
        **kwargs
    )

    # Compute averaged curves
    curve_keys = [
        'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy',
        'f1_score', 'precision', 'recall', 'lr', 'epoch_time',
        'cpu_usage', 'gpu_usage', 'samples_per_sec'
    ]
    per_fold_curves = {k: [f[k] for f in folds_data] for k in curve_keys}
    avg_curve = {
        k: [sum(vals) / len(vals) for vals in zip(*v)]
        for k, v in per_fold_curves.items()
    }

    # Aggregate test metrics
    test_keys = ['test_accuracy', 'test_loss', 'test_precision', 'test_recall', 'test_f1_score']
    test_avgs = {k: sum(f[k] for f in folds_data) / len(folds_data) for k in test_keys}

    # Aggregate performance metrics
    training_time = sum(avg_curve['epoch_time'])
    avg_cpu_usage = sum(avg_curve['cpu_usage']) / len(avg_curve['cpu_usage'])
    avg_gpu_usage = sum(avg_curve['gpu_usage']) / len(avg_curve['gpu_usage'])
    avg_samples_per_sec = sum(avg_curve['samples_per_sec']) / len(avg_curve['samples_per_sec'])

    # Aggregate architecture metrics
    avg_param_count = sum(f['param_count'] for f in folds_data) / len(folds_data)
    avg_inference_latency = sum(f['inference_latency'] for f in folds_data) / len(folds_data)

    return {
        'folds_data': folds_data,
        'name': name,
        'batch_size': kwargs.get('batch_size'),
        'param_count': avg_param_count,
        'inference_latency': avg_inference_latency,
        # Averaged epoch-wise curves
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
        # Aggregated performance metrics
        'training_time': training_time,
        'avg_cpu_usage': avg_cpu_usage,
        'avg_gpu_usage': avg_gpu_usage,
        'avg_samples_per_sec': avg_samples_per_sec,
        # Aggregated test metrics
        **test_avgs,
    }


def save_test_data(data, filename):
    """Save the metrics to CSV."""
    columns = [
        'name', 'batch_size', 'param_count', 'inference_latency',
        'training_time', 'avg_cpu_usage', 'avg_gpu_usage', 'avg_samples_per_sec',
        'test_accuracy', 'test_loss', 'test_precision', 'test_recall', 'test_f1_score'
    ]
    pd.DataFrame(data).to_csv(filename, index=False, columns=columns)


def main():
    transform = get_transforms()
    full_ds = load_dataset(transform)

    print("Running experiments...")
    args = parse_args(docopt(__doc__))
    architecture, batch_size, cpu_workers, device, k_folds, lr, epochs, patience, subsample_size = args
    ds = get_subsample(full_ds, subsample_size)

    common_args = dict(
        k_folds=k_folds, epochs=epochs, batch_size=batch_size,
        early_stopping_patience=patience, cpu_workers=cpu_workers, device=device
    )

    # 1) Learning rate comparison
    print("Running learning rate comparison...")
    lr_map = {'1e-5': 1e-5, '1e-4': 1e-4, '1e-3': 1e-3, '1e-2': 1e-2, '1e-1': 1e-1}
    runs = [
        run_experiment(name, architecture, ds, learning_rate=val, optimizer_fn=None, **common_args)
        for name, val in lr_map.items()
    ]
    plot_learning_rate_comparison(runs, 'learning_rate_comparison')

    # 2) Optimizer Comparison
    print("Running optimizer comparison...")
    optim_map = {
        'Adam': lambda p: torch.optim.Adam(p, lr=lr),
        'AdamW': lambda p: torch.optim.AdamW(p, lr=lr),
        'SGD': lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
        'RMSprop': lambda p: torch.optim.RMSprop(p, lr=lr)
    }
    runs = [
        run_experiment(name, architecture, ds, learning_rate=lr, optimizer_fn=opt_fn, **common_args)
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
        'CosineAnnealing': lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs),
        'OneCycle': lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=lr, total_steps=epochs)
    }
    runs = [
        run_experiment(name, architecture, ds, learning_rate=lr, optimizer_fn=best_opt_fn,
                       scheduler_fn=sched_fn, **common_args)
        for name, sched_fn in sched_map.items()
    ]
    save_test_data(runs, '../test_data/scheduler_comparison.csv')
    plot_scheduler_comparison(runs, 'scheduler_comparison')
    best_sched = max(runs, key=lambda r: r['test_f1_score'])['name']
    best_sched_fn = sched_map[best_sched]
    print(f"Best scheduler: {best_sched}")

    # 4) Batch Size Comparison
    print("Running batch size comparison...")
    batch_sizes = [64, 128, 256, 512]
    runs_bs = [
        run_experiment(f"BS={b}", architecture, ds, learning_rate=lr, optimizer_fn=best_opt_fn,
                       scheduler_fn=best_sched_fn, batch_size=b, k_folds=k_folds, epochs=epochs,
                       early_stopping_patience=patience, cpu_workers=cpu_workers, device=device)
        for b in batch_sizes
    ]
    save_test_data(runs_bs, '../test_data/batch_size_comparison.csv')
    plot_batch_size_comparison(runs_bs, 'batch_size_comparison')
    best_bs = max(runs_bs, key=lambda r: r['test_f1_score'])['batch_size']
    print(f"Best batch size: {best_bs}")

    # 5) Architecture Comparison
    print("Running final architecture comparison...")
    archs = list(SUPPORTED_ARCHITECTURES.keys())
    runs = [
        run_experiment(arch, arch, ds, learning_rate=lr, optimizer_fn=best_opt_fn,
                       scheduler_fn=best_sched_fn, batch_size=best_bs, k_folds=k_folds,
                       epochs=epochs, early_stopping_patience=patience,
                       cpu_workers=cpu_workers, device=device)
        for arch in archs
    ]
    save_test_data(runs, '../test_data/architecture_comparison.csv')
    plot_architecture_comparison(runs, 'architecture_comparison')
    best_run = sorted(runs, key=lambda r: (r['param_count'], -r['test_f1_score']))[0]
    plot_architecture_by_fold(best_run['folds_data'], best_run['name'].lower())

    # 6) Optuna HPO
    def objective(trial):
        trial_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [64, 128, 256])
        opt_name = trial.suggest_categorical("optimizer", list(optim_map.keys()))
        sched_name = trial.suggest_categorical("scheduler", list(sched_map.keys()))

        run = run_experiment(
            f"HPO_trial_{trial.number}", architecture, ds,
            k_folds=k_folds, epochs=epochs, batch_size=bs,
            learning_rate=trial_lr, optimizer_fn=optim_map[opt_name],
            scheduler_fn=sched_map[sched_name], weight_decay=wd,
            early_stopping_patience=patience, cpu_workers=cpu_workers, device=device
        )
        return run['test_f1_score']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    print(f"Best HPO params: {study.best_params}")

    # 7) Re-run architecture comparison with best HPO settings
    best = study.best_params
    runs = [
        run_experiment(arch, arch, ds, k_folds=k_folds, epochs=epochs,
                       batch_size=best["batch_size"], learning_rate=best["lr"],
                       optimizer_fn=optim_map[best["optimizer"]],
                       scheduler_fn=sched_map[best["scheduler"]],
                       weight_decay=best["wd"], early_stopping_patience=patience,
                       cpu_workers=cpu_workers, device=device)
        for arch in archs
    ]
    plot_architecture_comparison(runs, 'architecture_comparison_hpo')
    best_run = sorted(runs, key=lambda r: (r['param_count'], -r['test_f1_score']))[0]
    plot_architecture_by_fold(best_run['folds_data'], best_run['name'].lower())

    print(f"Best configuration summary -> Architecture: {best_run['name']}, "
          f"Optimizer: {best_opt}, Scheduler: {best_sched}, "
          f"Batch Size: {best_bs}, LR: {lr}, Epochs: {epochs}, "
          f"Patience: {patience}, Subsample: {subsample_size}")


if __name__ == "__main__":
    main()
