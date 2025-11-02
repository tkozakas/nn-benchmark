import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import confusion_matrix

BASE_DIR = "../test_data/plot"
CSV_DIR = "../test_data/plot"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_plot(fig, category, suffix):
    png_path = os.path.join(BASE_DIR, category, f"{suffix}.png")
    _ensure_dir(os.path.dirname(png_path))
    fig.savefig(png_path)
    plt.close(fig)


def _save_csv_line(series_dict, category, suffix):
    csv_path = os.path.join(CSV_DIR, category, f"{suffix}.csv")
    _ensure_dir(os.path.dirname(csv_path))
    max_len = max(len(v) for v in series_dict.values())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch"] + list(series_dict.keys())
        writer.writerow(header)
        for i in range(1, max_len + 1):
            row = [i]
            for k in series_dict:
                arr = series_dict[k]
                val = arr[i - 1] if i <= len(arr) else arr[-1]
                row.append(val)
            writer.writerow(row)


def _save_csv_bar(labels, values, category, suffix):
    csv_path = os.path.join(CSV_DIR, category, f"{suffix}.csv")
    _ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "value"])
        for l, v in zip(labels, values):
            writer.writerow([l, v])


def _plot_line(runs, key, title, xlabel, ylabel, category, suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in runs:
        ax.plot(range(1, len(r[key]) + 1), r[key], label=r['name'])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(fontsize='small')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylabel.lower().startswith("tikslumas"):
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{int(x * 100)}%")
        )
    _save_plot(fig, category, suffix)
    series = {r['name']: r[key] for r in runs}
    _save_csv_line(series, category, suffix)


def _plot_bar(labels, values, title, xlabel, ylabel, category, suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)

    _save_plot(fig, category, suffix)
    _save_csv_bar(labels, values, category, suffix)


def _plot_folds(folds_data, metric_key, title, xlabel, ylabel, category, suffix):
    max_epochs = max(len(f[metric_key]) for f in folds_data)
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, f in enumerate(folds_data, start=1):
        data = f[metric_key]
        padded = data + [data[-1]] * (max_epochs - len(data))
        ax.plot(range(1, max_epochs + 1), padded, label=f'Fold {idx}')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(title='Foldas', fontsize='small')
    _save_plot(fig, category, suffix)


def _plot_scatter(labels, xs, ys,
                  title, xlabel, ylabel,
                  category, suffix):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys)
    for l, x, y in zip(labels, xs, ys):
        ax.text(x, y, l, fontsize=8, ha='right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    _save_plot(fig, category, suffix)

    csv_path = os.path.join(CSV_DIR, category, f"{suffix}.csv")
    _ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "x", "y"])
        for l, x, y in zip(labels, xs, ys):
            writer.writerow([l, x, y])


def plot_learning_rate_comparison(runs, name):
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'learning_rate', name + '_train_loss')
    _plot_line(runs, 'train_accuracy_curve', 'Mokymo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'learning_rate', name + '_train_acc')


def plot_optimizer_comparison(runs, name):
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'optimizer', name + '_train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis pername +  epochas',
               'Epochos', 'Nuostolis', 'optimizer', name + '_val_loss')
    _plot_line(runs, 'val_accuracy_curve', 'Validavimo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'optimizer', name + '_val_acc')


def plot_scheduler_comparison(runs, name):
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'scheduler', name + '_train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'scheduler', name + '_val_loss')
    _plot_line(runs, 'val_accuracy_curve', 'Validavimo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'scheduler', name + '_val_acc')
    _plot_line(runs, 'lr_curve', 'Mokymosi greitis per epochas',
               'Epochos', 'Greičio koeficientas', 'scheduler', name + '_lr')


def plot_regularization_comparison(runs, name):
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'regularization', name + '_train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'regularization', name + '_val_loss')
    _plot_line(runs, 'val_accuracy_curve', 'Validavimo tikslumas per epochas',
               'Epochos', 'Tikslumas', 'regularization', name + '_val_acc')
    labels = [r['name'] for r in runs]
    values = [
        sum(r['train_accuracy_curve'][i] - r['val_accuracy_curve'][i]
            for i in range(len(r['val_accuracy_curve'])))
        for r in runs
    ]
    _plot_bar(labels, values, 'Bendroji spraga mokymas–validavimas',
              'Regularizacija', 'Spraga', 'regularization', name + '_gap')


def plot_batch_size_comparison(runs, name):
    labels = [str(r['batch_size']) for r in runs]
    _plot_bar(labels, [r['training_time'] for r in runs],
              'Visas mokymo laikas (s)', 'Partijos dydis', 'Sekundės',
              'batch_size', name + '_time')
    _plot_bar(labels, [r['avg_samples_per_sec'] for r in runs],
              'Pralaidumas (pavyzdžių/s)', 'Partijos dydis', 'Pavyzdžiai/s',
              'batch_size', name + '_throughput')
    _plot_bar(labels, [r['avg_gpu_usage'] for r in runs],
              'Vidutinis GPU naudojimas (%)', 'Partijos dydis', 'Procentai',
              'batch_size', name + '_gpu')


def plot_architecture_comparison(runs, name, acc_target=0.85):
    labels = [r['name'] for r in runs]

    _plot_line(runs, 'f1_score_curve', 'F1 rodiklis per epochas',
               'Epochos', 'F1 rodiklis', 'architecture', name + '_f1_score')

    params = [r['param_count'] for r in runs]
    accs = [r['test_accuracy'] for r in runs]
    _plot_scatter(
        labels, params, accs,
        'Parametrų skaičius vs tikslumas',
        'Parametrų skaičius', 'Tikslumas',
        'architecture', name + '_params'
    )

    _plot_bar(labels, [r['inference_latency'] for r in runs],
              'Inferencijos vėlinimas (ms)', 'Architektūra', 'Milisekundės',
              'architecture', name + '_latency')

    values = [
        sum(r['epoch_time_curve'][: next((i for i, v in enumerate(r['val_accuracy_curve'], 1)
                                          if v >= acc_target),
                                         len(r['val_accuracy_curve']))])
        for r in runs
    ]
    _plot_bar(labels, values, f'Laikas iki tikslumo ≥ {int(acc_target * 100)}%',
              'Architektūra', 'Sekundės', 'architecture', name + '_time_to_acc')


def plot_test_accuracy(runs, name):
    _plot_bar([r['name'] for r in runs], [r['test_accuracy'] for r in runs],
              'Testavimo tikslumas', 'Konfigūracija', 'Tikslumas',
              'test_accuracy', name + '_test_accuracy')


def plot_architecture_by_fold(folds_data, name):
    metrics = [
        ('train_accuracy', 'Mokymo tikslumas per epochą', 'Epochos', 'Tikslumas', 'train_accuracy'),
        ('val_accuracy', 'Validavimo tikslumas per epochą', 'Epochos', 'Tikslumas', 'val_accuracy'),
        ('f1_score', 'Validavimo F1 rodiklis per epochą', 'Epochos', 'F1 rodiklis', 'f1_score'),
    ]

    for key, t, x, y, suffix in metrics:
        _plot_folds(folds_data, key, t, x, y, 'architecture', name + '_' + suffix)


def plot_activation_function_comparison(runs, name):
    labels = [r['name'] for r in runs]
    f1s = [r['test_f1_score'] for r in runs]
    _plot_bar(labels,
              f1s,
              title='Galutinis F1 rodiklis',
              xlabel='Aktyvacijos funkcija',
              ylabel='F1 rodiklis',
              category='activation',
              suffix=name + '_f1_score')
    _plot_line(runs, 'train_loss_curve', 'Mokymo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'activation', name + '_train_loss')
    _plot_line(runs, 'val_loss_curve', 'Validavimo nuostolis per epochas',
               'Epochos', 'Nuostolis', 'activation', name + '_val_loss')


def plot_confusion_matrix(model, loader, device, classes):
    fig, ax = plt.subplots(figsize=(12, 12))
    preds, labels = [], []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs).argmax(dim=1).cpu().numpy()
            preds.extend(out)
            labels.extend(lbls.numpy())
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 8})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Prognozuojama klasė')
    ax.set_ylabel('Tikroji klasė')
    ax.set_title('Sumaišties matrica')
    _save_plot(fig, "confusion", "cm")
