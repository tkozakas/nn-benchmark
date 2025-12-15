"""Evaluation and metrics utilities."""

import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(model, loader, criterion, device):
    """Return loss, accuracy, precision, recall, f1."""
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
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
    prec = float(precision_score(y_true, y_pred, average='macro', zero_division='warn'))
    rec = float(recall_score(y_true, y_pred, average='macro', zero_division='warn'))
    f1 = float(f1_score(y_true, y_pred, average='macro', zero_division='warn'))

    return loss, acc, prec, rec, f1


def compute_inference_latency(model, loader, device, warmup_runs=5):
    """Measure average inference time per sample."""
    import time

    model.eval()

    # Warmup to avoid CUDA initialization overhead
    dummy = next(iter(loader))[0].to(device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)

    start = time.time()
    total_samples = 0
    with torch.no_grad():
        for batch, _ in loader:
            total_samples += batch.size(0)
            _ = model(batch.to(device))

    return (time.time() - start) / total_samples
