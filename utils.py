"""Utility functions for model evaluation and visualization."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    recall_score,
    f1_score,
    precision_score,
)


def accuracy(model, X, y):
    """Compute Top-1 accuracy (%) given a fitted model."""
    predict = model.predict(X)
    return 100.0 * float(np.mean(predict == y))


def top_k_accuracy(y_true, y_proba, k=5):
    """Compute top-k accuracy."""
    top_k_pred = np.argsort(y_proba, axis=1)[:, -k:]
    hits = [y_true[i] in top_k_pred[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def evaluate_model(model, X, y, n_classes):
    """Comprehensive evaluation (accuracy, precision, recall, F1, Top-5).

    Returns a dict of percentage values.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    kw = {"zero_division": 0}
    metrics = {
        "accuracy": float(np.mean(y_pred == y)) * 100.0,
        "recall_macro": recall_score(y, y_pred, average="macro", **kw) * 100.0,
        "recall_micro": recall_score(y, y_pred, average="micro", **kw) * 100.0,
        "precision_macro": precision_score(y, y_pred, average="macro", **kw) * 100.0,
        "precision_micro": precision_score(y, y_pred, average="micro", **kw) * 100.0,
        "f1_macro": f1_score(y, y_pred, average="macro", **kw) * 100.0,
        "f1_micro": f1_score(y, y_pred, average="micro", **kw) * 100.0,
    }
    if n_classes > 10:
        metrics["top5_accuracy"] = top_k_accuracy(y, y_proba, k=5) * 100.0

    return metrics


def print_metrics(metrics, prefix="", n_classes=10):
    """Pretty-print evaluation metrics."""
    tag = f" ({prefix})" if prefix else ""
    print(f"  --- Evaluation Results{tag} ---")
    print(f"  * Accuracy (Top-1):     {metrics['accuracy']:.2f}%")
    print(f"  * Precision (macro):    {metrics['precision_macro']:.2f}%")
    print(f"  * Recall (macro):       {metrics['recall_macro']:.2f}%")
    print(f"  * F1 (macro):           {metrics['f1_macro']:.2f}%")
    print(f"  * Precision (micro):    {metrics['precision_micro']:.2f}%")
    print(f"  * Recall (micro):       {metrics['recall_micro']:.2f}%")
    print(f"  * F1 (micro):           {metrics['f1_micro']:.2f}%")
    if n_classes > 10 and "top5_accuracy" in metrics:
        print(f"  * Top-5 Accuracy:       {metrics['top5_accuracy']:.2f}%")
    print("  ------------------------------------")


def plot_confusion_matrix(y_true, y_pred, classes=None, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if classes is not None:
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
