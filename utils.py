import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def valid_model(model, X, y):
    """Validate model on dataset and return accuracy (%)."""
    correct, total = 0, 0
    predict = model.predict(X)
    total += y.size
    correct += (predict == y).sum().item()
    return 100 * correct / total


def plot_confusion_matrix(y_true, y_pred, classes=None, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = sk_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if classes is not None:
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def get_cls_num_list(targets, n_classes, imb_factor):
    """Compute per-class sample counts for long-tail distribution."""
    counts = np.bincount(targets).astype(np.int64)
    max_num = counts.max()
    return [int(max_num * (1.0 / imb_factor) ** (c / (n_classes - 1.0)))
            for c in range(n_classes)]


def compute_imbalance_weights(cls_num_list, beta=0.5):
    """Compute class weights w_k = (1/|D_k|)^beta for ARBN."""
    cls_num_array = np.asarray(cls_num_list, dtype=np.float64)
    if np.any(cls_num_array <= 0):
        raise ValueError("All class counts must be strictly positive.")
    return np.power(1.0 / cls_num_array, beta)
