import os
import numpy as np
import torch
import torchvision
from torchvision import transforms


def get_dataset(dataset_name="MNIST", data_root=None, batch_size=None,
                imbalance_factor=None, train=True):
    """Load datasets with optional long-tail imbalance.

    Args:
        dataset_name: 'MNIST', 'FashionMNIST', 'CIFAR10', or 'CIFAR100'
        data_root: root path for dataset storage
        batch_size: batch size for DataLoader (None = full dataset)
        imbalance_factor: if set, creates long-tail training distribution
        train: if False, returns balanced test set only
    Returns:
        train_loader, test_loader, n_classes
    """
    if data_root is None:
        data_root = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Dataset configs
    configs = {
        "MNIST": {
            "n_classes": 10,
            "mean": 0.1307, "std": 0.3081,
            "loader": torchvision.datasets.MNIST,
        },
        "FashionMNIST": {
            "n_classes": 10,
            "mean": 0.2860, "std": 0.3530,
            "loader": torchvision.datasets.FashionMNIST,
        },
        "CIFAR10": {
            "n_classes": 10,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "loader": torchvision.datasets.CIFAR10,
        },
        "CIFAR100": {
            "n_classes": 100,
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
            "loader": torchvision.datasets.CIFAR100,
        },
    }

    cfg = configs[dataset_name]
    n_classes = cfg["n_classes"]

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    # Test set
    test_set = cfg["loader"](root=data_root, train=False,
                             download=True, transform=base_transform)
    test_size = len(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size if batch_size is not None else test_size,
        shuffle=False,
    )

    # Training set
    train_set = cfg["loader"](root=data_root, train=True,
                              download=True, transform=base_transform)
    train_size = len(train_set)

    if imbalance_factor is not None and train:
        train_set = _make_imbalanced(train_set, n_classes, imbalance_factor)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size if batch_size is not None else len(train_set),
        shuffle=True,
    )

    return train_loader, test_loader, n_classes


def _make_imbalanced(dataset, n_classes, imb_factor):
    """Create long-tail imbalanced version of a dataset by subsampling."""
    targets = np.array([y for _, y in dataset])
    cls_num_list = get_cls_num_list(targets, n_classes, imb_factor)
    indices = []
    for c in range(n_classes):
        c_indices = np.where(targets == c)[0]
        np.random.shuffle(c_indices)
        indices.append(c_indices[:cls_num_list[c]])
    indices = np.concatenate(indices)
    np.random.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices)


def get_cls_num_list(targets, n_classes, imb_factor):
    """Compute per-class sample counts for long-tail distribution."""
    max_num = np.bincount(targets).max()
    return [int(max_num * (1.0 / imb_factor) ** (c / (n_classes - 1.0)))
            for c in range(n_classes)]


def extract_data(data_loader):
    """Extract numpy arrays from a PyTorch DataLoader."""
    X, Y = [], []
    for x, y in data_loader:
        X = x.numpy()
        Y = y.numpy()
    return X, Y


def count_classes(y, n_classes=None):
    """Count samples per class. Returns list of counts."""
    y = np.asarray(y).flatten()
    if n_classes is None:
        n_classes = len(np.unique(y))
    return [int((y == c).sum()) for c in range(n_classes)]
