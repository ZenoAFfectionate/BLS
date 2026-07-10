#!/usr/bin/env python
"""Main training & evaluation script for BLS / ARBN."""

import argparse
import numpy as np

from models import BLS, ARBN
from utils import accuracy, evaluate_model, print_metrics
from loader.model_loader import load_model, store_model
from loader.data_loader import get_dataset, extract_data, count_classes


def auto_or_int(value):
    """Accept 'auto' string or integer value."""
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"invalid value: '{value}' (expected int or 'auto')"
    )


parser = argparse.ArgumentParser(description="BLS / ARBN Training Script")

# Dataset
parser.add_argument(
    "--dataset", type=str, default="MNIST",
    choices=["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"],
)
parser.add_argument("--data_root", type=str, default=None,
                    help="Root directory for datasets")
parser.add_argument("--imbalance_factor", type=float, default=None,
                    help="Long-tail imbalance factor (e.g. 100)")

# Model selection
parser.add_argument("--model", type=str, default="bls",
                    choices=["bls", "arbn"],
                    help="Model type: bls or arbn")

# Network structure
parser.add_argument("--feature_times", type=int, default=10)
parser.add_argument("--enhance_times", type=int, default=10)
parser.add_argument("--feature_size", type=auto_or_int, default=256)
parser.add_argument("--enhance_func", type=str, default="tanh")
parser.add_argument("--mapping_func", type=str, default="linear")

# Regularization
parser.add_argument("--reg", type=float, default=0.01)

# ARBN-specific
parser.add_argument("--class_weight_beta", type=float, default=0.5,
                    help="Beta for class weight computation in ARBN")
parser.add_argument("--no_adaptive_reg", action="store_true",
                    help="Disable adaptive regularization in ARBN")

# Incremental learning
parser.add_argument("--enhance_epoch", type=int, default=0,
                    help="Number of incremental enhancement rounds")
parser.add_argument("--enhance_nodes", type=int, default=10,
                    help="Enhancement nodes per round")

# I/O
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--loading", "-l", action="store_true", default=False)
parser.add_argument("--storing", "-s", action="store_true", default=False)
parser.add_argument("--sparse", "-p", action="store_true", default=False)

args = parser.parse_args()

np.random.seed(args.seed)

# ---- Data Loading ----------------------------------------------------------
print(f"> Loading the dataset {args.dataset} ...", end=" ")
train_loader, valid_loader, n_classes = get_dataset(
    args.dataset,
    data_root=args.data_root,
    imbalance_factor=args.imbalance_factor,
    train=True,
)
X_train, y_train = extract_data(train_loader)
X_valid, y_valid = extract_data(valid_loader)
print("Success")

cls_num_list = count_classes(y_train, n_classes)
print(f"  Train samples: {X_train.shape[0]}, Test samples: {X_valid.shape[0]}")
print(f"  Class distribution: min={min(cls_num_list)}, max={max(cls_num_list)}")

# ---- Model Initialization --------------------------------------------------
model_type_str = args.model.upper()
print(f"\n> Initialize the {model_type_str} model ...")

if args.loading:
    ModelClass = ARBN if args.model == "arbn" else BLS
    model = load_model(ModelClass, args.dataset + "_best.pkl")
else:
    common_kwargs = dict(
        feature_times=args.feature_times,
        enhance_times=args.enhance_times,
        feature_size=args.feature_size,
        n_classes=n_classes,
        mapping_function=args.mapping_func,
        enhance_function=args.enhance_func,
        reg=args.reg,
        use_sparse=args.sparse,
    )

    if args.model == "arbn":
        model = ARBN(
            **common_kwargs,
            cls_num_list=cls_num_list,
            adaptive_reg=not args.no_adaptive_reg,
            class_weight_beta=args.class_weight_beta,
        )
    else:
        model = BLS(**common_kwargs)

# ---- Training --------------------------------------------------------------
print("> Training model ...")
model.fit(X_train, y_train)
print("  Finish training\n")

# ---- Evaluation ------------------------------------------------------------
train_acc = accuracy(model, X_train, y_train)
print(f"  * Accuracy (train): {train_acc:.2f}%")

test_metrics = evaluate_model(model, X_valid, y_valid, n_classes=n_classes)
print_metrics(test_metrics, prefix="valid", n_classes=n_classes)
print()

# ---- Incremental Enhancement -----------------------------------------------
for epoch in range(args.enhance_epoch):
    print(f"> Incremental round {epoch + 1}/{args.enhance_epoch}")
    model.add_enhancement_nodes(X_train, y_train, args.enhance_nodes)
    t_acc = accuracy(model, X_train, y_train)
    print(f"  * Accuracy (train): {t_acc:.2f}%")
    metrics = evaluate_model(model, X_valid, y_valid, n_classes=n_classes)
    print_metrics(metrics, prefix="valid", n_classes=n_classes)
    print()

# ---- Save Model ------------------------------------------------------------
if args.storing:
    filename = f"{args.dataset}_{args.model}_best.pkl"
    store_model(model, filename)
    print(f"  Model saved to {filename}")

# ---- Final Summary ---------------------------------------------------------
print("-" * 50)
imb_str = (
    f"IF={int(args.imbalance_factor)}"
    if args.imbalance_factor
    else "balanced"
)
test_acc = test_metrics["accuracy"]
print(f"  Dataset: {args.dataset} ({imb_str})")
print(f"  Model: {model_type_str}")
print(f"  Test Accuracy: {test_acc:.2f}%")
if "recall_macro" in test_metrics:
    print(f"  Recall (macro): {test_metrics['recall_macro']:.2f}%")
if "f1_macro" in test_metrics:
    print(f"  F1 (macro): {test_metrics['f1_macro']:.2f}%")
if "top5_accuracy" in test_metrics:
    print(f"  Top-5 Accuracy: {test_metrics['top5_accuracy']:.2f}%")
print("-" * 50)
