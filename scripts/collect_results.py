#!/usr/bin/env python
"""
Parse experiment logs and render markdown result tables with multiple metrics.

Usage: python scripts/collect_results.py [log_dir]
Output: per-dataset tables with Accuracy, Recall (macro), F1 (macro),
        and Top-5 Accuracy (CIFAR100 only).
"""

import os
import re
import sys
from collections import defaultdict


METRIC_PATTERNS = {
    'accuracy':     re.compile(r'Test Accuracy:\s*([\d.]+)'),
    'recall_macro': re.compile(r'Recall \(macro\):\s*([\d.]+)'),
    'f1_macro':     re.compile(r'F1 \(macro\):\s*([\d.]+)'),
    'top5_accuracy': re.compile(r'Top-5 Accuracy:\s*([\d.]+)'),
}


def parse_log(filepath):
    """Extract all metrics from a log file."""
    if not os.path.exists(filepath):
        return None
    values = {}
    with open(filepath) as f:
        for line in f:
            for key, pattern in METRIC_PATTERNS.items():
                if key not in values:
                    m = pattern.search(line)
                    if m:
                        values[key] = float(m.group(1))
    return values if values else None


def collect(log_dir):
    """Collect metrics from all log files in log_dir."""
    results = defaultdict(dict)  # (dataset, imb, model) -> {metric_key: value}
    datasets = set()
    imbs = set()
    models = set()

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.log') or fname.startswith('_'):
            continue
        base = fname[:-4]
        m = re.match(r'(.+)_(bls|arbn)_IF(\d+)$', base)
        if not m:
            continue
        dataset, model, imb = m.group(1), m.group(2), int(m.group(3))

        datasets.add(dataset)
        imbs.add(imb)
        models.add(model)

        metrics = parse_log(os.path.join(log_dir, fname))
        results[(dataset, imb, model)] = metrics

    return results, sorted(datasets), sorted(imbs), sorted(models)


def fmt(val):
    """Format a metric value or return dash if missing."""
    if val is None:
        return "—"
    return f"{val:.2f}"


def render(results, datasets, imbs, models):
    lines = []
    lines.append("## Experimental Results\n")
    lines.append("所有实验使用默认参数: `feature_times=10, enhance_times=10, "
                 "feature_size=256, reg=0.01, seed=42`。\n")

    for ds in datasets:
        n_classes = 100 if ds == "CIFAR100" else 10
        lines.append(f"### {ds} ({n_classes} classes)\n")

        # Build header
        metric_cols = ["Acc", "Recall(m)", "F1(m)"]
        metric_keys = ["accuracy", "recall_macro", "f1_macro"]
        if n_classes > 10:
            metric_cols.append("Top5")
            metric_keys.append("top5_accuracy")

        header = "| IF |"
        sep = "| --- |"
        for m_name in models:
            for col in metric_cols:
                header += f" {m_name.upper()} {col} |"
                sep += " --- |"
        lines.append(header)
        lines.append(sep)

        for imb in imbs:
            cells = [str(imb)]
            for model in models:
                metrics = results.get((ds, imb, model)) or {}
                for key in metric_keys:
                    val = metrics.get(key)
                    cells.append(fmt(val))
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    results, datasets, imbs, models = collect(log_dir)
    if not results:
        print("No experiment logs found.")
        sys.exit(1)
    table = render(results, datasets, imbs, models)
    print(table)
