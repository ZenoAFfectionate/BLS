#!/usr/bin/env python
"""
解析实验日志，生成 README 格式的结果表格。

用法: python scripts/collect_results.py [log_dir]
输出: 每个数据集一个表格，行=IF, 列=BLS/ARBN
"""

import os
import re
import sys
from collections import defaultdict


def parse_log(filepath):
    """从日志中提取 Test Accuracy."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        for line in f:
            m = re.search(r'Test Accuracy:\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
    return None


def collect(log_dir):
    results = defaultdict(dict)  # (dataset, imb, model) -> acc
    datasets = set()
    imbs = set()
    models = set()

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.log') or fname.startswith('_'):
            continue
        # 格式: {dataset}_{model}_IF{imb}.log
        base = fname[:-4]
        m = re.match(r'(.+)_(bls|arbn)_IF(\d+)$', base)
        if not m:
            continue
        dataset, model, imb = m.group(1), m.group(2), int(m.group(3))

        datasets.add(dataset)
        imbs.add(imb)
        models.add(model)

        acc = parse_log(os.path.join(log_dir, fname))
        results[(dataset, imb, model)] = acc

    return results, sorted(datasets), sorted(imbs), sorted(models)


def render(results, datasets, imbs, models):
    lines = []
    lines.append("## Experimental Results\n")
    lines.append("所有实验使用默认参数: `feature_times=10, enhance_times=10, "
                 "feature_size=256, reg=0.01, seed=42`。\n")

    for ds in datasets:
        n_classes = 100 if ds == "CIFAR100" else 10
        lines.append(f"### {ds} ({n_classes} classes)\n")

        header = "| IF | " + " | ".join(f"{m.upper()} Acc (%)" for m in models) + " |"
        lines.append(header)
        lines.append("|" + " --- |" * (len(models) + 1))

        for imb in imbs:
            cells = [str(imb)]
            for model in models:
                acc = results.get((ds, imb, model))
                if acc is not None:
                    cells.append(f"{acc:.2f}")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    lines.append("> 运行 `bash scripts/run_all_experiments.sh` 获取实验结果并填入表格。")

    return "\n".join(lines)


if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    results, datasets, imbs, models = collect(log_dir)
    if not results:
        print("No experiment logs found.")
        sys.exit(1)
    table = render(results, datasets, imbs, models)
    print(table)
