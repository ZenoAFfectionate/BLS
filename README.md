# BLS & ARBN: Broad Learning System with Adaptive Re-weighting

Numpy-based implementation of the Broad Learning System (BLS) and Adaptive Re-weighting Broad Network (ARBN) for long-tail image classification.

## Mathematical Background

### Standard BLS (Broad Learning System)

The Broad Learning System (BLS) is a single-hidden-layer, width-augmented architecture that constructs a wide hidden representation via random feature mappings and nonlinear enhancement nodes and learns the linear readout layer in closed form, thereby enabling rapid training and efficient structural expansion.

Given supervised data $\mathbf{X} \in \mathbb{R}^{N \times d}$ and one-hot labels $\mathbf{Y} \in \mathbb{R}^{N \times C}$, BLS constructs $n$ groups of feature nodes together with $m$ groups of enhancement nodes. The output of the $i$-th feature-node group is:

$$\mathbf{Z}_i = \phi_i(\mathbf{X} \mathbf{W}_i + \mathbf{1} \beta_i^{\top}), \quad i = 1, 2, \dots, n,$$

where $\mathbf{W}_i \in \mathbb{R}^{d \times p_i}$ denotes the weight matrix and $\beta_i \in \mathbb{R}^{p_i}$ is the bias vector — both randomly initialized once and kept fixed. $\phi_i(\cdot)$ is an element-wise nonlinear activation (e.g., linear, ReLU), and $\mathbf{1} \in \mathbb{R}^{N \times 1}$ is an all-ones vector. Stacking feature groups yields $\mathbf{Z}^n = [\mathbf{Z}_1, \dots, \mathbf{Z}_n]$.

For the enhancement part, the $j$-th enhancement group is formulated as:

$$\mathbf{H}_j = \xi_j(\mathbf{Z}^n \mathbf{W}_j^{(h)} + \mathbf{1} \gamma_j^{\top}), \quad \text{s.t. } \mathbf{H}_j^{\top} \mathbf{H}_j = \mathbf{I}_{q_j}, \quad j = 1, 2, \dots, m,$$

where Gram–Schmidt orthogonalization is applied to reduce redundancy and mutual interference among enhancement groups. Stacking enhancement groups yields $\mathbf{H}^m = [\mathbf{H}_1, \dots, \mathbf{H}_m]$.

The hidden design matrix is formed by concatenating feature and enhancement groups:

$$\mathbf{A} = [\mathbf{Z}_1, \dots, \mathbf{Z}_n \mid \mathbf{H}_1, \dots, \mathbf{H}_m] = [\mathbf{Z}^n \mid \mathbf{H}^m].$$

Given $\mathbf{A}$ and target labels $\mathbf{Y}$, BLS estimates the output weight matrix $\mathbf{W}$ by solving a ridge-regularized least-squares problem:

$$\mathcal{L}_{\text{BLS}} = \| \mathbf{A} \mathbf{W} - \mathbf{Y} \|_F^2 + \lambda \| \mathbf{W} \|_F^2,$$

where $\lambda > 0$ controls the strength of $\ell_2$ regularization. The optimization admits a closed-form solution:

$$\mathbf{W}^* = (\mathbf{A}^{\top} \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^{\top} \mathbf{Y}.$$

This closed-form solution avoids gradient-based optimization and enables fast training.

### ARBN (Adaptive Re-weighting Broad Network)

Under long-tailed label distributions, the least-squares objective is dominated by head-class samples, causing the learned classifier to shift decision boundaries toward frequent classes and systematically underestimate tail classes. Sampling-based remedies (e.g., class-balanced resampling) partially mitigate this issue but introduce variance without modifying the final closed-form solution, leaving residual bias unaddressed.

To directly inject imbalance information at solve time, ARBN augments the BLS readout with class-aware sample weights. Let $|D_k|$ denote the number of training instances belonging to class $k$. Each class is assigned a weight according to the effective number of samples:

$$w_k = \left( \frac{1}{|D_k|} \right)^{\beta}, \quad k = 1, \dots, C,$$

where $\beta \in [0, 1]$ is a single smoothing hyper-parameter:
- $\beta = 0$: uniform weights, recovering standard (unweighted) BLS
- $\beta = 1$: inverse class frequency, maximally amplifying tail classes
- Intermediate $\beta$: a continuous trade-off between the two extremes, often outperforming rigid linear or logarithmic heuristics

The per-sample weight for the $i$-th training instance with label $y_i$ is $w_i = w_{y_i}$. These weights are assembled into a diagonal weighting matrix:

$$\mathbf{W}_d = \operatorname{diag}(w_1, \dots, w_N) \in \mathbb{R}^{N \times N}.$$

Instead of explicitly constructing weighted design matrices, ARBN incorporates the weights directly into the normal equations, yielding a weighted ridge regression objective:

$$\mathcal{L}_{\text{ARBN}} = \sum_{i=1}^{N} w_i \| \mathbf{a}_i \mathbf{W} - \mathbf{y}_i \|_2^2 + \lambda \| \mathbf{W} \|_F^2,$$

where $\mathbf{a}_i$ is the $i$-th row of $\mathbf{A}$. The optimization remains strictly convex and admits the closed-form solution:

$$\mathbf{W}^* = (\mathbf{A}^{\top} \mathbf{W}_d \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^{\top} \mathbf{W}_d \mathbf{Y}.$$

This formulation is algebraically equivalent to explicitly re-weighting the design matrix but offers superior computational efficiency and numerical stability: it avoids the $O(N \cdot L^2)$ cost of forming $\mathbf{A}_w = \mathbf{W}_d^{1/2} \mathbf{A}$ and preserves the sparsity structure of $\mathbf{A}^{\top}\mathbf{A}$ whenever $\mathbf{A}$ is sparse.

By preserving the flat, closed-form nature of BLS while explicitly counteracting head-class dominance through a tempered, class-aware re-weighting scheme, ARBN circumvents the opaque gradient bias inherent in iterative, back-propagation-based classifiers. Unlike conventional deep classifiers that rely on gradient statistics from imbalanced mini-batches, ARBN analytically solves for the classifier weights in a single deterministic step, yielding more equitable decision boundaries for tail classes without sacrificing the hallmark efficiency and structural expandability of BLS.

## Quick Start

### Environment

```bash
conda create -n bls python=3.12 -y
conda activate bls
pip install -r requirements.txt
```

### Run

```bash
# BLS on balanced MNIST
python main.py --dataset MNIST --model bls --data_root ./data

# ARBN on long-tail CIFAR-10 (IF=100)
python main.py --dataset CIFAR10 --model arbn --data_root ./data --imbalance_factor 100

# ARBN on CIFAR-100 with custom beta
python main.py --dataset CIFAR100 --model arbn --data_root ./data \
    --imbalance_factor 100 --class_weight_beta 0.5

# Run all experiments (40 configurations)
bash scripts/run_all_experiments.sh
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--dataset` | MNIST, FashionMNIST, CIFAR10, CIFAR100 | MNIST |
| `--model` | bls or arbn | bls |
| `--data_root` | Dataset root directory | ./data |
| `--imbalance_factor` | Long-tail imbalance factor (1=balanced) | 1 |
| `--feature_times` | Number of feature node groups | 10 |
| `--enhance_times` | Number of enhancement node groups | 10 |
| `--feature_size` | Size per feature node group (or "auto") | 256 |
| `--mapping_func` | Feature mapping activation | linear |
| `--enhance_func` | Enhancement node activation | relu |
| `--reg` | Regularization parameter λ | 0.005 |
| `--class_weight_beta` | ARBN weight exponent β | 0.5 |
| `--no_adaptive_reg` | Disable ARBN adaptive weighting | False |
| `--seed` | Random seed | 42 |
| `--storing` | Save model after training | False |

## Experimental Results

所有实验使用默认参数: `feature_times=10, enhance_times=10, feature_size=256, reg=0.01, seed=42`。
不平衡因子 (IF) 为 1 表示平衡数据集，数值越大不平衡程度越高。

### MNIST (10 classes)

| IF | BLS Acc (%) | ARBN Acc (%) |
| -: | :-: | :-: |
| 1 | 96.57 | 95.12 |
| 10 | 94.59 | 94.51 |
| 50 | 88.84 | 91.90 |
| 100 | 84.18 | 89.55 |
| 200 | 78.26 | 85.67 |

### Fashion-MNIST (10 classes)

| IF | BLS Acc (%) | ARBN Acc (%) |
| -: | :-: | :-: |
| 1 | 86.93 | 85.40 |
| 10 | 84.74 | 85.19 |
| 50 | 80.97 | 82.95 |
| 100 | 77.61 | 80.81 |
| 200 | 72.89 | 76.66 |

### CIFAR-10 (10 classes)

| IF | BLS Acc (%) | ARBN Acc (%) |
| -: | :-: | :-: |
| 1 | 46.79 | 45.71 |
| 10 | 32.11 | 36.67 |
| 50 | 25.31 | 28.36 |
| 100 | 23.79 | 27.17 |
| 200 | 22.62 | 24.91 |

### CIFAR-100 (100 classes)

| IF | BLS Acc (%) | ARBN Acc (%) |
| -: | :-: | :-: |
| 1 | 21.39 | 20.83 |
| 10 | 13.05 | 15.43 |
| 50 | 8.73 | 10.74 |
| 100 | 8.18 | 9.63 |
| 200 | 6.83 | 8.59 |


## Project Structure

```
BLS/
├── loader/
│   ├── data_loader.py      # Dataset loading with long-tail support
│   └── model_loader.py     # Model serialization
├── models/
│   ├── __init__.py          # Model registry
│   ├── bls.py               # Standard BLS model
│   └── arbn.py              # Adaptive Re-weighting Broad Network
├── scripts/
│   ├── run_all_experiments.sh  # Run all 40 experiment configs
│   ├── run_single.sh           # Run a single experiment
│   └── collect_results.py      # Parse logs and generate table
├── main.py                  # Main training script
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
└── README.md
```

## License

MIT License
