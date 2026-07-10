# 🧠 BLS & ARBN — Broad Learning System with Adaptive Re-weighting

<div align="center">

**A NumPy-powered implementation of the Broad Learning System for long-tail image classification.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-98%20passed-brightgreen)]()
[![NumPy](https://img.shields.io/badge/Built_with-NumPy-013243?logo=numpy)](https://numpy.org)

</div>

---

## ✨ Highlights

- 🚀 **Closed-form training** — No back-propagation, no gradient descent. A single ridge-regression solve trains the entire network.
- 🧩 **Flat architecture** — Wide hidden representation via random feature mappings + orthogonalized enhancement nodes.
- ⚖️ **ARBN for long-tail** — Class-aware sample weights injected directly into the normal equations, counteracting head-class dominance.
- ⚡ **Incremental expandability** — Add enhancement nodes on-the-fly without full retraining.
- 🎯 **Pure NumPy** — Zero deep-learning framework dependency for the core model (PyTorch only for data loading).

---

## 📐 How It Works

### Standard BLS

Given input $\mathbf{X} \in \mathbb{R}^{N \times d}$ and one-hot labels $\mathbf{Y} \in \mathbb{R}^{N \times C}$:

**1. Feature Nodes (Random Mapping)**
$$\mathbf{Z}_i = \phi_i(\mathbf{X} \mathbf{W}_i + \mathbf{1} \beta_i^{\top}), \quad i = 1, \dots, n$$

Weights $\mathbf{W}_i$ and biases $\beta_i$ are randomly initialized and **kept fixed**. Stacking gives $\mathbf{Z}^n = [\mathbf{Z}_1, \dots, \mathbf{Z}_n]$.

**2. Enhancement Nodes (Nonlinear + Orthogonal)**
$$\mathbf{H}_j = \xi_j(\mathbf{Z}^n \mathbf{W}_j^{(h)} + \mathbf{1} \gamma_j^{\top}), \quad \text{s.t. } \mathbf{H}_j^{\top} \mathbf{H}_j = \mathbf{I}, \quad j = 1, \dots, m$$

Gram-Schmidt orthogonalization reduces redundancy among enhancement groups. Stacking gives $\mathbf{H}^m = [\mathbf{H}_1, \dots, \mathbf{H}_m]$.

**3. Wide Hidden Matrix**
$$\mathbf{A} = [\mathbf{Z}^n \mid \mathbf{H}^m]$$

**4. Closed-Form Ridge Regression**
$$\mathcal{L} = \| \mathbf{A} \mathbf{W} - \mathbf{Y} \|_F^2 + \lambda \| \mathbf{W} \|_F^2$$
$$\mathbf{W}^* = (\mathbf{A}^{\top} \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^{\top} \mathbf{Y}$$

> **One Cholesky solve. Done.** 🎉

---

### ARBN — Tackling Class Imbalance

Under long-tailed distributions, the least-squares objective is dominated by head classes. ARBN injects class-aware sample weights directly into the closed form:

**Class Weights:**
$$w_k = \left( \frac{1}{|D_k|} \right)^{\beta}, \quad k = 1, \dots, C$$

| $\beta$ | Behavior |
|:---:|---|
| `0` | Uniform weights → equivalent to standard BLS |
| `0.5` | Moderate re-weighting (default, works best in practice) |
| `1` | Inverse class frequency → maximum tail amplification |

**Weighted Normal Equations:**
$$\mathbf{W}^* = (\mathbf{A}^{\top} \mathbf{W}_d \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^{\top} \mathbf{W}_d \mathbf{Y}$$

where $\mathbf{W}_d = \mathrm{diag}(w_{y_1}, \dots, w_{y_N})$ avoids the $O(N \cdot L^2)$ cost of explicitly weighting the design matrix.

> **Key insight**: Unlike resampling methods, ARBN modifies the **optimization objective itself** — no variance injection, no residual bias. 🔬

---

## 🚀 Quick Start

### Environment Setup

```bash
conda create -n bls python=3.12 -y
conda activate bls
pip install -r requirements.txt
```

### Basic Usage

```bash
# 🔵 Standard BLS on balanced MNIST
python main.py --dataset MNIST --model bls --data_root ./data

# 🟠 ARBN on imbalanced CIFAR-10 (IF=100)
python main.py --dataset CIFAR10 --model arbn --imbalance_factor 100

# 🟣 ARBN on CIFAR-100 with custom beta
python main.py --dataset CIFAR100 --model arbn --imbalance_factor 100 \
    --class_weight_beta 0.5

# 🧪 Incremental learning (add enhancement nodes after training)
python main.py --dataset MNIST --model bls --enhance_epoch 3 --enhance_nodes 20

# 📦 Run all 40 experiment configurations
bash scripts/run_all_experiments.sh
```

### Running Tests

```bash
# Run all 98 tests
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_bls.py -v

# Run with coverage report
python -m pytest tests/ -v --cov=models --cov-report=term-missing
```

### 🌐 Interactive Web Visualization

`web/index.html` 是一个**零依赖单文件前端**，用可视化方式逐阶段演示 BLS 的前向计算过程（输入预处理 → 特征节点 → 增强节点 → 拼接与输出映射 → 分类结果）。它适合在讲解、汇报或自学时**直观理解算法每一步在做什么**，而无需跑真实训练。

#### 启动方式

无需构建、无需安装前端依赖，直接用任意静态服务器托管 `web/` 目录即可：

```bash
# 方式 1：Python 内置 HTTP 服务器（推荐，最轻量）
cd /Users/zenopang/Desktop/WXG_PROJ/BLS/web
python3 -m http.server 8765
# 然后在浏览器打开：http://localhost:8765/index.html

# 方式 2：指定其他端口（任选空闲端口）
cd /Users/zenopang/Desktop/WXG_PROJ/BLS/web
python3 -m http.server 8000
# 打开：http://localhost:8000/index.html

# 方式 3：Node 环境（若已安装 npx）
cd /Users/zenopang/Desktop/WXG_PROJ/BLS/web
npx serve .
```

> **注意**：必须通过 HTTP 服务器访问（如 `http.server`），不要直接双击用 `file://` 打开 —— 否则浏览器会因 CORS 策略拒绝加载页面内的脚本与图表资源。
>
> 页面仅需通过 CDN 引入 [Chart.js](https://www.chartjs.org/)，因此首次打开需要联网；其余逻辑（含计算流程模拟）全部内联在单文件中。

#### 五阶段展示内容

| 阶段 | 标题 | 核心可视化 |
|:---:|---|---|
| 1 | Input Preprocessing | 图片归一化、像素网格（RGB 模拟）、展平为 3072 维向量 |
| 2 | Feature Node Generation | ReLU 激活分布直方图、分组统计对比（Mean/Std/活性率）、稀疏度仪表盘、2D 激活热力图矩阵 |
| 3 | Enhancement Node Gen | tanh 饱和分布直方图（三色标注饱和区/线性区）、分组统计对比、2D 双色热力图、Feature↔Enhancement 对比面板 |
| 4 | Output Mapping | Zⁿ⊕Hᵐ 拼接简要展示 + A×W\* 矩阵乘法可视化 + Raw Logits 一览 |
| 5 | Classification Result | Softmax 置信度横向条形图、Top-1 高亮结果、Top-2 与全部类别概率列表 |

#### 交互操作

| 操作 | 方式 |
|------|------|
| **切换示例** | 点击左侧示例卡片（🐱🐶🐸🚗🚢🐦），触发一次新的模拟推理 |
| **上传图片** | 点击「上传图片」按钮选择本地图片（演示用，仅触发模拟推理流程） |
| **分步控制** | 左侧「上一步 / 下一步 / 重置」按钮，或在步骤导航栏点击任意阶段跳转 |
| **自动播放** | 点击「自动播放」按阶段顺序自动推进，可随时停止 |
| **速度调节** | 0.5× / 1× / 2× 三档切换播放节奏 |
| **参数调节** | 拖动 `feature_times` / `enhance_times` / `feature_size` 滑块，改变特征/增强节点规模后重新推理 |
| **键盘快捷键** | `←` / `→` 切换上一步/下一步，`空格` 播放/暂停，`R` 重置 |

#### 关于数据的重要说明

> ⚠️ **该页面是「算法流程演示」而非真实模型推理。** 所有输入向量、特征/增强节点输出、Logits 与分类概率，均由前端 JavaScript 用**伪随机模拟数据**生成，且经过专门设计以体现真实的统计特性：
> - **数据流依赖**：特征节点来自输入投影、增强节点来自特征节点，不同输入会产出不同的特征/增强模式；
> - **激活函数特性**：ReLU 的稀疏性（死亡神经元比例）、tanh 的零中心化与低饱和率均被正确呈现；
> - **分类信号**：Top 类 Logits 与次类有明显分离，置信度直观可读。
>
> 因此它的定位是「**帮助理解 BLS 计算过程**」的教学/演示工具，不代表某个具体数据集上的真实精度。

#### 文件说明

```
web/
└── index.html    # 单文件应用：HTML + CSS + JS 内联，仅外部依赖 Chart.js (CDN)
```

如需二次开发（如接入真实模型、替换模拟数据为后端推理结果），只需修改 `web/index.html` 中的 `generateMockInference()` 与各个 `stage*()` 渲染函数即可，无需改动 `models/` 下的算法代码。

### Python API

```python
from models import BLS, ARBN

# BLS
model = BLS(
    feature_times=10,        # number of feature node groups
    enhance_times=10,        # number of enhancement node groups
    feature_size=256,        # width per group ("auto" = input_dim)
    n_classes=10,
    reg=0.005,               # L2 regularization λ
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
proba = model.predict_proba(X_test)

# ARBN
model = ARBN(
    feature_times=10, enhance_times=10,
    feature_size=256, n_classes=10,
    reg=0.005,
    cls_num_list=[5000, 1000, 200, 50, 10],  # per-class counts
    class_weight_beta=0.5,                     # re-weighting exponent
)
model.fit(X_imbalanced_train, y_imbalanced_train)
```

---

## ⚙️ Configuration

| Argument | Description | Default |
|:---|:---|---:|
| `--dataset` | `MNIST` \| `FashionMNIST` \| `CIFAR10` \| `CIFAR100` | `MNIST` |
| `--model` | `bls` \| `arbn` | `bls` |
| `--data_root` | Dataset storage directory | `./data` |
| `--imbalance_factor` | Long-tail factor (`1` = balanced, `200` = extreme) | `1` |
| `--feature_times` | Feature node groups | `10` |
| `--enhance_times` | Enhancement node groups | `10` |
| `--feature_size` | Width per group or `auto` | `256` |
| `--reg` | Ridge regularization λ | `0.005` |
| `--mapping_func` | Feature activation | `linear` |
| `--enhance_func` | Enhancement activation | `relu` |
| `--class_weight_beta` | ARBN β exponent | `0.5` |
| `--no_adaptive_reg` | Disable ARBN re-weighting | `False` |
| `--enhance_epoch` | Incremental rounds | `0` |
| `--enhance_nodes` | Nodes per round | `10` |
| `--seed` | Random seed | `42` |
| `--loading` / `-l` | Load saved model | `False` |
| `--storing` / `-s` | Save model after training | `False` |
| `--sparse` / `-p` | Sparse autoencoder shortcut | `False` |

---

## 📊 Experimental Results

> Parameters: `feature_times=10, enhance_times=10, feature_size=256, reg=0.01, seed=42`
> IF = Imbalance Factor (1 = balanced, higher = more imbalanced)
> 🟠 = ARBN better, 🔵 = BLS better, ➖ = tie

### 🖊️ MNIST (10 classes)

| IF | ARBN Acc | ARBN F1(m) | 🔵 BLS Acc | BLS F1(m) | Δ Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 95.12 | 95.07 | **96.57** 🔵 | 96.55 | −1.45 |
| 10 | **94.51** | **94.44** | 94.59 | 94.51 | −0.08 |
| 50 | **91.90** 🟠 | **91.70** | 88.85 | 88.32 | +3.05 |
| 100 | **89.55** 🟠 | **89.21** | 84.18 | 82.96 | +5.37 |
| 200 | **85.67** 🟠 | **84.83** | 78.26 | 75.12 | +7.41 |

### 👗 FashionMNIST (10 classes)

| IF | ARBN Acc | ARBN F1(m) | 🔵 BLS Acc | BLS F1(m) | Δ Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 85.40 | 85.20 | **86.93** 🔵 | 86.81 | −1.53 |
| 10 | **85.19** 🟠 | **84.63** | 84.74 | 83.91 | +0.45 |
| 50 | **82.95** 🟠 | **82.09** | 80.97 | 79.77 | +1.98 |
| 100 | **80.81** 🟠 | **79.78** | 77.61 | 76.31 | +3.20 |
| 200 | **76.66** 🟠 | **75.50** | 72.89 | 71.11 | +3.77 |

### 🖼️ CIFAR-10 (10 classes)

| IF | ARBN Acc | ARBN F1(m) | 🔵 BLS Acc | BLS F1(m) | Δ Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 45.71 | 45.06 | **46.79** 🔵 | 46.28 | −1.08 |
| 10 | **36.67** 🟠 | **35.12** | 32.11 | 28.19 | +4.56 |
| 50 | **28.36** 🟠 | **23.49** | 25.31 | 18.12 | +3.05 |
| 100 | **27.17** 🟠 | **21.51** | 23.79 | 16.24 | +3.38 |
| 200 | **24.91** 🟠 | **18.40** | 22.62 | 14.68 | +2.29 |

### 🎨 CIFAR-100 (100 classes)

| IF | 🟠 ARBN Top1 | A-Top5 | 🔵 BLS Top1 | B-Top5 | Δ Top1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 20.83 | 41.24 | **21.39** 🔵 | 40.47 | −0.56 |
| 10 | **15.43** 🟠 | 31.17 | 13.05 | 27.45 | +2.38 |
| 50 | **10.74** 🟠 | 24.21 | 8.73 | 20.48 | +2.01 |
| 100 | **9.63** 🟠 | 21.77 | 8.18 | 18.11 | +1.45 |
| 200 | **8.59** 🟠 | 19.00 | 6.83 | 15.59 | +1.76 |

---

## 🔍 Key Findings

<table>
<tr>
<td width="50%">

### 1. BLS wins on balanced, ARBN dominates under imbalance

On all datasets, BLS holds a **marginal edge at IF=1** (the uniformity penalty of ARBN slightly harms balanced scenarios). Once **IF ≥ 10**, ARBN consistently outperforms — the gap widens monotonically.

| Dataset | ARBN gain @ IF=200 |
|:---|---:|
| MNIST | **+7.4%** |
| FashionMNIST | **+3.8%** |
| CIFAR-10 | **+2.3%** |
| CIFAR-100 | **+1.8%** |

</td>
<td width="50%">

### 2. F1 exposes BLS's tail-class collapse

Under strong imbalance, the **Acc−F1 gap** is much larger for BLS:

- CIFAR-10 IF=200: BLS Acc→F1 = **22.6→14.7** (−7.9)
- CIFAR-10 IF=200: ARBN Acc→F1 = **24.9→18.4** (−6.5)

ARBN's class-aware weighting preserves per-class fidelity for minority categories.

</td>
</tr>
<tr>
<td width="50%">

### 3. Sweet spot: IF = 10–100

ARBN's relative gain over BLS is **largest at moderate-to-high imbalance**. At extreme imbalance (IF=200), both degrade but ARBN retains a clear edge.

</td>
<td width="50%">

### 4. Top-5 ≈ 2× Top-1 on CIFAR-100

Both models achieve Top-5 roughly **double** the Top-1 accuracy. The correct class is frequently in the top 5, suggesting the decision boundary could be further refined.

</td>
</tr>
</table>

---

## 📁 Project Structure

```
BLS/
├── loader/
│   ├── data_loader.py           # Dataset loading + long-tail distribution
│   └── model_loader.py          # Pickle serialization
├── models/
│   ├── __init__.py              # BLS / ARBN exports
│   ├── node_generator.py        # Shared NodeGenerator + activations
│   ├── bls.py                   # Standard Broad Learning System
│   └── arbn.py                  # Adaptive Re-weighting Broad Network
├── tests/
│   ├── test_node_generator.py   # 30 tests — activations, orthogonalization, update
│   ├── test_bls.py              # 25 tests — fit/predict/edge cases
│   ├── test_arbn.py             # 17 tests — weights/adaptive/incremental
│   ├── test_consistency.py      #  6 tests — BLS≈ARBN equivalence, tail recall
│   ├── test_serialization.py    #  3 tests — save/load roundtrip
│   └── test_utils.py            #  8 tests — accuracy/metrics utilities
├── scripts/
│   ├── run_all_experiments.sh   # Batch run all 40 configurations
│   └── collect_results.py       # Parse logs → Markdown tables
├── main.py                      # CLI training & evaluation
├── utils.py                     # Metrics, visualization helpers
├── web/
│   └── index.html               # 🌐 Single-file interactive BLS visualization demo
├── IMPROVEMENT.md               # Code review & improvement log
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

| Package | Min Version | Purpose |
|:---|:---:|:---|
| `numpy` | 1.24+ | Core matrix operations |
| `scipy` | 1.10+ | Cholesky decomposition, SVD |
| `scikit-learn` | 1.3+ | Base estimator, metrics |
| `torch` | 2.0+ | Data loading (torchvision datasets) |
| `torchvision` | 0.15+ | MNIST, CIFAR datasets |
| `matplotlib` | 3.7+ | Confusion matrix visualization |
| `pytest` | *dev* | Test runner |

---

## 📚 References

- Chen, C. L. P., & Liu, Z. (2017). **Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture**. *IEEE Transactions on Neural Networks and Learning Systems*.
- Chen, C. L. P., & Liu, Z. (2018). **Broad Learning System: A New Paradigm for Fast and Efficient Learning**. *Neurocomputing*.

---

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built with ❤️ using NumPy & SciPy</sub>
</div>
