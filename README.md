# Numpy Implementation of BLS (Broad Learning System)

## 1. Introduction to the Broad Learning System

The **Broad Learning System (BLS)** is an innovative machine learning framework that offers an efficient and effective alternative to traditional deep neural networks. Unlike deep learning architectures that rely on multiple layers of abstraction, BLS constructs a flat network structure by expanding nodes in the feature and enhancement spaces. This approach enables rapid learning with fewer computational resources while maintaining competitive performance on various tasks.

### Key Advantages of BLS:
1. **High Efficiency**: BLS can be trained much faster than deep learning models, making it suitable for real-time applications and resource-constrained environments.
2. **Incremental Learning**: It supports incremental training, allowing the model to adapt to new data without retraining from scratch.
3. **Structural Simplicity**: The flat architecture avoids the complexity of deep networks, reducing the risk of overfitting and simplifying interpretation.
4. **Lower Computational Requirements**: BLS operates efficiently on CPUs, eliminating the need for expensive GPU infrastructure.

### References:
[1] Chen, C. L. P., & Liu, Z. (2017). **Broad Learning System: An Effective and Efficient Incremental Learning System Without the Need for Deep Architecture**. *IEEE Transactions on Neural Networks and Learning Systems*, 29(1), 10-24.  

[2] Chen, C. L. P., & Liu, Z. (2020). **Research Review for Broad Learning System: Algorithms, Theory, and Applications**. *Information Sciences*, 504, 37-62.  


## 2. Project Objectives

This project aims to reproduce the Broad Learning System described in [1], implementing its core functionalities using a numpy-based framework. Our implementation focuses on:
- **Accuracy**: Replicating the original BLS algorithm with high fidelity.
- **Efficiency**: Optimizing performance for CPU-based execution.
- **Modularity**: Designing a flexible architecture that supports easy customization and extension.
- **Accessibility**: Providing clear documentation and examples for researchers and practitioners.


## 3. Project Structure and Usage

### Requirements
- Python 3.12+
- NumPy
- SciPy
- torch
- torchvision
- scikit-learn
- Optional: MNIST dataset (automatically downloaded if not found)

### File Structure
```
BLS/
├── data/                  # Place your datasets here
├── bls.py                 # BLS model definition
├── test_bls.py            # Test script for classification tasks
├── utils.py               # utility functions for BLS testing
├── requirements.txt       # Dependencies
└── README.md              # This documentation
```

### Model Initialization
The BLS model can be initialized with the following parameters:
```python
model = BLS(
    feature_times=args.feature_times,        # Number of feature mapping groups
    enhance_times=args.enhance_times,        # Number of enhancement node groups
    feature_size=args.feature_size,          # Size of each feature mapping (or 'auto')
    n_classes=n_classes,                     # Number of output classes
    mapping_function=args.mapping_func,      # Activation function for feature mapping
    enhance_function=args.enhance_func,      # Activation function for enhancement nodes
    reg=args.reg,                            # Regularization parameter
    sig=args.sig,                            # Scaling factor for input data
    use_sparse=args.sparse                   # Enable sparse matrix computation
)
```

### Running the Test Script
To test the BLS model on a classification task (e.g., MNIST), execute:
```bash
python test_bls.py
```

### Available Command-Line Arguments
```
--dataset          Dataset name (default: 'MNIST')
--feature_times    Number of feature mapping groups (default: 10)
--enhance_times    Number of enhancement node groups (default: 10)
--feature_size     Size of each feature mapping (default: 256)
--mapping_func     Mapping activation function (default: 'linear')
--enhance_func     Enhancement activation function (default: 'relu')
--reg              Regularization parameter (default: 0.01)
--sig              Scaling factor (default: 0.01)
--enhance_epoch    Number of enhancement training epochs (default: 5)
--enhance_nodes    Number of enhancement nodes to add per epoch (default: 10)
--seed             Random seed (default: 42)
--sparse, -s       Enable sparse matrix computation (default: False)
```


## 4. Example Usage

### Basic Classification Workflow
```python
# Initialize model
model = BLS(
    feature_times=10,
    enhance_times=10,
    feature_size=256,
    n_classes=10,
    mapping_function='linear',
    enhance_function='relu',
    reg=0.01,
    sig=0.01
)

# Train model
model.fit(X_train, y_train)

# Valid model
correct, total = 0, 0
predict = model.predict(X_train)
total += y_train.size
correct += (predict == y_train).sum().item()
accuracy = 100 * correct / total
print(f'acc = {accuracy:.4f}')
```


## 5. Contact Information

For questions, suggestions, or collaborations, please contact the project maintainer:  
**Email**: [202420143663@mail.scut.edu.cn]


## 6. License

This project is licensed under the [MIT License](LICENSE).


## 7. Acknowledgments

We would like to thank the authors of the original BLS papers for their pioneering work in developing this innovative learning system.