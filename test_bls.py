import warnings
import argparse
import numpy as np
from bls import BLS

from data.load_data import get_dataset
from utils import valid_model, extract_data

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Variational AutoEncoder')

parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--feature_times', type=int, default=10)
parser.add_argument('--enhance_times', type=int, default=10)
parser.add_argument('--feature_size', type=int, default=256)
parser.add_argument('--mapping_func', type=str, default='linear')
parser.add_argument('--enhance_func', type=str, default='relu')
parser.add_argument('--reg', type=float, default=0.01, help='')
parser.add_argument('--sig', type=float, default=0.01, help='')

parser.add_argument('--enhance_epoch', type=int, default=5)
parser.add_argument('--enhance_nodes', type=int, default=10)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--sparse', '-s', action='store_true', default=False)

args = parser.parse_args()

# Set random seeds for reproducibility
np.random.seed(args.seed)

print(f'> Loading the dataset {args.dataset} ...', end=' ')
train_loader, valid_loader, n_classes = get_dataset(args.dataset)
X_train, y_train = extract_data(train_loader)  # get train data and labels
X_valid, y_valid = extract_data(valid_loader)  # get valid data and labels
print(f'Success\n')

# Initialize BLS model
print('> Initialize the BLS model ...')
model = BLS(
    feature_times=args.feature_times,
    enhance_times=args.enhance_times,
    feature_size=args.feature_size,  # 'auto'
    n_classes=n_classes,
    mapping_function=args.mapping_func,
    enhance_function=args.enhance_func,
    reg=args.reg,
    sig=args.sig,
    use_sparse=args.sparse,
)


# Train the model
print("> Training BLS model ...")
model.fit(X_train, y_train)
print(f'  Finish training\n')

# Evaluate the model on train set
print("> Testing BLS model in train set ...")
accuracy = valid_model(model, X_train, y_train)
print(f"  *Accuracy of train set: {accuracy:.2f}%\n")

# Evaluate the model on valid set
print("> Testing BLS model in test set ...")
accuracy = valid_model(model, X_valid, y_valid)
print(f"  *Accuracy of valid set: {accuracy:.2f}%\n\n")


for _ in range(args.enhance_epoch):
    # add enhancement nodes
    model.add_enhancement_nodes(X_train, y_train, args.enhance_nodes)
    print()

    # Evaluate the model on train set
    print("> Testing BLS model in train set ...")
    accuracy = valid_model(model, X_train, y_train)
    print(f"  *Accuracy of train set: {accuracy:.2f}%\n")

    # Evaluate the model on valid set
    print("> Testing BLS model in test set ...")
    accuracy = valid_model(model, X_valid, y_valid)
    print(f"  *Accuracy of valid set: {accuracy:.2f}%\n\n")

