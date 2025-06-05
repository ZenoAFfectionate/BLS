import torch
import torchvision
from torchvision import transforms


def get_dataset(dataset_name="MNIST", batch_size=None):

    if dataset_name == "MNIST":
        n_classes = 10
        mean = 0.1307
        std  = 0.3081

        transform = transforms.Compose([
            # transforms.RandomRotation(15),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.3)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        train_set = torchvision.datasets.MNIST(root='./data', 
            train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', 
            train=False, download=True, transform=transform)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=batch_size if batch_size is not None else 60000,  # Full train set
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size if batch_size is not None else 10000,  # Full test set
            shuffle=False
        )

    if dataset_name == "CIFAR10":
        n_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2470, 0.2435, 0.2616]

        train_transform  = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.3)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', 
            train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', 
            train=False, download=True, transform=transform)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=batch_size if batch_size is not None else 50000,  # Full train set
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size if batch_size is not None else 10000,  # Full test set
            shuffle=False
        )

    if (dataset_name == 'CIFAR100'):
        n_classes = 100
        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2565, 0.2761]

        if batch_size == None:
            train_batch = 50000
            test_batch  = 10000

        train_transform  = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.3)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        train_set = torchvision.datasets.CIFAR100(root='./data', 
            train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', 
            train=False, download=True, transform=transform)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=batch_size if batch_size is not None else 50000,  # Full train set
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size if batch_size is not None else 10000,  # Full test set
            shuffle=False
        )

    return train_loader, test_loader, n_classes
