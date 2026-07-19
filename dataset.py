import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_dir="./data"):
    """Download MNIST and return train dataset + test DataLoader."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_dataset, test_loader


def split_clients(train_dataset, num_clients=3, seed=42):
    """IID split: shuffle once, partition into equal contiguous chunks."""
    n = len(train_dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    shard_size = n // num_clients
    client_datasets = []
    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else n
        client_datasets.append(Subset(train_dataset, indices[start:end]))
    return client_datasets
