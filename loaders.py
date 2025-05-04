import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from typing import List


def _dataloaders_from(
    train: Dataset,
    test: Dataset,
    batch_size: int,
    num_clients: int,
    iid: bool,
    alpha: float,
    root_data: bool,
) -> tuple[List[DataLoader], List[DataLoader], None | DataLoader]:

    # Root dataset is not needed for testing
    client_train_datasets = _load_splits(train, num_clients, iid, alpha, root_data)
    client_test_datasets = _load_splits(test, num_clients, iid, alpha, False)

    train_loaders, test_loaders = [], []

    # Create train & test dataloaders for each client (not including the server's root dataset)
    for client in range(num_clients):
        train_loaders.append(
            DataLoader(
                client_train_datasets[client],
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
        )

        test_loaders.append(
            DataLoader(
                client_test_datasets[client],
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
            )
        )

    # Root dataset is last if `root_data` is true
    root_train_loader = None
    if root_data:
        root_train_loader = train_loaders[num_clients]

    for client in range(num_clients):
        print(len(train_loaders[client]))

    return train_loaders, test_loaders, root_train_loader


def _load_splits(
    dataset: Dataset,
    num_clients: int,
    iid: bool,
    alpha: float,
    root_data: bool,
    root_data_pr=0.15,
    seed=7,
) -> List[Subset]:

    np.random.seed(seed)
    targets = np.array([data[1] for data in dataset])
    num_classes = len(np.unique(targets))

    if iid:
        # Evenly distributed
        client_dists = np.full((num_classes, num_clients), (1 / num_clients))
    else:
        # Dirichlet sample
        client_dists = np.random.dirichlet([alpha] * num_clients, num_classes)

    # If the server requires a root dataset, allocate `root_data_pr` samples from each class
    if root_data:
        client_dists = client_dists * (1 - root_data_pr)
        server_probs = np.array([root_data_pr] * num_classes).reshape(-1, 1)
        client_dists = np.hstack((client_dists, server_probs))

    # Map clients to list of indices in original dataset
    client_indices = defaultdict(list)

    total_clients = num_clients + 1 if root_data else num_clients

    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        probs = client_dists[cls]
        probs = np.cumsum(probs * len(cls_indices)).astype(int)[:-1]
        splits = np.split(cls_indices, probs)

        for client in range(total_clients):
            client_indices[client].extend(list(splits[client]))

    client_datasets = [
        Subset(dataset, client_indices[client]) for client in range(total_clients)
    ]

    return client_datasets


def load_cifar(
    batch_size: int, num_clients: int, iid: bool, alpha: float, root_data: bool
) -> tuple[List[DataLoader], List[DataLoader], None | DataLoader]:
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar_train = sorted(
        datasets.CIFAR10(
            root="./data", train=True, download=True, transform=cifar_transform
        ),
        key=lambda x: x[1],
    )
    cifar_test = sorted(
        datasets.CIFAR10(
            root="./data", train=False, download=True, transform=cifar_transform
        ),
        key=lambda x: x[1],
    )

    return _dataloaders_from(
        cifar_train, cifar_test, batch_size, num_clients, iid, alpha, root_data
    )


def load_mnist(
    batch_size: int, num_clients: int, iid: bool, alpha: float, root_data: bool
) -> tuple[List[DataLoader], List[DataLoader], None | DataLoader]:
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=mnist_transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=mnist_transform
    )

    return _dataloaders_from(
        mnist_train, mnist_test, batch_size, num_clients, iid, alpha, root_data
    )
