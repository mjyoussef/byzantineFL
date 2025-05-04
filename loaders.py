import numpy as np
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def _dataloaders_from(train, test, batch_size, num_clients, iid, alpha):
    print(num_clients)
    client_train_datasets = _load_splits(train, num_clients, iid, alpha)
    client_test_datasets = _load_splits(test, num_clients, iid, alpha)

    train_loaders, test_loaders = [], []
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

    for client in range(num_clients):
        print(len(train_loaders[client]))

    return train_loaders, test_loaders


def _load_splits(dataset, num_clients, iid, alpha, seed=7):
    np.random.seed(seed)

    targets = np.array([data[1] for data in dataset])
    num_classes = len(np.unique(targets))
    if iid:
        client_dists = np.full((num_classes, num_clients), 1 / num_clients)
    else:
        client_dists = np.random.dirichlet([alpha] * num_clients, num_classes)
    client_indices = defaultdict(list)

    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        props = client_dists[cls]
        props = np.cumsum(props * len(cls_indices)).astype(int)[:-1]
        splits = np.split(cls_indices, props)

        for client in range(num_clients):
            client_indices[client].extend(list(splits[client]))

    client_datasets = [
        Subset(dataset, client_indices[client]) for client in range(num_clients)
    ]

    return client_datasets


def load_cifar(batch_size, num_clients, iid, alpha):
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
        cifar_train, cifar_test, batch_size, num_clients, iid, alpha
    )


def load_mnist(batch_size, num_clients, iid, alpha):
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
        mnist_train, mnist_test, batch_size, num_clients, iid, alpha
    )


# if __name__ == "__main__":
#     load_cifar(64, 20, False, 0.5)
