from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def dataloaders_from(train, test, batch_size=32, num_workers=2):
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def load_cifar(batch_size=64):
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar_train = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=cifar_transform
    )
    cifar_test = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=cifar_transform
    )

    return dataloaders_from(cifar_train, cifar_test)


def load_mnist(batch_size=64):
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=mnist_transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=mnist_transform
    )

    return dataloaders_from(mnist_train, mnist_test, batch_size=batch_size)
