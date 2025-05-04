import yaml
from loaders import load_cifar, load_mnist
from model.cct import cct_6_3x1_32
import torch


def train():
    pass


if __name__ == "__main__":

    # Parse YAML
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Offload to GPU (if possible)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Select dataset
    train_loader, test_loader = (
        load_cifar() if (config["dataset"] == "cifar10") else load_mnist()
    )

    train()
