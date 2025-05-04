import yaml
from loaders import load_cifar, load_mnist
from model.cct import cct_6_3x1_32
import torch
from typing import Any


def train(config: dict[str, Any]):
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

    # Select dataset and distribute to clients
    train_loaders, test_loaders, root_dataset = (
        load_cifar(
            config["batch_size"],
            config["K"],
            config["iid"],
            config["alpha"],
            config["method"] == "TrustFedKD",
        )
        if (config["dataset"] == "cifar10")
        else load_mnist(
            config["batch_size"],
            config["K"],
            config["iid"],
            config["alpha"],
            config["method"] == "TrustFedKD",
        )
    )

    # Initialize models
    if config["method"] == "TrustFedKD":
        pass

    train(config)
