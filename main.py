import yaml
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from loaders import load_cifar, load_mnist
from model.cct import cct_2_3x2_32_sine, cct_4_3x2_32_sine
from typing import Any, Dict, Union, List, Tuple

# Offload to GPU (default to CPU)
DEVICE = None


def train(
    config: Dict[str, Any],
    student_model: Union[Module, None],
    client_models: List[Module],
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    root_loader: Union[DataLoader, None],
):
    # Track client losses and validation accuracy after each communication round
    client_loss = [[0 for _ in range(config["K"])] for _ in config["rounds"]]
    client_acc = [[0 for _ in range(config["K"])] for _ in config["rounds"]]

    # For each communication round
    for r in range(config["rounds"]):

        # Train student model if applicable
        # TODO

        # Train clients and track loss/acc
        # TODO

        # Aggregate
        # TODO

        pass


def train_local(
    config: Dict[str, Any],
    model: Module,
    train_loader: DataLoader,
    test_loader: Union[DataLoader, None],
) -> Tuple[float, float]:

    for e in range(config["epochs"]):
        pass

    return 0, 0


if __name__ == "__main__":

    # Parse YAML
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set device (global variable)
    if config["device"] == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif config["device" == "mps" and torch.backends.mps.is_available()]:
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # Select dataset and distribute to clients
    train_loaders, test_loaders, root_loader = (
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

    # Initialize models (same for CIFAR-10 and MNIST)
    client_models = [cct_4_3x2_32_sine() for _ in range(config["K"])]

    # Need a student model if using TrustFedKD
    student_model = cct_2_3x2_32_sine() if config["method"] == "TrustFedKD" else None

    train(config)
