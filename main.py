import argparse
from loaders import dataloaders_from
from model.cct import cct_6_3x1_32
import torch


def main():
    pass


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--data", type=("cifar10", "mnist"), required=True)
    parser.add_argument("--iid", type=bool, required=True)
    parser.add_argument("--K", type=bool, required=True, help="Number of clients")
    parser.add_argument(
        "--M", option=bool, required=True, help="Number of byzantine clients"
    )
    parser.add_argument(
        "--algorithm", type=("FedAvg", "Krum", "AutoGM", "TrimmedMean"), required=True
    )

    # Offload to GPU (if possible)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main()
