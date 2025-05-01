from loaders import dataloaders_from
from model.cct import cct_6_3x1_32


def main():
    pass


if __name__ == "__main__":
    model = cct_6_3x1_32()
    print(sum(p.numel() for p in model.parameters()))
    main()
