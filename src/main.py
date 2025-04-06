import torch


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)


if __name__ == "__main__":
    main()