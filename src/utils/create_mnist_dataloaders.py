from torch.utils.data import DataLoader

from src.utils.io import load_MNIST
from src.utils.mnist_dataset import MNISTDataset


def create_mnist_dataloaders(
    mnist_dir_path: str,
    chunk_size: int,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_images, train_labels, test_images, test_labels = load_MNIST(mnist_dir_path)

    train_dataset = MNISTDataset(train_images, train_labels, chunk_size)
    test_dataset = MNISTDataset(test_images, test_labels, chunk_size)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
    )
    return train_dataloader, test_dataloader
