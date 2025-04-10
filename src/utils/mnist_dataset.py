import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    def __init__(self, images: np.ndarray, labels: np.ndarray, chunk_size: int):
        super().__init__()
        assert chunk_size > 0, "Chunk size must be more than 0"

        total_pixels = images.shape[-2] * images.shape[-1]

        assert (
            chunk_size > 0 and chunk_size <= total_pixels
        ), f"Chunk size must be more than 0 and less than or equal to {total_pixels}"

        self.images = (torch.from_numpy(images.copy()).float() / 255.0).view(
            -1, int(total_pixels / chunk_size), chunk_size
        )

        self.labels = torch.from_numpy(labels.copy()).long()
        self.classes = torch.unique(self.labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple:
        return self.images[idx], self.labels[idx]
