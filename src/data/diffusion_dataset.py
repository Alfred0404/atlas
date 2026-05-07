import torch
from torch.utils.data import Dataset
from pathlib import Path


class DiffusionDataset(Dataset):
    """Dataset of padded (N, 6) LEGO set tensors for diffusion training.

    Each item is a dict:
        positions:    (N, 3) float32, normalized and centered
        rot_ids:      (N,) int64, indices into 24 octahedral rotations
        part_ids:     (N,) int64, 1-indexed (0 = UNK/pad)
        color_ids:    (N,) int64, 1-indexed (0 = UNK/pad)
        padding_mask: (N,) bool, True = padded brick
        n_bricks:     int, number of real bricks
    """

    def __init__(self, data_dir: str, max_sets: int = None):
        self.files = sorted(Path(data_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        if max_sets is not None:
            self.files = self.files[:max_sets]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], weights_only=True)
