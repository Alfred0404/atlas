import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to pad sequences in a batch to the same length and create input-target pairs.
    Args:
        batch (list[torch.Tensor]): List of sequence tensors.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Padded batch tensors (input and target).
    """

    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)

    # prepare input and target sequences for next-token prediction
    x = padded_batch[:, :-1]  # All tokens except the last
    y = padded_batch[:, 1:]  # All tokens except the first

    return x, y


class SequenceDataset(Dataset):
    """Class for loading a dataset from .npy files and converting them into sequences."""

    def __init__(self, npy_dir: str, max_bricks: int = 400, max_files: int = 10):
        """
        Args:
            npy_dir (str): Directory containing .npy files.
            max_bricks (int, optional): Maximum number of bricks to consider from each file. Defaults to 400.
            max_files (int, optional): Maximum number of .npy files to load. Defaults to 10.
        """
        self.max_files = max_files
        self.max_bricks = max_bricks
        self.npy_files = [
            os.path.join(npy_dir, file)
            for file in os.listdir(npy_dir)
            if file.endswith(".npy")
        ][: self.max_files]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get the sequence tensor for the .npy file at the given index.
        Args:
            index (int): Index of the .npy file.
            Returns:
                torch.Tensor: Sequence tensor with start and end tokens.
        """
        matrix = np.load(self.npy_files[index])

        if len(matrix) > self.max_bricks:
            matrix = matrix[: self.max_bricks]

        sequence = matrix.flatten()

        full_sequence = np.concatenate(
            [[1], sequence, [2]]
        )  # Add start and end tokens to the sequence

        return torch.tensor(full_sequence, dtype=torch.long)

if __name__ == "__main__":
    # Example usage
    dataset = SequenceDataset(npy_dir="./tokenized_sets", max_bricks=400, max_files=10)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )

    for batch_idx, (x, y) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("Input (x):", x)
        print("Target (y):", y)
        break