import sys
from pathlib import Path
import json
import numpy as np

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from ..config import Config
from ..utils.logging import setup_logging
from ..maths.rotations import (
    find_closest_rotation_matrix,
    generate_chiral_rotation_matrices,
)

logger = setup_logging()


class AtlasTokenizer:
    """**Tokenizer for ATLAS model.**

    Handles **tokenization** *(encoding)* of brick attributes *(such as position, rotation, color, and part ID)* as well as the **detokenization** *(decoding)* process, to reconstruct brick attributes from tokens.
    """

    def __init__(self):
        """Initialize the tokenizer with empty vocabulary state."""
        self._vocabulary = None
        self._offsets = None
        self._reference_matrices = None
        self._rotation_vocab = None
        logger.info("Initialized AtlasTokenizer.")

    def load_vocabulary(self, config_path: str) -> None:
        """Load and cache vocabulary from atlas_config.json. Must be called before tokenizing."""
        with open(config_path, "r", encoding="utf-8") as f:
            atlas_config = json.load(f)
        self._vocabulary = atlas_config["vocabulary"]
        self._offsets = atlas_config["offsets"]
        self._rotation_vocab = atlas_config["vocabulary"]["rotations"]

        # Pre-parse rotation matrices from string keys
        self._reference_matrices = []
        self._rotation_keys = []
        for key, idx in sorted(self._rotation_vocab.items(), key=lambda x: x[1]):
            values = [float(x) for x in key.strip("[]").split(",")]
            self._reference_matrices.append(np.array(values).reshape(3, 3))
            self._rotation_keys.append(key)

        logger.info(f"Vocabulary cached: {len(self._vocabulary['parts'])} parts, "
                     f"{len(self._vocabulary['colors'])} colors, "
                     f"{len(self._reference_matrices)} rotations")

    def position_to_bin_id(self, position: float, axis: str) -> int:
        """Convert a real-valued position to its corresponding bin ID.
        Args:
            position (float): Real-valued position.
            axis (str): Axis of the position ('x', 'y', or 'z').
        Returns:
            int: Corresponding bin ID.
        """
        if Config.PRECISION == 0:
            raise ValueError("Config.PRECISION cannot be zero.")

        offset = Config.OFFSETS[f"positions_{axis}"]
        num_bins = (Config.MAX_POSITION - Config.MIN_POSITION) // Config.PRECISION
        raw_bin = int((position - Config.MIN_POSITION) / Config.PRECISION)
        bin_id = max(0, min(raw_bin, num_bins - 1)) + offset

        return bin_id

    def bin_id_to_position(self, bin_id: int, axis: str) -> float:
        """Convert a bin ID back to its corresponding real-valued position.
        Args:
            bin_id (int): Bin ID.
            axis (str): Axis of the position ('x', 'y', or 'z').
        Returns:
            float: Corresponding real-valued position.
        """
        position = (
            bin_id - Config.OFFSETS[f"positions_{axis}"]
        ) * Config.PRECISION + Config.MIN_POSITION

        return position

    def _get_token_from_vocabulary(
        self, value: str, vocab_key: str, offset_key: str
    ) -> int:
        """Helper method to get token ID from vocabulary with offset.
        Args:
            value (str): Value to look up in the vocabulary.
            vocab_key (str): Key in the vocabulary dictionary.
            offset_key (str): Key in the offsets dictionary.
        returns:
            int: Corresponding token ID.
        """
        if self._vocabulary is None:
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")

        vocab_dict = self._vocabulary[vocab_key]
        offset = self._offsets[offset_key]
        unknown_token_id = self._vocabulary["special"]["UNK"]

        token_id = vocab_dict.get(value, None)

        if token_id is None:
            logger.warning(f"Unknown {vocab_key[:-1]}: {value}")
            return unknown_token_id

        return token_id + offset

    def brick_id_to_token(self, brick_id: str) -> int:
        """Convert a brick ID to its corresponding token ID."""
        return self._get_token_from_vocabulary(
            value=brick_id, vocab_key="parts", offset_key="parts"
        )

    def color_id_to_token(self, color_id: str) -> int:
        """Convert a color ID to its corresponding token ID."""
        return self._get_token_from_vocabulary(
            value=color_id, vocab_key="colors", offset_key="colors"
        )

    def rotation_matrix_to_token(self, rotation_matrix: np.ndarray) -> int:
        """Convert a rotation matrix to its closest chiral matrix token ID."""
        if self._reference_matrices is None:
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")

        offset = self._offsets["rotations"]
        closest_index = find_closest_rotation_matrix(
            rotation_matrix, self._reference_matrices
        )
        vocab_index = self._rotation_vocab[self._rotation_keys[closest_index]]

        return vocab_index + offset

    # --- Batch methods for vectorized tokenization ---

    def batch_positions_to_bin_ids(self, positions: np.ndarray) -> np.ndarray:
        """Convert an (N, 3) array of positions to bin IDs for x, y, z.

        Args:
            positions: Array of shape (N, 3) with [x, y, z] per row.
        Returns:
            Array of shape (N, 3) with bin IDs including offsets.
        """
        offsets = np.array([
            Config.OFFSETS["positions_x"],
            Config.OFFSETS["positions_y"],
            Config.OFFSETS["positions_z"],
        ])
        num_bins = Config.NUM_BINS_PER_AXIS
        raw_bins = ((positions - Config.MIN_POSITION) / Config.PRECISION).astype(int)
        clamped = np.clip(raw_bins, 0, num_bins - 1)
        return clamped + offsets

    def batch_rotation_matrices_to_tokens(self, matrices: np.ndarray) -> np.ndarray:
        """Convert (N, 3, 3) rotation matrices to token IDs.

        Args:
            matrices: Array of shape (N, 3, 3).
        Returns:
            Array of shape (N,) with rotation token IDs.
        """
        if self._reference_matrices is None:
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")

        offset = self._offsets["rotations"]
        refs = np.array(self._reference_matrices)  # (24, 3, 3)

        # Compute Frobenius distances: (N, 24)
        # matrices[:, None] is (N, 1, 3, 3), refs[None] is (1, 24, 3, 3)
        diffs = matrices[:, None, :, :] - refs[None, :, :, :]
        distances = np.sum(diffs ** 2, axis=(2, 3))
        closest_indices = np.argmin(distances, axis=1)

        # Map indices to token IDs via vocabulary
        vocab_indices = np.array([
            self._rotation_vocab[self._rotation_keys[i]]
            for i in closest_indices
        ])
        return vocab_indices + offset

    def batch_brick_ids_to_tokens(self, brick_ids: list) -> np.ndarray:
        """Convert a list of brick IDs to token IDs.

        Args:
            brick_ids: List of N brick ID strings.
        Returns:
            Array of shape (N,) with part token IDs.
        """
        if self._vocabulary is None:
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")

        parts_vocab = self._vocabulary["parts"]
        offset = self._offsets["parts"]
        unk = self._vocabulary["special"]["UNK"]
        return np.array([
            parts_vocab.get(bid, None) + offset if parts_vocab.get(bid) is not None else unk
            for bid in brick_ids
        ])

    def batch_color_ids_to_tokens(self, color_ids: list) -> np.ndarray:
        """Convert a list of color IDs to token IDs.

        Args:
            color_ids: List of N color ID strings.
        Returns:
            Array of shape (N,) with color token IDs.
        """
        if self._vocabulary is None:
            raise RuntimeError("Vocabulary not loaded. Call load_vocabulary() first.")

        colors_vocab = self._vocabulary["colors"]
        offset = self._offsets["colors"]
        unk = self._vocabulary["special"]["UNK"]
        return np.array([
            colors_vocab.get(cid, None) + offset if colors_vocab.get(cid) is not None else unk
            for cid in color_ids
        ])


if __name__ == "__main__":
    tokenizer = AtlasTokenizer()

    test_position = 150.5
    # Convert position to bin ID
    bin_id = tokenizer.position_to_bin_id(test_position, "x")
    logger.debug(f"Position {test_position} maps to bin ID: {bin_id}")

    # Convert bin ID back to position
    recovered_position = tokenizer.bin_id_to_position(bin_id, "x")
    logger.debug(f"Bin ID {bin_id} maps back to position: {recovered_position}")

    # Convert brick ID to token ID
    tokenizer.brick_id_to_token("3001")
    # Convert color ID to token ID
    tokenizer.color_id_to_token("383")

    # Test rotation matrix to token ID
    test_rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tokenizer.rotation_matrix_to_token(test_rotation_matrix)
    logger.info("Generated Rotations (Rotation Matrices):")
