"""
VocabularyManager module for managing parts, colors, and rotations in atlas_config.json.

This module provides a clean interface to update and retrieve vocabulary data
without storing continuous variables (rotations, positions) in memory.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
import numpy as np

from ..config import Config
from ..maths.rotations import generate_chiral_rotation_matrices
from ..utils.logging import setup_logging

logger = setup_logging()


class VocabularyManager:
    """
    Manages the vocabulary for parts, colors, and rotations in atlas_config.json.

    This class follows the Single Responsibility Principle by handling only
    vocabulary-related operations. Rotations are calculated on-the-fly and not stored.
    """

    def __init__(self, atlas_config_path: str):
        """
        Initialize the VocabularyManager.

        Args:
            atlas_config_path: Path to the atlas_config.json file.
        """
        self.config_path = Path(atlas_config_path)
        self.config_data: Dict = {}
        self._load_or_initialize_config()

    def _load_or_initialize_config(self) -> None:
        """
        Load existing configuration or initialize a new one.

        Creates the config file with default structure if it doesn't exist.
        Ensures rotations are populated even in existing configs.
        """

        if not self.config_path.exists():
            logger.warning(f"{self.config_path} does not exist. Creating a new config.")
            self._initialize_default_config()
            self._save_config()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")

            # Check if rotations are empty and populate them if needed
            if not self.config_data.get("vocabulary", {}).get("rotations"):
                logger.info("Rotations empty in existing config. Populating...")
                rotations = self._generate_rotation_vocabulary()
                self.config_data["vocabulary"]["rotations"] = rotations
                self.config_data["offsets"]["rotations"] = Config.OFFSETS["rotations"]
                self.config_data["offsets"]["colors"] = Config.OFFSETS["colors"]
                self.config_data["offsets"]["parts"] = Config.OFFSETS["parts"]
                self._save_config()
                logger.info("Rotations populated and config updated")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {self.config_path}: {e}")
            self._initialize_default_config()
            self._save_config()

    def _initialize_default_config(self) -> None:
        """Initialize configuration with default structure."""
        rotations = self._generate_rotation_vocabulary()

        # maybe move this dict to a constant in config.py
        self.config_data = {
            "version": "1.0",
            "spatial": {
                "l_min": Config.MIN_POSITION,
                "l_max": Config.MAX_POSITION,
                "step": Config.PRECISION,
                "num_bins": Config.NUM_BINS_PER_AXIS,
            },
            "offsets": {
                "special": Config.OFFSETS["special"],  # 4 special tokens
                "rotations": Config.OFFSETS["rotations"],  # 24 rotations
                "positions_x": Config.OFFSETS["positions_x"],  # 1000 position bins
                "positions_y": Config.OFFSETS["positions_y"],  # 1000 position bins
                "positions_z": Config.OFFSETS["positions_z"],  # 1000 position bins
                "colors": Config.OFFSETS["colors"],  # starting after positions
                "parts": Config.OFFSETS["parts"],  # starting after colors
            },
            "vocabulary": {
                "special": {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3},
                "rotations": rotations,
                "parts": {},
                "colors": {},
                # Note: positions are not stored in vocabulary because the bins can be calculated
            },
            "vocab_size": Config.OFFSETS["parts"],
        }
        logger.info("Initialized default configuration")

    def _generate_rotation_vocabulary(self) -> Dict[str, int]:
        """
        Generate the 24 chiral rotations vocabulary.

        Returns:
            Dictionary mapping rotation matrix strings to indices.
        """
        rotations = generate_chiral_rotation_matrices()
        rotation_vocab = {}

        for idx, rot_matrix in enumerate(rotations):
            # Create a unique key for each rotation matrix
            # Flatten the matrix and create a string representation
            key = "[" + ",".join(f"{val:.6f}" for val in rot_matrix.flatten()) + "]"
            rotation_vocab[key] = idx

        logger.debug(f"Generated {len(rotation_vocab)} rotation entries")
        return rotation_vocab

    def _recalculate_offsets(self) -> None:
        """Recalculate parts offset and vocab_size from current vocabulary state."""
        colors_offset = self.config_data["offsets"]["colors"]
        num_colors = len(self.config_data["vocabulary"]["colors"])
        num_parts = len(self.config_data["vocabulary"]["parts"])
        self.config_data["offsets"]["parts"] = colors_offset + num_colors
        self.config_data["vocab_size"] = colors_offset + num_colors + num_parts

    def _save_config(self) -> None:
        """Save the current configuration to the JSON file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")

    def save(self) -> None:
        """Explicitly save the current configuration to disk."""
        self._recalculate_offsets()
        self._save_config()
        logger.info(f"Vocabulary saved to {self.config_path}")

    def add_part(self, part_id: str) -> int:
        """
        Add a new part to the vocabulary if it doesn't exist.

        Args:
            part_id: The unique identifier for the part (e.g., "3001.dat").

        Returns:
            The index assigned to the part.
        """
        try:
            parts = self.config_data["vocabulary"]["parts"]
        except KeyError:
            logger.warning("Parts vocabulary missing, initializing.")
            parts = {}
            self.config_data["vocabulary"]["parts"] = parts

        if part_id not in parts:
            parts[part_id] = len(parts)
            self._recalculate_offsets()

        return parts[part_id]

    def add_parts(self, part_ids: Set[str]) -> Dict[str, int]:
        """
        Add multiple parts to the vocabulary.

        Args:
            part_ids: Set of part identifiers to add.

        Returns:
            Dictionary mapping part IDs to their indices.
        """

        result = {}

        for part_id in sorted(part_ids):  # Sort for consistency
            result[part_id] = self.add_part(part_id)

        return result

    def add_color(self, color: int) -> int:
        """
        Add a new color to the vocabulary if it doesn't exist.

        Args:
            color: The color code (LDraw color ID).

        Returns:
            The index assigned to the color.
        """

        try:
            colors = self.config_data["vocabulary"]["colors"]
        except KeyError:
            logger.warning("Colors vocabulary missing, initializing.")
            colors = {}
            self.config_data["vocabulary"]["colors"] = colors

        color_key = str(color)

        if color_key not in colors:
            colors[color_key] = len(colors)
            self._recalculate_offsets()

        return colors[color_key]

    def add_colors(self, color_ids: Set[int]) -> Dict[str, int]:
        """
        Add multiple colors to the vocabulary.

        Args:
            color_ids: Set of color codes to add.

        Returns:
            Dictionary mapping color IDs (as strings) to their indices.
        """
        result = {}

        for color in sorted(color_ids):  # Sort for consistency
            result[str(color)] = self.add_color(color)

        return result

    def get_part_index(self, part_id: str) -> int:
        """
        Get the index of a part in the vocabulary.

        Args:
            part_id: The part identifier.

        Returns:
            The index of the part, or the UNK token index (3) if not found.
        """
        return self.config_data["vocabulary"]["parts"].get(part_id, 3)

    def get_color_index(self, color: int) -> int:
        """
        Get the index of a color in the vocabulary.

        Args:
            color: The color code.

        Returns:
            The index of the color, or the UNK token index (3) if not found.
        """
        return self.config_data["vocabulary"]["colors"].get(str(color), 3)

    def get_rotation_index(self, rotation_matrix: np.ndarray) -> int:
        """
        Get the index of the closest matching rotation.

        This method calculates which of the 24 chiral rotations best matches
        the given rotation matrix.

        Args:
            rotation_matrix: Rotation matrix of shape (3, 3).

        Returns:
            The index of the closest rotation (0-23).
        """
        rotations = generate_chiral_rotation_matrices()

        # Find the closest rotation by comparing Frobenius norm
        min_distance = float("inf")
        best_idx = 0

        for idx, rot_mat in enumerate(rotations):
            # Matrix distance using Frobenius norm
            dist = np.linalg.norm(rotation_matrix - rot_mat, "fro")

            if dist < min_distance:
                min_distance = dist
                best_idx = idx

        return best_idx

    def get_special_token_index(self, token: str) -> int:
        """
        Get the index of a special token.

        Args:
            token: The special token name (PAD, SOS, EOS, or UNK).

        Returns:
            The index of the special token.

        Raises:
            KeyError: If the token is not found.
        """
        return self.config_data["vocabulary"]["special"][token]

    def get_vocab_size(self) -> int:
        """
        Get the current vocabulary size.

        Returns:
            The total number of tokens in the vocabulary.
        """
        return self.config_data["vocab_size"]

    def get_parts_count(self) -> int:
        """
        Get the number of parts in the vocabulary.

        Returns:
            The number of unique parts.
        """
        return len(self.config_data["vocabulary"]["parts"])

    def get_colors_count(self) -> int:
        """
        Get the number of colors in the vocabulary.

        Returns:
            The number of unique colors.
        """
        return len(self.config_data["vocabulary"]["colors"])

    def get_offsets(self) -> Dict[str, int]:
        """
        Get the offset values for different vocabulary sections.

        Returns:
            Dictionary containing offset values.
        """
        return self.config_data["offsets"].copy()

    def get_all_parts(self) -> Dict[str, int]:
        """
        Get all parts in the vocabulary.

        Returns:
            Dictionary mapping part IDs to their indices.
        """
        return self.config_data["vocabulary"]["parts"].copy()

    def get_all_colors(self) -> Dict[str, int]:
        """
        Get all colors in the vocabulary.

        Returns:
            Dictionary mapping color IDs to their indices.
        """
        return self.config_data["vocabulary"]["colors"].copy()
