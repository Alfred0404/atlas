import sys
from pathlib import Path
from typing import NamedTuple, List, Set, Optional
import numpy as np
from tqdm import tqdm

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from .parser import MPDParser, RawBrickData
from ..core.vocabulary import VocabularyManager
from ..core.tokenizer import AtlasTokenizer
from ..maths.transforms import (
    get_position_from_world_matrix,
    get_rotation_matrix_from_world_matrix,
)
from ..utils.logging import setup_logging
from ..config import Config

logger = setup_logging()


class TokenizedBrickData(NamedTuple):
    """Training ready data with brick_id and color mapped to integers"""

    brick_idx: int  # mapped unique integer for brick_id
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_idx: int  # rotation token ID from vocabulary
    color_idx: int  # mapped unique integer for color


class DatasetBuilder:
    """Build and process LEGO datasets from MPD files.

    This class handles the complete pipeline from raw MPD files to processed
    tensor data, managing vocabulary through VocabularyManager, and the tokenization process through AtlasTokenizer.
    """

    def __init__(self, atlas_config_path: str):
        """
        Initialize the DatasetBuilder.

        Args:
            atlas_config_path: Path to the atlas_config.json file.
        """
        self.vocab_manager = VocabularyManager(atlas_config_path)
        self.tokenizer = AtlasTokenizer()
        self.all_brick_ids: Set[str] = set()
        self.all_colors: Set[int] = set()

        # Data storage for processing individual files
        self.raw_data: List[RawBrickData] = []
        self.tokenized_data: List[TokenizedBrickData] = []
        self.final_tensor = None

    def process_dataset(self, max_files: Optional[int] = None) -> None:
        """
        Process all MPD files in the raw dataset directory.

        Iterates through all .mpd files, parses them, transforms the data,
        and updates the vocabulary with encountered parts and colors.

        Args:
            max_files: Maximum number of files to process. If None, process all files.
        """
        logger.info("Starting dataset processing...\n")

        all_files = [
            f for f in Path(Config.RAW_DATASET_DIR).glob("*.mpd")
            if not (Path(Config.TOKENIZED_DATASET_DIR) / f.stem).with_suffix(".npy").exists()
        ]

        if not all_files:
            logger.error(f"No MPD files found in directory: {Config.RAW_DATASET_DIR}")
            return

        # Limit number of files if specified
        if max_files is not None:
            all_files = all_files[:max_files]
            logger.info(f"Processing limited to {len(all_files)} files.\n")

        for mpd_file in tqdm(all_files, desc="Processing MPD files", unit="file"):
            if not self._process_single_file(mpd_file):
                logger.warning(f"Skipping file due to processing error: {mpd_file}\n")
                continue

        logger.info("Dataset processing complete.\n")
        logger.info("Vocabulary summary:\n")
        logger.info(f"Total unique parts: {self.vocab_manager.get_parts_count()}")
        logger.info(f"Total unique colors: {self.vocab_manager.get_colors_count()}")
        logger.info(f"Total vocabulary size: {self.vocab_manager.get_vocab_size()}")

        if self.tokenized_data:
            logger.info(f"final processed sample: {self.tokenized_data[0]}")

    def _process_single_file(self, mpd_file_path: str) -> bool:
        """
        Process a single MPD file through the complete pipeline.

        Parses the MPD file, extracts and transforms brick data, collects vocabulary,
        tokenizes attributes, converts to tensor format, and saves the processed dataset.

        Args:
            mpd_file_path: Path to the MPD file to process.

        Returns:
            bool: True if processing succeeded, False if file should be skipped.
        """
        parser = MPDParser(str(mpd_file_path))

        logger.info(f"Processing {mpd_file_path} with {len(self.raw_data)} bricks.")
        logger.debug(f"submodels: {parser._submodels}")

        if not parser._submodels:
            logger.warning(f"No submodels found in {mpd_file_path}. Skipping file.")
            return False

        first_submodel_key = list(parser._submodels.keys())[0]

        self.raw_data = parser.flatten(first_submodel_key, np.eye(4))

        if not self.raw_data:
            logger.warning(
                f"No raw data extracted from {mpd_file_path}. Skipping file."
            )
            return False

        self.raw_data = parser.sort_bricks_by_position()

        # Transform data
        self.center_around_origin()

        # Collect and update vocabulary
        new_brick_ids = self.collect_brick_ids()
        new_colors = self.collect_brick_colors()

        self.all_brick_ids.update(new_brick_ids)
        self.all_colors.update(new_colors)

        # Update vocabulary incrementally
        self._update_vocabulary(new_brick_ids, new_colors)

        # Tokenize all brick attributes (positions, IDs, colors)
        self.tokenize_brick_data()

        self._to_tensor()

        self.save_dataset(
            str(Path(Config.TOKENIZED_DATASET_DIR) / f"{mpd_file_path.stem}.npy")
        )

        return True

    def center_around_origin(self) -> None:
        """
        Normalize model space by centering X and Z axes on 10 LDU grid.

        Aligns the model base (lowest point) to Y = 0. This ensures consistent
        positioning across all models.
        """
        if not self.raw_data:
            logger.warning("No raw data to center.")
            return

        # 1. Extraction des positions actuelles
        positions = np.array(
            [
                get_position_from_world_matrix(brick.world_matrix)
                for brick in self.raw_data
            ]
        )

        # 2. Calcul des limites (Bounding Box)
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)

        # 3. Calcul du centre théorique pour X et Z
        center_x = (min_coords[0] + max_coords[0]) / 2
        center_z = (min_coords[2] + max_coords[2]) / 2

        # 4. Snapping sur la grille (10 LDU pour X/Z, 8 LDU pour Y)
        # Pour Y, on prend le point le plus bas (max_coords[1] en LDraw car Y positif descend)
        # On le snappe à 8 LDU pour rester propre
        snapped_offset = np.array(
            [
                np.round(center_x / 10.0) * 10.0,
                np.round(max_coords[1] / 8.0) * 8.0,
                np.round(center_z / 10.0) * 10.0,
            ]
        )

        logger.debug(f"Applying grid-snapped centering. Offset: {snapped_offset}")

        # 5. Mise à jour des matrices mondiales
        for brick in self.raw_data:
            brick.world_matrix[:3, 3] -= snapped_offset

    def collect_brick_ids(self) -> Set[str]:
        """
        Collect unique brick IDs from current raw data.

        Returns:
            Set of unique brick identifiers from the current file.
        """
        if not self.raw_data:
            logger.warning("No raw data to collect brick IDs.")
            return set()

        # Get unique brick IDs
        unique_brick_ids = set(brick.brick_id for brick in self.raw_data)
        logger.debug(
            f"Collected {len(unique_brick_ids)} unique brick IDs from current file."
        )
        return unique_brick_ids

    def collect_brick_colors(self) -> Set[int]:
        """
        Collect unique brick colors from current raw data.

        Returns:
            Set of unique color codes from the current file.
        """
        if not self.raw_data:
            logger.warning("No raw data to collect brick colors.")
            return set()

        # Get unique colors
        unique_colors = set(brick.color for brick in self.raw_data)
        logger.debug(f"Collected {len(unique_colors)} unique colors from current file.")

        return unique_colors

    def _update_vocabulary(self, new_brick_ids: Set[str], new_colors: Set[int]) -> None:
        """
        Update vocabulary with new parts and colors.

        This method adds newly encountered parts and colors to the atlas_config.json
        vocabulary using the VocabularyManager.

        Args:
            new_brick_ids: Set of new brick identifiers to add.
            new_colors: Set of new color codes to add.
        """
        if new_brick_ids:
            self.vocab_manager.add_parts(new_brick_ids)
            logger.debug(f"Added {len(new_brick_ids)} new parts to vocabulary")

        if new_colors:
            self.vocab_manager.add_colors(new_colors)
            logger.debug(f"Added {len(new_colors)} new colors to vocabulary")

    def tokenize_brick_data(self) -> None:
        """
        Tokenize all brick attributes in one pass.

        Converts raw brick data to processed brick data by:
        - Tokenizing positions to discrete bins
        - Mapping brick IDs to vocabulary indices
        - Mapping colors to vocabulary indices
        - Extracting rotation matrices
        """
        if not self.raw_data:
            logger.warning("No raw data to tokenize.")
            return

        self.tokenized_data = []  # Clear previous data

        for brick in self.raw_data:
            # Extract position and rotation from world matrix
            position = get_position_from_world_matrix(brick.world_matrix)
            rotation_matrix = get_rotation_matrix_from_world_matrix(brick.world_matrix)

            # Tokenize position coordinates to bin IDs
            tokenized_x = self.tokenizer.position_to_bin_id(position[0], "x")
            tokenized_y = self.tokenizer.position_to_bin_id(position[1], "y")
            tokenized_z = self.tokenizer.position_to_bin_id(position[2], "z")
            tokenized_position = np.array([tokenized_x, tokenized_y, tokenized_z])

            # Map brick ID and color to vocabulary indices
            brick_idx = self.tokenizer.brick_id_to_token(brick.brick_id)
            color_idx = self.tokenizer.color_id_to_token(str(brick.color))

            # Tokenize rotation matrix to vocabulary index
            rotation_idx = self.tokenizer.rotation_matrix_to_token(rotation_matrix)

            # Create processed brick data with all tokenized attributes
            self.tokenized_data.append(
                TokenizedBrickData(
                    brick_idx=brick_idx,
                    position=tokenized_position,
                    rotation_idx=rotation_idx,
                    color_idx=color_idx,
                )
            )

        logger.info(
            f"Tokenized {len(self.tokenized_data)} bricks (positions, IDs, and colors)"
        )

    def _to_tensor(self) -> None:
        """
        Convert TokenizedBrickData to a numpy tensor.

        Creates a tensor by concatenating brick attributes (brick index, position,
        rotation index, and color index) for each brick.
        """

        for brick in self.tokenized_data:
            # create the tensor for each brick by unpacking its attributes
            # [brick_idx, x, y, z, rotation_idx, color_idx]
            brick_tensor = np.concatenate(
                (
                    [brick.brick_idx],
                    brick.position,
                    [brick.rotation_idx],
                    [brick.color_idx],
                )
            )

            if self.final_tensor is None:
                self.final_tensor = brick_tensor[np.newaxis, :]
            else:
                self.final_tensor = np.vstack(
                    (self.final_tensor, brick_tensor[np.newaxis, :])
                )

        logger.debug(f"Final tensor shape: {self.final_tensor.shape}")
        logger.debug(
            f"Final tensor data: {self.final_tensor[0:5, :]}"
        )  # log first 5 entries

    def save_dataset(self, output_path: str) -> None:
        """
        Save the final tensor dataset to a .npy file.

        Args:
            output_path: Path where the .npy file will be saved.
        """
        if self.final_tensor is not None:
            np.save(output_path, self.final_tensor)
            logger.info(f"Dataset saved to {output_path}\n")
        else:
            logger.warning("Final tensor is empty. Nothing to save.\n")


if __name__ == "__main__":
    dataset_builder = DatasetBuilder(Config.ATLAS_CONFIG_PATH)
    # Process only 10 files for testing
    dataset_builder.process_dataset(max_files=10)
