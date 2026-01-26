from typing import NamedTuple, List, Set
import numpy as np
from pathlib import Path

from MPDParser import MPDParser, RawBrickData
from VocabularyManager import VocabularyManager
from AtlasTokenizer import AtlasTokenizer

from utils import (
    get_position_from_world_matrix,
    get_rotation_matrix_from_world_matrix,
    setup_logging,
)
from rotation_matrix_to_quaternion import rotation_matrix_to_quaternion
from config import Config

logger = setup_logging()


class BrickDataQuat(NamedTuple):
    """Brick data with rotation in quaternion format"""

    brick_id: str
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_quat: np.ndarray  # shape (4,) [w,x,y,z]
    color: int


class ProcessedBrickData(NamedTuple):
    """Training ready data with brick_id and color mapped to integers"""

    brick_idx: int  # mapped unique integer for brick_id
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_quat: np.ndarray  # shape (4,) [w,x,y,z]
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
        self.global_max_distance = 0.0
        self.all_brick_ids: Set[str] = set()
        self.all_colors: Set[int] = set()

        # Data storage for processing individual files
        self.raw_data: List[RawBrickData] = []
        self.quat_data: List[BrickDataQuat] = []
        self.processed_data: List[ProcessedBrickData] = []
        self.final_tensor = None

    def process_dataset(self) -> None:
        """
        Process all MPD files in the raw dataset directory.

        Iterates through all .mpd files, parses them, transforms the data,
        and updates the vocabulary with encountered parts and colors.
        """
        logger.info("Starting dataset processing...\n")

        all_files = list(Path(Config.RAW_DATASET_DIR).glob("*.mpd"))

        for mpd_file in all_files:
            parser = MPDParser(str(mpd_file))

            logger.info(f"Processing {mpd_file} with {len(self.raw_data)} bricks.")
            logger.debug(f"submodels: {parser._submodels}")

            if not parser._submodels:
                logger.warning(f"No submodels found in {mpd_file}. Skipping file.")
                continue

            first_submodel_key = list(parser._submodels.keys())[0]

            self.raw_data = parser.flatten(first_submodel_key, np.eye(4))
            self.raw_data = parser.sort_bricks_by_position()

            # Transform data
            self.center_around_origin()
            self.to_quat_representation()

            logger.info(f"Transformation completed for {mpd_file}\n")

            # Collect and update vocabulary
            new_brick_ids = self.collect_brick_ids()
            new_colors = self.collect_brick_colors()

            self.all_brick_ids.update(new_brick_ids)
            self.all_colors.update(new_colors)

            # Update vocabulary incrementally
            self._update_vocabulary(new_brick_ids, new_colors)

            # Tokenize all brick attributes (positions, IDs, colors)
            self.tokenize_brick_data()

        logger.info("Dataset processing complete.\n")
        logger.info(f"Total unique parts: {self.vocab_manager.get_parts_count()}")
        logger.info(f"Total unique colors: {self.vocab_manager.get_colors_count()}")
        logger.info(f"Total vocabulary size: {self.vocab_manager.get_vocab_size()}")

        if self.process_dataset:
            logger.info(f"final quat data sample: {self.processed_data[0]}")

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

    def to_quat_representation(self) -> List[BrickDataQuat]:
        """
        Convert raw brick data to quaternion-based representation.

        Iterates over all entries in self.raw_data (4x4 world transformation matrices)
        and for each brick:
        * Extracts the position from the world matrix
        * Extracts the rotation matrix from the world matrix
        * Converts the rotation matrix to a quaternion
        * Creates a BrickDataQuat instance

        Returns:
            List of BrickDataQuat objects with quaternion rotations.
        """
        self.quat_data = []  # Clear previous data
        for brick in self.raw_data:
            position = get_position_from_world_matrix(brick.world_matrix)
            rotation_matrix = get_rotation_matrix_from_world_matrix(brick.world_matrix)
            rotation_quat = rotation_matrix_to_quaternion(rotation_matrix)

            brick_quat_data = BrickDataQuat(
                brick_id=brick.brick_id,
                position=position,
                rotation_quat=rotation_quat,
                color=brick.color,
            )

            self.quat_data.append(brick_quat_data)
        return self.quat_data

    def calculate_max_brick_distance(self) -> float:
        """
        Calculate the maximum distance between a brick and the origin.

        Returns:
            The maximum distance from origin to any brick position.
        """
        max_distance_squared = 0.0

        for brick in self.raw_data:
            position = get_position_from_world_matrix(brick.world_matrix)
            distance_squared = np.dot(
                position, position
            )  # squared distance from origin

            if distance_squared > max_distance_squared:
                max_distance_squared = distance_squared

        max_distance = np.sqrt(max_distance_squared)
        logger.debug(f"Max brick distance: {max_distance}")

        # Update global max distance
        self._update_global_max_distance(max_distance)

        return max_distance

    def collect_brick_ids(self) -> Set[str]:
        """
        Collect unique brick IDs from current quaternion data.

        Returns:
            Set of unique brick identifiers from the current file.
        """
        if not self.quat_data:
            logger.warning("No quaternion data to collect brick IDs.")
            return set()

        # Get unique brick IDs
        unique_brick_ids = set(brick.brick_id for brick in self.quat_data)
        logger.debug(
            f"Collected {len(unique_brick_ids)} unique brick IDs from current file."
        )
        return unique_brick_ids

    def collect_brick_colors(self) -> Set[int]:
        """
        Collect unique brick colors from current quaternion data.

        Returns:
            Set of unique color codes from the current file.
        """
        if not self.quat_data:
            logger.warning("No quaternion data to collect brick colors.")
            return set()

        # Get unique colors
        unique_colors = set(brick.color for brick in self.quat_data)
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

        Converts quaternion brick data to processed brick data by:
        - Tokenizing positions to discrete bins
        - Mapping brick IDs to vocabulary indices
        - Mapping colors to vocabulary indices
        - Preserving rotation quaternions
        """
        if not self.quat_data:
            logger.warning("No quaternion data to tokenize.")
            return

        self.processed_data = []  # Clear previous data

        for brick in self.quat_data:
            # Tokenize position coordinates to bin IDs
            tokenized_x = self.tokenizer.position_to_bin_id(brick.position[0], "x")
            tokenized_y = self.tokenizer.position_to_bin_id(brick.position[1], "y")
            tokenized_z = self.tokenizer.position_to_bin_id(brick.position[2], "z")
            tokenized_position = np.array([tokenized_x, tokenized_y, tokenized_z])

            # Map brick ID and color to vocabulary indices
            brick_idx = self.tokenizer.brick_id_to_token(brick.brick_id)
            color_idx = self.tokenizer.color_id_to_token(str(brick.color))

            # Create processed brick data with all tokenized attributes
            self.processed_data.append(
                ProcessedBrickData(
                    brick_idx=brick_idx,
                    position=tokenized_position,
                    rotation_quat=brick.rotation_quat,
                    color_idx=color_idx,
                )
            )

        logger.info(
            f"Tokenized {len(self.processed_data)} bricks (positions, IDs, and colors)"
        )

    def _update_global_max_distance(self, distance: float) -> None:
        """
        Update the global maximum distance if the new distance is larger.

        Args:
            distance: New distance to compare against current maximum.
        """
        if distance > self.global_max_distance:
            self.global_max_distance = distance
            logger.debug(f"Updated global max distance to: {distance}")

    def _to_tensor(self) -> None:
        """
        Convert ProcessedBrickData to a numpy tensor.

        Creates a tensor by concatenating brick attributes (index, position,
        rotation quaternion, and color index) for each brick.
        """

        for brick in self.processed_data:
            # create the tensor for each brick by unpacking its attributes
            brick_tensor = np.concatenate(
                (
                    [brick.brick_idx],
                    brick.position,
                    brick.rotation_quat,
                    [brick.color_idx],
                )
            )

            if self.final_tensor is None:
                self.final_tensor = brick_tensor[np.newaxis, :]
            else:
                self.final_tensor = np.vstack(
                    (self.final_tensor, brick_tensor[np.newaxis, :])
                )

        logger.info(f"Final tensor shape: {self.final_tensor.shape}")
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
            logger.info(f"Dataset saved to {output_path}")
        else:
            logger.warning("Final tensor is empty. Nothing to save.")


if __name__ == "__main__":
    dataset_builder = DatasetBuilder(Config.ATLAS_CONFIG_PATH)
    dataset_builder.process_dataset()
