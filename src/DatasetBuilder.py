import numpy as np
from pathlib import Path
import logging
import json

from utils import get_position_from_world_matrix, get_rotation_matrix_from_world_matrix
from MPDParser import MPDParser, RawBrickData
from rotation_matrix_to_quaternion import (
    rotation_matrix_to_quaternion,
    generate_quat_chiral_rotations,
)
from formating.customFormatter import CustomFormatter
from typing import NamedTuple

from config import LOGGING_LEVEL, RAW_DATASET_DIR, PARSED_DATASET_DIR, METADATA_PATH

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

raw_dataset_dir = RAW_DATASET_DIR
parsed_dataset_dir = PARSED_DATASET_DIR


class BrickDataQuat(NamedTuple):
    # data with quaternion rotation representation
    brick_id: str
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_quat: np.ndarray  # shape (4,) [w,x,y,z]
    color: int


class ProcessedBrickData(NamedTuple):
    # training ready data with brick_id and color mapped to integers
    brick_idx: int  # mapped unique integer for brick_id
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_quat: np.ndarray  # shape (4,) [w,x,y,z]
    color_idx: int  # mapped unique integer for color


class DatasetBuilder:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.global_max_distance = 0.0
        self.vocabulary = {"[SOS]": 0, "[EOS]": 1, "[PAD]": 2, "[UNK]": 3}
        self.all_brick_ids = set()
        self.all_colors = set()
        self.unique_rotations = []

        # Data storage for processing individual files
        self.raw_data: list[RawBrickData] = []
        self.quat_data: list[BrickDataQuat] = []
        self.processed_data: list[ProcessedBrickData] = []
        self.final_tensor = None

    def process_dataset(self):
        """Process all MPD files in the raw dataset directory."""

        logger.info("Starting dataset processing...\n")

        all_files = list(Path(raw_dataset_dir).glob("*.mpd"))

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

            # self.calculate_max_brick_distance() not needed because we use transformer architecture instead of diffusion

            new_brick_ids = self.collect_brick_ids()
            new_colors = self.collect_brick_colors()

            self.all_brick_ids.update(new_brick_ids)
            self.all_colors.update(new_colors)

            # self._to_tensor()
            # self.save_dataset(output_path=f"{parsed_dataset_dir}{mpd_file.stem}.npy")

        # Build unified vocabulary: special tokens (0-3), then all bricks, then all colors
        self.build_unified_vocabulary()
        self.save_vocabulary()

        logger.info("Dataset processing complete.\n")

        logger.debug(f"final quat data sample: {self.quat_data[0]}")

    def center_around_origin(self):
        """Center the model around the origin based on the average position of all bricks."""
        if not self.raw_data:
            logger.warning("No raw data to center.")
            return

        # Compute average position
        positions = np.array(
            [
                get_position_from_world_matrix(brick.world_matrix)
                for brick in self.raw_data
            ]
        )
        barycenter = np.mean(positions, axis=0)

        logger.debug(f"Centering model around origin. Average position: {barycenter}")

        # Update world matrices to center around origin
        for brick in self.raw_data:
            brick.world_matrix[:3, 3] -= barycenter

    def to_quat_representation(self):
        """
        Populate the quaternion-based brick representation from the current raw data.

        This method iterates over all entries in ``self.raw_data``, which are expected
        to hold 4x4 world transformation matrices. For each brick it:

        * extracts the position from the world matrix,
        * extracts the rotation matrix from the world matrix,
        * converts the rotation matrix to a quaternion, and
        * creates a ``BrickDataQuat`` instance containing the brick id, position,
          quaternion rotation, and color.

        The resulting ``BrickDataQuat`` objects are appended to
        ``self.quat_data``. Existing contents of ``self.quat_data``
        are preserved; this method does not clear the list beforehand.
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

    def calculate_max_brick_distance(self):
        """Calculate the maximum distance between a brick and the origin."""
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

    def collect_brick_ids(self):
        """Collect unique brick IDs from current quaternion data."""
        if not self.quat_data:
            logger.warning("No quaternion data to collect brick IDs.")
            return set()

        # Get unique brick IDs
        unique_brick_ids = set(brick.brick_id for brick in self.quat_data)
        logger.debug(
            f"Collected {len(unique_brick_ids)} unique brick IDs from current file."
        )
        return unique_brick_ids

    def collect_brick_colors(self):
        """Collect unique brick colors from current quaternion data."""
        if not self.quat_data:
            logger.warning("No quaternion data to collect brick colors.")
            return set()

        # Get unique colors
        unique_colors = set(brick.color for brick in self.quat_data)
        logger.debug(f"Collected {len(unique_colors)} unique colors from current file.")
        return unique_colors

    def build_unified_vocabulary(self):
        """Build unified vocabulary with all bricks first, then all colors.

        Vocabulary structure:
        - Indices 0-3: Special tokens ([SOS], [EOS], [PAD], [UNK])
        - Indices 4+: All brick IDs (sorted)
        - Indices after bricks: All colors (sorted)
        """

        # Start indexing after special tokens
        logger.info("Building unified vocabulary.\n")

        current_index = 4

        # Add all unique rotations to vocabulary before bricks and colors because they are a fix number of 24
        self.unique_rotations = generate_quat_chiral_rotations()
        for rot in self.unique_rotations:
            rot_key = f"rotation_{','.join(map(str, rot))}"
            self.vocabulary[rot_key] = current_index
            current_index += 1

        # Add all brick IDs (sorted for consistency)
        sorted_brick_ids = sorted(self.all_brick_ids)
        for brick_id in sorted_brick_ids:
            self.vocabulary[brick_id] = current_index
            current_index += 1

        logger.info(
            f"Added {len(sorted_brick_ids)} bricks to vocabulary (indices 4-{current_index-1})"
        )

        # Add all colors (sorted for consistency)
        sorted_colors = sorted(self.all_colors)
        for color in sorted_colors:
            color_key = f"color_{color}"
            self.vocabulary[color_key] = current_index
            current_index += 1


        logger.info(
            f"Added {len(sorted_colors)} colors to vocabulary (indices {current_index-len(sorted_colors)}-{current_index-1})\n"
        )
        logger.info(
            f"Final vocabulary size: {len(self.vocabulary)} (Bricks: {len(self.all_brick_ids)}, Colors: {len(self.all_colors)}, Special tokens: 4)\n"
        )
        logger.debug(f"Sample vocabulary entries: {list(self.vocabulary.items())[:10]}")

    def add_rotations_to_vocabulary(self, output_path: str):
        """Add the 24 chiral octahedral rotations to the vocabulary JSON file.

        Args:
            output_path (str): Path to the metadata JSON file.
        """
        rotations = generate_quat_chiral_rotations()
        rotation_strings = [",".join(map(str, rot)) for rot in rotations]
        logger.info(
            f"Adding {len(rotation_strings)} chiral octahedral rotations to vocabulary."
        )
        logger.debug(f"Sample rotations: {rotation_strings[:3]}")


    def update_json(self, output_path: str, key: str, new_items: list):
        """
        Add new items to a vocabulary category ensuring
        unique and increasing integer IDs.

        Args:
            output_path (str): Path to the metadata JSON file.
            key (str): The category key in the JSON (e.g., "brick_vocabulary").
            new_items (list): List of new items to add to the category.
        """
        json_data = {}

        # load existing data
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

        except (FileNotFoundError, json.JSONDecodeError):
            json_data = {}

        # add new category if not present
        if key not in json_data:
            json_data[key] = {}

        category_dict = json_data[key]

        # add new items with unique IDs
        # check for existing IDs and find the max
        current_ids = list(category_dict.values())
        next_id = max(current_ids) + 1 if current_ids else 0

        # add new items
        changes_made = False
        for item in new_items:
            item_str = str(item)
            if item_str not in category_dict:
                category_dict[item_str] = next_id
                next_id += 1
                changes_made = True

        # write back only if changes were made
        if changes_made:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, sort_keys=True)

    def use_id_mapping(self, id_mapping: dict):
        """Fill processed_data using a provided unified vocabulary mapping."""
        if not self.quat_data:
            logger.warning("No quaternion data to process.")
            return

        self.processed_data = []  # Clear previous data
        vocabulary = id_mapping.get("vocabulary", {})

        for brick in self.quat_data:
            brick_idx = vocabulary.get(
                brick.brick_id, 3
            )  # Use [UNK] token (3) if not found
            color_idx = vocabulary.get(
                f"color_{brick.color}", 3
            )  # Use [UNK] token (3) if not found

            processed_brick = ProcessedBrickData(
                brick_idx=brick_idx,
                position=brick.position,
                rotation_quat=brick.rotation_quat,
                color_idx=color_idx,
            )

            self.processed_data.append(processed_brick)
        logger.debug(f"Processed {len(self.processed_data)} bricks using ID mapping.")
        logger.debug(f"Sample quaternion brick: {self.quat_data[0]}")
        logger.debug(f"Sample processed brick: {self.processed_data[0]}")

        return self.processed_data

    def save_vocabulary(self):
        """Save unified vocabulary to metadata file."""
        vocab_data = {"vocabulary": self.vocabulary}

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=4, sort_keys=True)

        logger.info(
            f"Vocabulary saved to {self.metadata_path} with {len(self.vocabulary)} total entries\n"
        )

    def _update_global_max_distance(self, distance: float):
        """Update the global maximum distance if the new distance is larger."""
        if distance > self.global_max_distance:
            self.global_max_distance = distance
            logger.debug(f"Updated global max distance to: {distance}")

    def to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to a quaternion.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            np.ndarray: Quaternion [w, x, y, z]
        """
        return rotation_matrix_to_quaternion(rotation_matrix)

    def _to_tensor(self):
        """Finalize the process by converting data from ProcessedBrickData to a tensor."""

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

    def save_dataset(self, output_path: str):
        """Save the final tensor dataset to a .npy file."""
        if self.final_tensor is not None:
            np.save(output_path, self.final_tensor)
            logger.info(f"Dataset saved to {output_path}")
        else:
            logger.warning("Final tensor is empty. Nothing to save.")


if __name__ == "__main__":
    dataset_builder = DatasetBuilder(METADATA_PATH)
    dataset_builder.process_dataset()
