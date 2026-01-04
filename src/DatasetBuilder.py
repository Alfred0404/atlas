from MPDParser import MPDParser, RawBrickData, BrickDataQuat, ProcessedBrickData
from rotation_matrix_to_quaternion import rotation_matrix_to_quaternion
from utils import get_position_from_world_matrix, get_rotation_matrix_from_world_matrix
import numpy as np
from pathlib import Path
import json
import logging
from config import LOGGING_LEVEL
from formating.customFormatter import CustomFormatter

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

mpd_file_dir = "./mpd_files/dataset/"


class DatasetBuilder:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.global_max_distance = 0.0
        self.brick_vocabulary = {}
        self.color_vocabulary = {}

        # Data storage for processing individual files
        self.raw_data: list[RawBrickData] = []
        self.quat_data: list[BrickDataQuat] = []
        self.processed_data: list[ProcessedBrickData] = []

    def process_dataset(self):

        for mpd_file in Path(mpd_file_dir).glob("*.mpd"):
            # parse mpd file
            parser = MPDParser(str(mpd_file))
            logger.debug(parser._submodels)
            first_submodel_key = list(parser._submodels.keys())[0]
            self.raw_data = parser.flatten(first_submodel_key, np.eye(4))
            logger.info(f"Processing {mpd_file} with {len(self.raw_data)} bricks.")

            # Transform data
            self.center_around_origin()
            self.to_quat_representation()
            self.calculate_max_brick_distance()

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

    def map_brick_ids(self):
        """Map bricks IDs to unique integers, and store the mapping in metadata.json."""
        if not self.quat_data:
            logger.warning("No quaternion data to map brick IDs.")
            return

        # Get unique brick IDs
        unique_brick_ids = sorted(set(brick.brick_id for brick in self.quat_data))

        # Create mapping from brick_id to integer index
        brick_id_to_idx = {
            brick_id: idx for idx, brick_id in enumerate(unique_brick_ids)
        }

        # Write mapping to metadata.json
        self._update_metadata(self.metadata_path, "brick_id_mapping", brick_id_to_idx)

        logger.debug(f"Mapped {len(unique_brick_ids)} unique brick IDs to integers.")
        return brick_id_to_idx

    def map_brick_colors(self):
        """Map brick colors to unique integers, and store the mapping in metadata.json."""
        if not self.quat_data:
            logger.warning("No quaternion data to map brick colors.")
            return

        # Get unique colors
        unique_colors = sorted(set(brick.color for brick in self.quat_data))

        # Create mapping from color to integer index
        color_to_idx = {color: idx for idx, color in enumerate(unique_colors)}

        # Write mapping to metadata.json
        self._update_metadata(self.metadata_path, "color_mapping", color_to_idx)

        logger.debug(f"Mapped {len(unique_colors)} unique colors to integers.")
        return color_to_idx

    def _update_metadata(self, output_path: str, key: str, value: dict):
        """Update or add a key-value pair in the metadata JSON file.
        Args:
        output_path (str): Path to the metadata JSON file.
        key (str): The key to update or add.
        value (dict): The value to set for the key.
        """
        try:
            with open(output_path, "r+") as json_file:
                json_data = json.load(json_file)
                json_data[key] = value
                json_file.seek(0)
                json_file.truncate()
                json.dump(json_data, json_file, indent=4)

        except FileNotFoundError:
            # Create new file if it doesn't exist
            with open(output_path, "w") as json_file:
                json.dump({key: value}, json_file, indent=4)

    def use_id_mapping(self, id_mapping: dict):
        """Fill processed_data using a provided brick and color ID mapping."""
        if not self.quat_data:
            logger.warning("No quaternion data to process.")
            return

        self.processed_data = []  # Clear previous data
        for brick in self.quat_data:
            brick_idx = id_mapping["brick_id_mapping"].get(brick.brick_id, -1)
            color_idx = id_mapping["color_mapping"].get(brick.color, -1)

            processed_brick = ProcessedBrickData(
                brick_idx=brick_idx,
                position=brick.position,
                rotation_quat=brick.rotation_quat,
                color_idx=color_idx,
            )

            self.processed_data.append(processed_brick)
        logger.debug(f"Processed {len(self.processed_data)} bricks using ID mapping.")
        logger.debug(f"Sample processed brick: {self.processed_data[0]}")

        return self.processed_data

    def _finalize_dataset(self):
        """Finalize the dataset by writing the global max distance to metadata."""
        self._update_metadata(
            self.metadata_path, "global_max_brick_distance", self.global_max_distance
        )
        logger.info(
            f"Dataset finalized. Global max distance: {self.global_max_distance}"
        )

    def _update_global_max_distance(self, distance: float):
        """Update the global maximum distance if the new distance is larger."""
        if distance > self.global_max_distance:
            self.global_max_distance = distance
            logger.debug(f"Updated global max distance to: {distance}")

    def _update_brick_vocabulary(self, brick_id: str):
        """Update brick vocabulary with a new brick ID."""
        if brick_id not in self.brick_vocabulary:
            idx = len(self.brick_vocabulary)
            self.brick_vocabulary[brick_id] = idx
            logger.debug(f"Added brick {brick_id} to vocabulary at index {idx}")

    def _update_color_vocabulary(self, color: int):
        """Update color vocabulary with a new color."""
        if color not in self.color_vocabulary:
            idx = len(self.color_vocabulary)
            self.color_vocabulary[color] = idx
            logger.debug(f"Added color {color} to vocabulary at index {idx}")

    def to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to a quaternion.

        Args:
            rotation_matrix: 3x3 rotation matrix

        Returns:
            np.ndarray: Quaternion [w, x, y, z]
        """
        return rotation_matrix_to_quaternion(rotation_matrix)


if __name__ == "__main__":
    metadata_path = "./mpd_files/metadata.json"
    dataset_builder = DatasetBuilder(metadata_path)
    dataset_builder.process_dataset()
