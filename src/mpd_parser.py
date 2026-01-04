from rotation_matrix_to_quaternion import rotation_matrix_to_quaternion
import numpy as np
import logging
from typing import NamedTuple

from formating.customFormatter import CustomFormatter
from config import LOGGING_LEVEL
from utils import get_position_from_world_matrix, get_rotation_matrix_from_world_matrix
import json

# Set up logging
# --------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)

# create console handler with CustomFormatter
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
# --------------------------------


mpd_file_path = "./mpd_files/test.mpd"


class RawBrickData(NamedTuple):
    brick_id: str
    world_matrix: (
        np.ndarray
    )  # shape (4,4) transformation matrix, contains position and rotation
    color: float


class BrickDataQuat(NamedTuple):
    brick_id: str
    position: np.ndarray  # shape (3,) [x,y,z]
    rotation_quat: np.ndarray  # shape (4,) [w,x,y,z]
    color: float


class MPDParser:
    """Parser for LEGO set mpd files.
    Args:
        mpd_file_path (str): Path to the mpd file.
    """

    def __init__(self, mpd_file_path: str):
        self.mpd_file_path = mpd_file_path
        self.lines: list[str] = self.read_lego_set_file()  # lines of the mpd file
        self._submodels: dict[str, list[str]] = {}  # registry of submodels
        self.raw_data: list[RawBrickData] = []
        self.quat_data: list[BrickDataQuat] = [] # data with quaternion rotations
        self.max_brick_distance: float = 0.0  #  max distance of a brick from origin

    def read_lego_set_file(self) -> list[str]:
        """Read the content of a lego set mpd file.
        Args:
            mpd_file_path (str): Path to the mpd file.
        Returns:
            list[str]: List of lines from the mpd file.
        """
        with open(self.mpd_file_path, "r") as file:
            return file.readlines()

    def _build_registry(self):
        """Build a registry of the mpd file structure."""

        for line in self.lines:
            if line.startswith(
                "0 FILE"
            ):  # means the following lines are part of a submodel (add file key to registry, and following lines to its list)
                # New submodel key
                current_file = line.split(maxsplit=2)[2]
                self._submodels[current_file] = []

            elif line.startswith(
                "1 "
            ):  # means information about a brick or another submodel
                # Add line to current submodel
                self._submodels[current_file].append(line)

    def flatten(self, model_name: str, parent_matrix: np.array):
        """Flatten the mpd file structure into a list of vectors.
        Args:
            model_name (str): Name of the submodel to flatten.
            parent_matrix (np.array): Transformation matrix from parent.
        Returns:
            list[np.ndarray]: List of flattened vectors.
        """

        submodel_lines = self._submodels.get(model_name, [])
        logger.debug(f"Flattening model: {model_name} with {len(submodel_lines)} lines")

        for line in submodel_lines:
            if line.startswith("1 "):
                # Parse transformation matrix components from the line
                parts = line.split(maxsplit=14)
                # Format: 1 color x y z a b c d e f g h i file
                color = float(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                # Transformation matrix elements
                a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
                d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
                g, h, i = float(parts[11]), float(parts[12]), float(parts[13])

                local_matrix = np.array(
                    [
                        [a, b, c, x],
                        [d, e, f, y],
                        [g, h, i, z],
                        [0, 0, 0, 1],
                    ]
                )

                # compute world matrix to get the overall brick position and rotation
                world_matrix = parent_matrix @ local_matrix

                # if submodel, recursively flatten it
                if line.endswith(".ldr\n"):
                    submodel_name = line.split(maxsplit=14)[-1]
                    logger.debug(f"Found submodel: {submodel_name}")
                    self.flatten(submodel_name, world_matrix)

                # if brick, extract its data and store it
                elif line.endswith(".dat\n"):
                    brick_vector, brick_id = line_to_vector(line)
                    brick_color = brick_vector[0]

                    # Store as dict or structured array to preserve string brick_id
                    brick_data = RawBrickData(
                        brick_id=brick_id,
                        world_matrix=world_matrix,
                        color=brick_color,
                    )
                    self.raw_data.append(brick_data)

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
        avg_position = np.mean(positions, axis=0)

        logger.debug(f"Centering model around origin. Average position: {avg_position}")

        # Update world matrices to center around origin
        for idx, brick in enumerate(self.raw_data):
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = -avg_position

            new_world_matrix = translation_matrix @ brick.world_matrix

            # Update the raw_data with the new world matrix
            self.raw_data[idx] = RawBrickData(
                brick_id=brick.brick_id,
                world_matrix=new_world_matrix,
                color=brick.color,
            )

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

        self.max_brick_distance = np.sqrt(max_distance_squared)
        logger.debug(f"Max brick distance: {self.max_brick_distance}")

    def process_file(self, model_name: str):
        """Process the mpd file: build registry, flatten structure, center model, and convert to quaternion representation.
        Args:
            model_name (str): Name of the main model to process.
        """
        self._build_registry()
        self.flatten(model_name, np.eye(4))
        self.center_around_origin()
        self.to_quat_representation()
        self.calculate_max_brick_distance()
        write_max_brick_distance("./metadata.json", self.max_brick_distance)

    def reset(self):
        """Reset the parser state, clearing registry and data."""
        self._submodels = {}
        self.raw_data = []
        self.quat_data = []


def line_to_vector(line: str) -> np.ndarray:
    """Convert a line from the mpd file to a vector of floats.
    Exemple line:
        1 71 0 0 0 0 0 -1 0 -1 0 -1 0 0 32324.dat
    -> [71, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 32324]

    Args:
        line (str): A line from the mpd file representing a brick.
    Returns:
        np.ndarray: A numpy array representing the line.
    """

    # Remove the first character
    line = line.split(" ", 1)[1].strip()
    # extract all components
    line_vec = [field for field in line.split()]
    # get only the brick id (last element) without the .dat extension : "32324.dat" -> "32324" or "2412b.dat" -> "2412b"
    brick_id = line_vec[-1].split(".")[0]
    # convert numeric values to float
    numeric_values = np.array([float(val) for val in line_vec[:-1]])

    return numeric_values, brick_id


def write_max_brick_distance(output_path: str, max_distance: float):
    """Write max brick distance to a JSON file.
    Only updates the value if it is greater than the existing value.
    Args:
        output_path (str): Path to the output JSON file.
        max_distance (float): The new max distance to potentially write.
    """

    with open(output_path, "r+") as json_file:
        json_data = json.load(json_file)

        # Only update if the new value is greater than the existing one
        if (
            "max_brick_distance" not in json_data
            or max_distance > json_data["max_brick_distance"]
        ):
            json_data["max_brick_distance"] = max_distance
            json_file.seek(0)
            json_file.truncate()
            json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    parser = MPDParser(mpd_file_path)
    parser.process_file("4484 - Main Model.ldr\n")
