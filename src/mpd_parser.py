from rotation_matrix_to_quaternion import rotation_matrix_to_quaternion
import numpy as np
import logging
from typing import NamedTuple

from formating.customFormatter import CustomFormatter
from utils import get_position_from_world_matrix, get_rotation_matrix_from_world_matrix
import json

# Set up logging
# --------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler with CustomFormatter
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
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


class MPDParser:
    """Parser for LEGO set mpd files.
    Args:
        mpd_file_path (str): Path to the mpd file.
    """

    def __init__(self, mpd_file_path: str):
        self.mpd_file_path = mpd_file_path
        self.lines: list[str] = self.read_lego_set_file()
        self._registry: dict[str, list[str]] = {}  # to store the graph structure
        self.raw_data: list[RawBrickData] = []  # to store the raw data lines

    def read_lego_set_file(self) -> list[str]:
        """Read the content of a lego set mpd file.
        Args:
            mpd_file_path (str): Path to the mpd file.
        Returns:
            list[str]: List of lines from the mpd file.
        """
        with open(self.mpd_file_path, "r") as file:
            self.lines = file.readlines()

    def _build_registry(self):
        """Build a registry of the mpd file structure."""

        for line in self.lines:
            if line.startswith(
                "0 FILE"
            ):  # means the following lines are part of a submodel (add file key to registry, and following lines to its list)
                # New submodel key
                current_file = line.split(maxsplit=2)[2]
                self._registry[current_file] = []

            elif line.startswith(
                "1 "
            ):  # means information about a brick or another submodel
                # Add line to current submodel
                self._registry[current_file].append(line)

    def flatten(self, model_name: str, parent_matrix: np.array):
        """Flatten the mpd file structure into a list of vectors.
        Args:
            model_name (str): Name of the submodel to flatten.
            parent_matrix (np.array): Transformation matrix from parent.
        Returns:
            list[np.ndarray]: List of flattened vectors.
        """

        submodel_lines = self._registry.get(model_name, [])
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
                    logger.debug(f"Found submodel: {submodel_name}, flattening...")
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


if __name__ == "__main__":
    parser = MPDParser(mpd_file_path)
    lego_set_data = parser.read_lego_set_file()

    # test flattening and registry building
    parser._build_registry()
    logger.info("Building registry...")
    logger.info(f"Registry built:\n{json.dumps(parser._registry, indent=2)}")

    parser.flatten("4484 - Main Model.ldr\n", np.eye(4))
    # print(parser.raw_data)
    logger.info(
        f"Flattened data:\n{[raw_brick_data for raw_brick_data in parser.raw_data]}"
    )

    # logger.info(lego_set_data)
    # vector = line_to_vector("1 71 0 0 0 0 0 -1 0 -1 0 -1 0 0 32324.dat")
    # logger.info(f"Final vector: {vector}\nVector shape: {vector.shape}")
