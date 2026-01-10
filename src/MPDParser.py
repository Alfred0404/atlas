import numpy as np
import logging
from typing import NamedTuple

from formating.customFormatter import CustomFormatter
from config import LOGGING_LEVEL

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
    # raw data with world matrix representation
    brick_id: str
    world_matrix: (
        np.ndarray
    )  # shape (4,4) transformation matrix, contains position and rotation
    color: int


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


class MPDParser:
    """Parser for LEGO set mpd files.
    Args:
        mpd_file_path (str): Path to the mpd file.
    """

    def __init__(self, mpd_file_path: str):
        self.mpd_file_path = mpd_file_path
        self._submodels: dict[str, list[str]] = {}  # registry of submodels
        self.lines: list[str] = self.read_lego_set_file()  # lines of the mpd file
        self.raw_data: list[RawBrickData] = []  # parsed brick data

    def read_lego_set_file(self) -> list[str]:
        """Read the content of a lego set mpd file and build registry in one pass.
        Args:
            mpd_file_path (str): Path to the mpd file.
        Returns:
            list[str]: List of lines from the mpd file.
        """
        lines = []
        current_file = None

        with open(self.mpd_file_path, "r") as file:
            for line in file:
                lines.append(line)

                if line.startswith("0 FILE"):
                    # New submodel key
                    current_file = line.split(maxsplit=2)[2]
                    logger.debug(f"Registering submodel: {current_file}")
                    self._submodels[current_file] = []

                elif line.startswith("1 ") and current_file is not None:
                    # Add line to current submodel
                    self._submodels[current_file].append(line)

        return lines

    def flatten(self, model_name: str, parent_matrix: np.array):
        """Flatten the mpd file structure into a list of vectors.
        Args:
            model_name (str): Name of the submodel to flatten.
            parent_matrix (np.array): Transformation matrix from parent.
        Returns:
            list[np.ndarray]: List of flattened vectors.
        """

        submodel_lines = self._submodels.get(model_name, [])

        if submodel_lines is None:
            return

        logger.debug(f"Flattening model: {model_name} with {len(submodel_lines)} lines")

        for line in submodel_lines:
            if line.startswith("1 "):
                # Parse transformation matrix components from the line
                parts = line.split(maxsplit=14)
                # Format: 1 color x y z a b c d e f g h i file
                color = int(parts[1])
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
                    brick_color = int(brick_vector[0])

                    # Store as dict or structured array to preserve string brick_id
                    brick_data = RawBrickData(
                        brick_id=brick_id,
                        world_matrix=world_matrix,
                        color=brick_color,
                    )
                    self.raw_data.append(brick_data)
        return self.raw_data

    def parse(self, model_name: str) -> list[RawBrickData]:
        """Parse the MPD file and return raw brick data.

        Args:
            model_name (str): Name of the main model to parse.

        Returns:
            list[RawBrickData]: List of raw bricks with world matrices.
        """
        logger.info(f"Parsing MPD file: {self.mpd_file_path}")
        self.flatten(model_name, np.eye(4))
        logger.info(f"Parsed {len(self.raw_data)} bricks")
        return self.raw_data

        # self.save_parsed_set(f"./processed_sets/{model_name}.npy")

    def reset(self):
        """Reset the parser state, clearing data."""
        self.raw_data = []

    def save_parsed_set(self, output_path: str):
        """Save the parsed raw data to a numpy file.
        Args:
            output_path (str): Path to the output numpy file.
        """
        np.save(output_path, self.raw_data)

    def sort_bricks_by_position(self):
        """Sort the raw_data bricks by their position (x, y, z)., first by Y then X then Z. (y is up)"""
        self.raw_data.sort(key=lambda brick: (
            brick.world_matrix[1, 3],  # Y position
            brick.world_matrix[0, 3],  # X position
            brick.world_matrix[2, 3]   # Z position
        ))
        return self.raw_data


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
    line_vec = [field for field in line.split(maxsplit=13)]
    # get only the brick id (last element) without the .dat extension : "32324.dat" -> "32324" or "2412b.dat" -> "2412b"
    brick_id = line_vec[-1].split(".")[0]
    # convert numeric values to float
    logger.debug(f"Parsing line for brick ID: {brick_id}")
    numeric_values = np.array([float(val) for val in line_vec[:-1]])

    return numeric_values, brick_id


if __name__ == "__main__":
    # Example: Parse a single MPD file
    parser = MPDParser(mpd_file_path)
    raw_data = parser.parse("4484 - Main Model.ldr\n")

    logger.info(f"Successfully parsed {len(raw_data)} bricks")
    logger.info(
        "Use DatasetBuilder to transform this raw data into training-ready format"
    )
