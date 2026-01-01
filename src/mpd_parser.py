from rotation_matrix_to_quaternion import rotation_matrix_to_quaternion
import numpy as np
import logging

from formating.customFormatter import CustomFormatter
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


class MPDParser:
    def __init__(self, mpd_file_path: str):
        self.mpd_file_path = mpd_file_path
        self.lines = self.read_lego_set_file()
        self._registry = {} # to store the graph structure
        self.raw_data = [] # to store the raw data lines

    def _build_registry(self):
        """Build a registry of the mpd file structure."""

        for line in self.lines:
            if line.startswith("0 FILE"): # means the following lines are part of a submodel (add file key to registry, and following lines to its list)
                # New submodel key
                current_file = line.split(maxsplit=2)[2]
                print(f"Current file: {current_file}")
                self._registry[current_file] = []
            elif line.startswith("1 "): # means information about a brick or another submodel
                # Add line to current submodel
                self._registry[current_file].append(line)

    def read_lego_set_file(self) -> list[str]:
        """Read the content of a lego set mpd file.
        Args:
            mpd_file_path (str): Path to the mpd file.
        Returns:
            list[str]: List of lines from the mpd file.
        """
        with open(mpd_file_path, "r") as file:
            parsed_data = file.readlines()
        return parsed_data


def line_to_vector(line: str) -> np.ndarray:
    """Convert a line from the mpd file to a vector of floats, with quaternions convertion.
    Exemple line:
        1 71 0 0 0 0 0 -1 0 -1 0 -1 0 0 32324.dat
    -> [71, 0, 0, 0, 0, 0, -1, 0, 32324]

    Args:
        line (str): A line from the mpd file representing a brick.
    Returns:
        np.ndarray: A numpy array representing the line.
    """

    # Remove the first character
    line = line.split(" ", 1)[1].strip()
    # extract all components
    line_vec = [field for field in line.split()]
    # get only the brick id (last element) without the .dat extension : "32324.dat" -> 32324
    line_vec[-1] = int(line_vec[-1].split(".")[0])
    # convert all back to float
    line_vec = np.array([float(val) for val in line_vec])
    # Convert rotation matrix to quaternions for compute efficiency
    quaternions = rotation_matrix_to_quaternion(
        np.array(
            [
                [line_vec[4], line_vec[5], line_vec[6]],
                [line_vec[7], line_vec[8], line_vec[9]],
                [line_vec[10], line_vec[11], line_vec[12]],
            ]
        )
    )
    # Rebuild the final vector with quaternions [brick_id, x, y, z, q0, q1, q2, q3, color_id]
    final_vector = np.array([line_vec[-1], *line_vec[1:4], *quaternions, line_vec[0]])
    logger.info(f"Converted line to vector: {line}\n-> {final_vector}")

    return final_vector


if __name__ == "__main__":
    parser = MPDParser(mpd_file_path)
    lego_set_data = parser.read_lego_set_file()
    # logger.info(lego_set_data)
    # vector = line_to_vector("1 71 0 0 0 0 0 -1 0 -1 0 -1 0 0 32324.dat")
    # logger.info(f"Final vector: {vector}\nVector shape: {vector.shape}")

    logger.info("Building registry...")
    parser._build_registry()
    logger.info(f"Registry built:\n{json.dumps(parser._registry, indent=2)}")