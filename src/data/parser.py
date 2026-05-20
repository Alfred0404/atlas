import sys
from pathlib import Path
import numpy as np
from typing import NamedTuple

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from ..utils.logging import setup_logging

logger = setup_logging()


mpd_file_path = "./mpd_files/test.mpd"


class RawBrickData(NamedTuple):
    # raw data with world matrix representation
    brick_id: str
    world_matrix: (
        np.ndarray
    )  # shape (4,4) transformation matrix, contains position and rotation
    color: int


class MPDParser:
    """Parser for LEGO set mpd files.
    Args:
        mpd_file_path (str): Path to the mpd file.
    """

    def __init__(self, mpd_file_path: str):
        """Initialize the parser, read the file and build the submodel registry."""
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

        if not Path(self.mpd_file_path).exists():
            logger.error(f"MPD file path does not exist: {self.mpd_file_path}")
            return lines

        if not Path(self.mpd_file_path).is_file():
            logger.error(f"MPD file not found: {self.mpd_file_path}")
            return lines

        with open(self.mpd_file_path, "r") as file:
            for line in file:
                lines.append(line)

                if line.startswith("0 FILE"):
                    # Normalize submodel name (strip trailing spaces/newlines)
                    current_file = line.split(maxsplit=2)[2].strip()
                    self._submodels[current_file] = []

                elif line.startswith("1 ") and current_file is not None:
                    # Add line to current submodel
                    # keep raw line (may contain trailing spaces) — we'll parse safely later
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

        for line in submodel_lines:
            if line.startswith("1 "):
                # Parse transformation matrix components from the line
                parts = line.split(maxsplit=14)
                # Format: 1 color x y z a b c d e f g h i file
                filename = (
                    parts[-1] if len(parts) >= 15 else line.rsplit(maxsplit=1)[-1]
                )
                filename = filename.strip().lower().replace("\\", "/")
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
                if filename.endswith(".ldr"):
                    submodel_name = filename
                    self.flatten(submodel_name, world_matrix)

                # if brick, extract its data and store it
                elif filename.endswith(".dat"):
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
        logger.info(f"Parsing MPD file: {self.mpd_file_path}\n")
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
        """Sort bricks deterministically: bottom-to-top (Y), then X, then Z."""
        self.raw_data.sort(
            key=lambda brick: (
                -brick.world_matrix[1, 3],  # Y in LDraw grows downward
                brick.world_matrix[0, 3],  # X position
                brick.world_matrix[2, 3],  # Z position
            )
        )
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
