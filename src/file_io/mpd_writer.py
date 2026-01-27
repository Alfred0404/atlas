import sys
from pathlib import Path
import numpy as np
from typing import List
from logging import setup_logging

logger = setup_logging()

# Add src to path for direct execution
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))

from data.parser import RawBrickData
from maths.transforms import (
    get_position_from_world_matrix,
    get_rotation_matrix_from_world_matrix,
)


def write_mpd_file(
    output_path: str, raw_data: List[RawBrickData], model_name: str = "main"
):
    """Write an MPD file from raw brick data.

    Args:
        output_path (str): Path where the MPD file will be written.
        raw_data (List[RawBrickData]): List of brick data with world matrices.
        model_name (str): Name of the model (default: "main").
    """

    if not Path(output_path).parent.exists():
        logger.error(f"Output directory does not exist: {Path(output_path).parent}")
        return


    with open(output_path, "w") as file:
        # Write header
        file.write(f"0 FILE {model_name}.ldr\n")
        file.write(f"0 Name: {model_name}.ldr\n")
        file.write("\n")

        # Write each brick
        for brick in raw_data:
            # Extract position from world matrix (last column, first 3 rows)
            position = get_position_from_world_matrix(brick.world_matrix)
            x, y, z = position[0], position[1], position[2]

            # Extract rotation matrix (3x3 upper-left block)
            rotation = get_rotation_matrix_from_world_matrix(brick.world_matrix)

            a, b, c = rotation[0, 0], rotation[0, 1], rotation[0, 2]
            d, e, f = rotation[1, 0], rotation[1, 1], rotation[1, 2]
            g, h, i = rotation[2, 0], rotation[2, 1], rotation[2, 2]

            # Get color
            color = int(brick.color)

            # Write line in LDraw format:
            # 1 color x y z a b c d e f g h i filename.dat
            file.write(
                f"1 {color} {x} {y} {z} {a} {b} {c} {d} {e} {f} {g} {h} {i} {brick.brick_id}.dat\n"
            )


if __name__ == "__main__":
    # Example usage with mpd_parser
    from MPDParser import MPDParser

    mpd_file_path = "./mpd_files/test.mpd"
    parser = MPDParser(mpd_file_path)
    parser.read_lego_set_file()
    parser._build_registry()
    parser.flatten("4484 - Main Model.ldr\n", np.eye(4))
    parser.center_around_origin()

    # Write the flattened model to a new MPD file
    output_path = "./mpd_files/output.mpd"
    write_mpd_file(output_path, parser.raw_data, "flattened_model")

    print(f"MPD file written to: {output_path}")
    print(f"Total bricks: {len(parser.raw_data)}")
