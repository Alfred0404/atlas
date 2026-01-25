from config import Config
import logging
from formating.customFormatter import CustomFormatter

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(Config.LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(Config.LOGGING_LEVEL)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class AtlasTokenizer:
    """**Tokenizer for ATLAS model.**

    Handles **tokenization** *(encoding)* of brick attributes *(such as position, rotation, color, and part ID)* as well as the **detokenization** *(decoding)* process, to reconstruct brick attributes from tokens.
    """

    def __init__(self):
        pass

    def position_to_bin_id(self, position: float, axis: str) -> int:
        """Convert a real-valued position to its corresponding bin ID.
        Args:
            position (float): Real-valued position.
            axis (str): Axis of the position ('x', 'y', or 'z').
        Returns:
            int: Corresponding bin ID.
        """
        logger.debug(f"Converting position {position} on axis {axis} to bin ID.")
        bin_id = int((position - Config.MIN_POSITION) / Config.PRECISION) + Config.OFFSETS[f"positions_{axis}"]
        return bin_id

    def bin_id_to_position(self, bin_id: int, axis: str) -> float:
        """Convert a bin ID back to its corresponding real-valued position.
        Args:
            bin_id (int): Bin ID.
            axis (str): Axis of the position ('x', 'y', or 'z').
        Returns:
            float: Corresponding real-valued position.
        """
        logger.debug(f"Converting bin ID {bin_id} on axis {axis} back to position.")
        position = (bin_id - Config.OFFSETS[f"positions_{axis}"]) * Config.PRECISION + Config.MIN_POSITION
        return position


if __name__ == "__main__":
    tokenizer = AtlasTokenizer()

    test_position = 150.5
    # Convert position to bin ID
    bin_id = tokenizer.position_to_bin_id(test_position, "x")
    logger.debug(f"Position {test_position} maps to bin ID: {bin_id}")

    # Convert bin ID back to position
    recovered_position = tokenizer.bin_id_to_position(bin_id, "x")
    logger.debug(f"Bin ID {bin_id} maps back to position: {recovered_position}")

    # the current precision may lead to slight differences due to rounding, but is very close, because 1 LDU is quite small