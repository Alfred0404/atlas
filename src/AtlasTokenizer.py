from config import Config
import logging
import json
from utils import setup_logging

logger = setup_logging()


class AtlasTokenizer:
    """**Tokenizer for ATLAS model.**

    Handles **tokenization** *(encoding)* of brick attributes *(such as position, rotation, color, and part ID)* as well as the **detokenization** *(decoding)* process, to reconstruct brick attributes from tokens.
    """

    def __init__(self):
        logger.info("Initialized AtlasTokenizer.")

    def position_to_bin_id(self, position: float, axis: str) -> int:
        """Convert a real-valued position to its corresponding bin ID.
        Args:
            position (float): Real-valued position.
            axis (str): Axis of the position ('x', 'y', or 'z').
        Returns:
            int: Corresponding bin ID.
        """
        logger.debug(f"Converting position {position} on axis {axis} to bin ID.")
        bin_id = (
            int((position - Config.MIN_POSITION) / Config.PRECISION)
            + Config.OFFSETS[f"positions_{axis}"]
        )

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
        position = (
            bin_id - Config.OFFSETS[f"positions_{axis}"]
        ) * Config.PRECISION + Config.MIN_POSITION

        return position

    def _get_token_from_vocabulary(self, value: str, vocab_key: str, offset_key: str) -> int:
        """Helper method to get token ID from vocabulary with offset.
        Args:
            value (str): Value to look up in the vocabulary.
            vocab_key (str): Key in the vocabulary dictionary.
            offset_key (str): Key in the offsets dictionary.
        returns:
            int: Corresponding token ID.
        """
        with open(Config.ATLAS_CONFIG_PATH, "r") as f:
            atlas_config = json.load(f)

            offsets = atlas_config["offsets"]
            vocabulary = atlas_config["vocabulary"]

            vocab_dict: dict = vocabulary[vocab_key]
            offset: int = offsets[offset_key]
            unknown_token_id: int = vocabulary["special"]["UNK"]

            token_id: int = vocab_dict.get(value, None)

        if token_id is None:
            logger.warning(f"Value {value} not found in vocabulary for key {vocab_key}.")
            return unknown_token_id


        final_token_id: int = token_id + offset
        logger.debug(f"Value {value} maps to token ID {final_token_id}.")

        return final_token_id

    def brick_id_to_token(self, brick_id: str) -> int:
        """Convert a brick ID to its corresponding token ID.
        Args:
            brick_id (str): Brick ID.
        Returns:
            int: Corresponding token ID.
        """
        logger.debug(f"Converting brick ID {brick_id} to token ID.")
        token_id = self._get_token_from_vocabulary(
            value=brick_id,
            vocab_key="parts",
            offset_key="parts"
        )

        return token_id

    def color_id_to_token(self, color_id: str) -> int:
        """Convert a color ID to its corresponding token ID.
        Args:
            color_id (str): Color ID.
        Returns:
            int: Corresponding token ID.
        """
        logger.debug(f"Converting color ID {color_id} to token ID.")
        token_id = self._get_token_from_vocabulary(
            value=color_id,
            vocab_key="colors",
            offset_key="colors"
        )

        return token_id


if __name__ == "__main__":
    tokenizer = AtlasTokenizer()

    test_position = 150.5
    # Convert position to bin ID
    bin_id = tokenizer.position_to_bin_id(test_position, "x")
    logger.debug(f"Position {test_position} maps to bin ID: {bin_id}")

    # Convert bin ID back to position
    recovered_position = tokenizer.bin_id_to_position(bin_id, "x")
    logger.debug(f"Bin ID {bin_id} maps back to position: {recovered_position}")

    # Convert brick ID to token ID
    tokenizer.brick_id_to_token("3001")
    # Convert color ID to token ID
    tokenizer.color_id_to_token("383")
