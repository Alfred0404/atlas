import logging


class Config:
    """Configuration settings for the codebase."""

    # logging
    LOGGING_LEVEL = logging.INFO

    # dataset paths
    RAW_DATASET_DIR = "./mpd_files/dataset/LDraw_sets/"
    PARSED_DATASET_DIR = "./processed_sets/"

    ATLAS_CONFIG_PATH = "./atlas_config.json"

    # spatial configuration, for position binning (encoding/decoding)
    PRECISION = 2 # in LDU
    MIN_POSITION = -1000
    MAX_POSITION = 1000

    OFFSETS = {
        "special": 0,
        "rotations": 4,
        "positions_x": 28,
        "positions_y": 1028,
        "positions_z": 2028,
        "colors": 3028,
        "parts": 3128
    }