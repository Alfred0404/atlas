import logging


class Config:
    """Configuration settings for the codebase. only storing consants for now."""

    # logging
    LOGGING_LEVEL = logging.INFO

    # dataset paths
    RAW_DATASET_DIR = "./dataset/mpd_files/"
    TOKENIZED_DATASET_DIR = "./tokenized_sets/"

    ATLAS_CONFIG_PATH = "./atlas_config.json"
    TECHNIC_BLACKLIST_PATH = "./dataset/technic_blacklist.txt"

    # spatial configuration, for position binning (encoding/decoding)
    PRECISION = 2  # in LDU
    MIN_POSITION = -1000
    MAX_POSITION = 1000

    # derived constants
    NUM_SPECIAL = 4       # PAD, SOS, EOS, UNK
    NUM_ROTATIONS = 24    # octahedral group
    NUM_BINS_PER_AXIS = (MAX_POSITION - MIN_POSITION) // PRECISION  # 1000

    # adjacency-based brick sorting (BFS)
    ADJACENCY_THRESHOLD = 1.5  # in normalized grid units
    ADJACENCY_GRID_STEPS = {"x": 20.0, "y": 8.0, "z": 20.0}  # LDU per grid step

    # token offsets — computed from constants above
    _rot_start = NUM_SPECIAL
    _pos_x_start = _rot_start + NUM_ROTATIONS
    _pos_y_start = _pos_x_start + NUM_BINS_PER_AXIS
    _pos_z_start = _pos_y_start + NUM_BINS_PER_AXIS
    _colors_start = _pos_z_start + NUM_BINS_PER_AXIS

    OFFSETS = {
        "special": 0,
        "rotations": _rot_start,        # 4
        "positions_x": _pos_x_start,    # 28
        "positions_y": _pos_y_start,    # 1028
        "positions_z": _pos_z_start,    # 2028
        "colors": _colors_start,        # 3028
        "parts": _colors_start,         # dynamic, updated by VocabularyManager
    }