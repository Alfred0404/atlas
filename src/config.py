import logging


class Config:
    """Configuration settings for the application."""

    # logging
    LOGGING_LEVEL = logging.INFO

    # dataset paths
    RAW_DATASET_DIR = "./mpd_files/dataset/LDraw_sets/"
    PARSED_DATASET_DIR = "./processed_sets/"

    ATLAS_CONFIG_PATH = "./atlas_config.json"
