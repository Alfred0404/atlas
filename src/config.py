import logging

class Config:
    """Configuration settings for the application."""

    # logging
    LOGGING_LEVEL = logging.INFO

    # dataset paths
    RAW_DATASET_DIR = "./mpd_files/dataset"
    PARSED_DATASET_DIR = "./processed_sets/"

    VOCAB_PATH = "./vocab.json"
