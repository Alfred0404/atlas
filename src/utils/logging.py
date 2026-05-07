import logging

from ..formating.customFormatter import CustomFormatter


def setup_logging() -> logging.Logger:
    """Set up and return a logger with custom formatting."""
    logger = logging.getLogger("atlas_logger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    return logger
