def setup_logging():
    """Set up and return a logger with custom formatting."""
    
    import logging
    from formating.customFormatter import CustomFormatter
    from config import Config

    logger = logging.getLogger("atlas_logger")

    # Only configure if handlers haven't been added yet
    if not logger.handlers:
        logger.setLevel(Config.LOGGING_LEVEL)
        ch = logging.StreamHandler()
        ch.setLevel(Config.LOGGING_LEVEL)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    return logger
