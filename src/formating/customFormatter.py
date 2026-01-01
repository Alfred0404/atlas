# Source - https://stackoverflow.com/a
# Posted by Sergey Pleshakov, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-05, License - CC BY-SA 4.0

import logging


# Custom log formatter with colors and detailed info
class CustomFormatter(logging.Formatter):

    # ANSI escape codes for colors
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(filename)s:%(lineno)d - %(asctime)s - %(message)s"

    # Define different formats for different log levels
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    # Override format method to use different formats based on log level
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)
