# Source - https://stackoverflow.com/a
# Posted by Sergey Pleshakov, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-05, License - CC BY-SA 4.0

import logging


class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors and detailed info.
    Formats log messages with filename, line number, timestamp, and message.
    Different log levels have different colors for better visibility.
    """

    # ANSI escape codes for colors
    grey = "\x1b[38;20m"
    magenta = "\x1b[35;3m"
    blue = "\x1b[34m"
    cyan = "\x1b[36m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format = "%(filename)s:%(lineno)d|%(asctime)s| %(message)s"

    # Define different formats for different log levels
    FORMATS = {
        logging.DEBUG: magenta + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        """Format log record based on its level.
        Args:
            record (logging.LogRecord): Log record to format.
        Returns:
            str: Formatted log message."""

        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)
