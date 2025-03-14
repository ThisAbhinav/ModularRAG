import logging
import logging.config
import os


class CustomFormatter(logging.Formatter):
    """
    A custom log formatter that provides colored output based on the log level.

    Attributes:
        grey (str): ANSI escape sequence for grey color.
        yellow (str): ANSI escape sequence for yellow color.
        red (str): ANSI escape sequence for red color.
        bold_red (str): ANSI escape sequence for bold red color.
        reset (str): ANSI escape sequence to reset color.
        format (str): Log message format string.
        FORMATS (dict): Dictionary mapping log levels to formatted log messages.

    Methods:
        format(record): Formats the log record based on the log level.

    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        """
        Formats the log record based on the log level.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log message.

        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Ensure the log directory exists
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configuration dictionary
# LOGGING_CONFIG = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "default": {
#             "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         },
#         "detailed": {
#             "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]",
#         },
#     },
#     "handlers": {
#         "console": {
#             "class": "logging.StreamHandler",
#             "formatter": "default",
#         },
#         "file": {
#             "class": "logging.FileHandler",
#             "filename": os.path.join(log_dir, "app.log"),
#             "formatter": "detailed",
#             "level": "DEBUG",
#         },
#     },
#     "loggers": {
#         "logger_config": {  # Change '__main__' to 'logger_config'
#             "handlers": ["console", "file"],
#             "level": "DEBUG",
#             "propagate": True,
#         },
#     },
# }

# Apply logging configuration
# logging.config.dictConfig(LOGGING_CONFIG)

# Get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
