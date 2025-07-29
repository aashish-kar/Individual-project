# src/utils/logger.py

import logging
import os
from datetime import datetime

def setup_logging(log_file='app.log', log_level=logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_file (str): The name of the log file. Default is 'app.log'.
        log_level (int): The logging level. Default is logging.INFO.
    """
    # Create a logs directory if it doesn't exist
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Full path to the log file
    log_file_path = os.path.join(logs_dir, log_file)

    # Create a logger object
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)

    # Stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log the start of the application
    logger.info(f"Logging initialized. Logs will be written to {log_file_path}.")

# Example usage
if __name__ == "__main__":
    setup_logging()
    logging.info("This is an info message.")
    logging.error("This is an error message.")
