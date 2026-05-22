import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from app.configs.config import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with standard formatting,
    file rotation, and console output.
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, it means it was already configured
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_file_path = Path(settings.LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert string log level from config to logging level
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=settings.LOG_FILE,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Avoid propagating to root logger to prevent duplicate logs
    logger.propagate = False

    return logger
