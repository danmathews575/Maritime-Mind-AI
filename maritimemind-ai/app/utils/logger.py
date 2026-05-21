import logging
import sys
from logging.handlers import RotatingFileHandler
from app.configs.config import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a centralized, structured logger with console and file rotation support.
    
    Args:
        name (str): The name of the logger (typically __name__ of the calling module).
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    
    # Prevent adding handlers multiple times if instantiated repeatedly
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. Rotating File Handler
        file_handler = RotatingFileHandler(
            filename=settings.LOGS_DIR / "maritimemind.log",
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
