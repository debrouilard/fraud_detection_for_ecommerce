"""
Logging utilities for the fraud detection system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from src.config import logging_config


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = None
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Optional log file name (will be saved in logs directory)
        level: Logging level (defaults to config)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level or logging_config.log_level
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(logging_config.log_format)
    
    # Console handler
    if logging_config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if logging_config.log_to_file and log_file:
        log_path = logging_config.log_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
