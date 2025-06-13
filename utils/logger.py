import logging
import sys
import os
from pathlib import Path

def setup_logger(name="tiktok_agent", log_file=None, level=None):
    """Set up and return a logger instance"""
    
    # Get log level from environment variable, with fallback to INFO
    if level is None:
        level_name = os.environ.get("TIKTOK_AGENT_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to prevent duplicates when reconfiguring
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()

# Add a function to change log level dynamically
def set_log_level(level):
    """Set the log level for the logger
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger.setLevel(level)