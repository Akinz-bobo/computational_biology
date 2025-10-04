"""
Logging utilities for the WNV analysis pipeline
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "wnv_analysis",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressLogger:
    """Logger for tracking progress in long-running operations"""
    
    def __init__(self, logger: logging.Logger, total: int, description: str = "Processing"):
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0
        self.last_reported_percent = -1
    
    def update(self, n: int = 1):
        """Update progress by n items"""
        self.current += n
        percent = int(100 * self.current / self.total)
        
        # Report progress at 10% intervals
        if percent >= self.last_reported_percent + 10:
            self.logger.info(f"{self.description}: {percent}% ({self.current}/{self.total})")
            self.last_reported_percent = percent
    
    def finish(self):
        """Mark progress as complete"""
        self.logger.info(f"{self.description}: Complete! ({self.total}/{self.total})")