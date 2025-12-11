#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
        name: str = "calibclip",
        save_dir: Optional[str] = None,
        filename: str = "log.txt",
        level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        save_dir: Directory to save log file
        filename: Log filename
        level: Logging level

    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(save_dir, filename), mode="a"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "calibclip") -> logging.Logger:
    """
    Get logger by name.

    Args:
        name: Logger name

    Returns:
        logger: Logger instance
    """
    return logging.getLogger(name)
