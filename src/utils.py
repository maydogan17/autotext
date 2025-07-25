"""
Utility functions for the AutoML text classification pipeline.

This module contains reusable helper functions used across the project,
including logging setup, device selection, file operations, and other
common utilities.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def setup_logger(
    logger_name: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers with rotation.
    
    Args:
        logger_name: Name of the logger (uses module name if None)
        console_level: Logging level for console output (default: INFO)
        file_level: Logging level for file output (default: DEBUG)
        log_dir: Directory for log files (default: logs/)
    
    Returns:
        Configured logger instance
    """
    from logging.handlers import RotatingFileHandler
    
    # Use module name if no logger name provided
    if logger_name is None:
        logger_name = "autotext"
    
    # Create logger
    logger = logging.getLogger(logger_name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (DEBUG and above)
    if log_dir is None:
        # Default to logs directory in project root
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler (10MB max, keep 5 backups)
    log_file = log_dir / "autotext.log"
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        mode='a'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: The device to use for training (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def create_directory(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """
    Create a directory and all necessary parent directories.
    
    Args:
        path: Path to the directory to create
        exist_ok: If True, don't raise an error if directory already exists
    
    Returns:
        Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
    """
    file_path = Path(file_path)
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    
    Args:
        file_path: Path to the JSON file to load
    
    Returns:
        Dictionary containing the JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_model_size(model: torch.nn.Module) -> int:
    """
    Calculate the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_file_size(file_path: Union[str, Path]) -> str:
    """
    Get the size of a file in a human-readable format.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File size as a formatted string (e.g., "1.5 MB")
    """
    size_bytes = Path(file_path).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False