"""
CountsDiff Package

A modular package for training and generating count-based data using diffusion models.
"""

__version__ = "0.1.0"
__author__ = "CountsDiff Team"

# Core models and data
from .models.diffusion import Countsdiff
from .config.config import Config

# Training components
from .training.trainer import CountsdiffTrainer
from .training import utils as training_utils

# Generation components  
from .generation.generator import CountsdiffGenerator
from .generation import sampling

# Utilities
from .utils import checkpoints, metrics, logging

# CLI interface
from . import cli

__all__ = [
    # Core components
    "Countsdiff",
    "Config",
    
    # Training
    "CountsdiffTrainer", 
    "training_utils",
    
    # Generation
    "CountsdiffGenerator",
    "sampling",
    
    # Utilities
    "checkpoints",
    "metrics", 
    "logging",
    
    # CLI
    "cli"
]
