"""
SNP Blackout Diffusion Package

A modular package for training and generating SNP data using blackout diffusion models.
Supports hierarchical generation across different SNP levels (0, 1, 2).
"""

__version__ = "0.1.0"
__author__ = "SNP Diffusion Team"
__email__ = "team@snpdiffusion.com"

# Core models and data
from .models.diffusion import Countsdiff
# from .data.datasets import SNPDataset
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
    "SNPDataset",
    "Config",
    
    # Training
    "CountsdiffTrainer", 
    "training_utils",
    
    # Generation
    "SNPGenerator",
    "sampling",
    
    # Utilities
    "checkpoints",
    "metrics", 
    "logging",
    
    # CLI
    "cli"
]
