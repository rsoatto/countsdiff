"""
Logging utilities for SNP diffusion package
"""

import logging
import os
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = 'INFO',
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Optional file to write logs to
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('countsdiff')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Logger specifically for training metrics"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = setup_logging(log_file, 'INFO')
        self.step = 0
    
    def log_step(
        self,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log training step metrics"""
        self.step = step
        
        msg = f"Step {step}: Train Loss = {train_loss:.6f}"
        
        if val_loss is not None:
            msg += f", Val Loss = {val_loss:.6f}"
        
        for key, value in kwargs.items():
            msg += f", {key} = {value:.6f}"
        
        self.logger.info(msg)
    
    def log_epoch(
        self,
        epoch: int,
        avg_train_loss: float,
        avg_val_loss: Optional[float] = None
    ) -> None:
        """Log epoch summary"""
        msg = f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.6f}"
        
        if avg_val_loss is not None:
            msg += f", Avg Val Loss = {avg_val_loss:.6f}"
        
        self.logger.info(msg)
    
    def log_checkpoint(self, checkpoint_path: str) -> None:
        """Log checkpoint save"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def log_completion(self, total_steps: int, final_loss: float) -> None:
        """Log training completion"""
        self.logger.info(f"Training completed after {total_steps} steps. Final loss: {final_loss:.6f}")
