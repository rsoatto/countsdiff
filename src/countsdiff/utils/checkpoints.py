"""
Checkpoint handling utilities extracted from utils.py
"""

import torch
import os
from typing import Dict, Any, Optional


def save_checkpoint_with_sum(checkpoint_path: str, state: Dict[str, Any]) -> None:
    """
    Save checkpoint including sum loss history
    
    Args:
        checkpoint_path: Path to save checkpoint
        state: State dictionary containing model, optimizer, and training state
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': state['model'].state_dict(),
        'optimizer_state_dict': state['optimizer'].state_dict(),
        'ema_state_dict': state['ema'].state_dict() if hasattr(state['ema'], 'state_dict') else None,
        'step': state['step'],
        'lossHistory': state['lossHistory'],
        'blackoutLossHistory': state['blackoutLossHistory'],
        'sumLossHistory': state['sumLossHistory'],
        'evalLossHistory': state['evalLossHistory'],
        'blackoutEvalLossHistory': state['blackoutEvalLossHistory'],
        'sumEvalLossHistory': state['sumEvalLossHistory']
    }
    
    torch.save(checkpoint, checkpoint_path)


def restore_checkpoint_with_sum(
    checkpoint_path: str, 
    state: Dict[str, Any], 
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Restore checkpoint including sum loss history
    
    Args:
        checkpoint_path: Path to checkpoint file
        state: Current state dictionary to update
        device: Device to load tensors to
        
    Returns:
        Updated state dictionary
    """
    if not os.path.exists(checkpoint_path):
        return state
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model and optimizer states
    if 'model' in state and 'model_state_dict' in checkpoint:
        state['model'].load_state_dict(checkpoint['model_state_dict'])
    
    if 'optimizer' in state and 'optimizer_state_dict' in checkpoint:
        state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'ema' in state and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
        if hasattr(state['ema'], 'load_state_dict'):
            state['ema'].load_state_dict(checkpoint['ema_state_dict'])
    
    # Load training history
    history_keys = [
        'step', 'lossHistory', 'blackoutLossHistory', 'sumLossHistory',
        'evalLossHistory', 'blackoutEvalLossHistory', 'sumEvalLossHistory'
    ]
    
    for key in history_keys:
        if key in checkpoint:
            state[key] = checkpoint[key]
    
    return state


def save_checkpoint_withEval(checkpoint_path: str, state: Dict[str, Any]) -> None:
    """Legacy function name for backward compatibility"""
    return save_checkpoint_with_sum(checkpoint_path, state)


def restore_checkpoint_withEval(
    checkpoint_path: str, 
    state: Dict[str, Any], 
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Legacy function name for backward compatibility"""
    return restore_checkpoint_with_sum(checkpoint_path, state, device)
