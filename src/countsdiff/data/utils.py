"""
Data utility functions
"""

import numpy as np
import torch
from typing import Tuple


def positional_encoding(pos: torch.Tensor, dim: int = 4) -> torch.Tensor:
    """
    Simple sinusoidal positional encoding
    
    Args:
        pos: Position tensor
        dim: Encoding dimension
        
    Returns:
        Positional encoding tensor
    """
    pe = torch.zeros(*(pos.shape[:-1]), dim).to(pos.device)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim)).to(pos.device)
    pe[..., 0::2] = torch.sin(pos * div_term)
    pe[..., 1::2] = torch.cos(pos * div_term)
    return pe


def cos_p_scheduler(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine probability scheduler for blackout diffusion
    
    Args:
        t: Timestep tensor
        
    Returns:
        Probability tensor
    """
    t = t.reshape(-1, 1)
    p_t = torch.cos(t * np.pi / 2) ** 2
    return p_t


def weight_scheduler(t: torch.Tensor) -> torch.Tensor:
    """
    Weight scheduler for loss weighting
    
    Args:
        t: Timestep tensor
        
    Returns:
        Weight tensor
    """
    return torch.pi / 2 * torch.sin(t * torch.pi)


def generate_batch_data_with_ancestry(
    batch_data: Tuple[torch.Tensor, ...], 
    p_scheduler=cos_p_scheduler, 
    device: str = 'cuda', 
    pos_encoding: str = 'absolute',
    encoding_dim: int = 4, 
    start_pos: int = 0
) -> Tuple[torch.Tensor, ...]:
    """
    Generate batch data for diffusion with ancestry conditioning
    
    Args:
        batch_data: Tuple of (vals, true_pos, dataset_pos, valid_mask, ancestry)
        p_scheduler: Probability scheduler function
        device: Device to use
        pos_encoding: Position encoding type ('absolute' or 'sinusoidal')
        encoding_dim: Encoding dimension for sinusoidal
        start_pos: Starting position for absolute encoding
        
    Returns:
        Tuple of (model_input, noised, level_sum, diff_batch, timesteps, valid_mask)
    """
    vals, true_pos, dataset_pos, valid_mask, ancestry = batch_data

    batch_size = vals.shape[0]
    seq_len = vals.shape[1]
    
    # Move data to device
    vals = vals.to(device).float()
    true_pos = true_pos.to(device).float()
    valid_mask = valid_mask.to(device).float()
    ancestry = ancestry.to(device).float()
    level_sum = ancestry[:, -1]
    ancestry = torch.log(ancestry + 1e-8)  # Log transform ancestry

    # Process position data based on encoding type
    if pos_encoding == 'sinusoidal':
        # Generate positional encoding from true_pos
        pos_channel = positional_encoding(true_pos, dim=encoding_dim).to(device)
    else:  # absolute (default)
        pos_channel = true_pos - start_pos
        encoding_dim = 1

    # Generate random timesteps for the batch
    timesteps = torch.rand(batch_size).to(device)

    # Calculate probability of keeping signal based on timestep
    p_t = p_scheduler(timesteps.view(batch_size, 1, 1)).expand(*vals.shape)

    # Apply binomial noise (blackout diffusion)
    noised = torch.binomial(count=vals, prob=p_t)

    # Calculate mean value for centering
    noised_normalized = (noised - noised.mean(dim=1, keepdim=True)) / (noised.std(dim=1, keepdim=True) + 1e-8)

    # Prepare ancestry conditioning
    # Smear ancestry across a new channel
    ancestry_channel = ancestry.unsqueeze(1).repeat(1, seq_len, 1)

    # Stack channels: [noised_normalized, pos_channel, ancestry_channel]
    # For the model input, we want shape [batch_size, channels, seq_len]
    if pos_encoding == 'sinusoidal':
        model_input = torch.cat([
            noised_normalized.unsqueeze(2),
            pos_channel,
            ancestry_channel
        ], dim=2).permute(0, 2, 1)
    else:  # absolute
        model_input = torch.cat([
            noised_normalized.unsqueeze(2),
            pos_channel,
            ancestry_channel
        ], dim=2).permute(0, 2, 1)
    
    # Difference between original and noised data is our target
    diff_batch = vals - noised

    return model_input, noised, level_sum, diff_batch, timesteps, valid_mask
