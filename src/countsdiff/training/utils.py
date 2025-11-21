"""
Training utilities for SNP diffusion models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math
import scipy.optimize


def cos_p_scheduler(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine probability scheduler for blackout diffusion
    
    Args:
        t: Time steps in [0, 1]
        
    Returns:
        Survival probabilities
    """
    return torch.cos(t * np.pi / 2) ** 2


def weight_scheduler(t: torch.Tensor) -> torch.Tensor:
    """
    Weight scheduler for loss weighting
    
    Args:
        t: Time steps in [0, 1]
        
    Returns:
        Loss weights
    """
    return torch.pi/2 * torch.sin(t * torch.pi)


# -----------------------------
# Legacy blackout schedule helpers (from notebook)
# -----------------------------

def build_legacy_blackout_observation_times(T: int, t_end: float) -> torch.Tensor:
    """
    Build fixed observation times matching the blackout notebook.

    Derivation in notebook:
      - Define f(x) = log(x/(1-x)), sample f uniformly in [-f(x_end), f(x_end)],
        where x_end = exp(-t_end), and map back x = sigmoid(f).
      - observation_times = -log(x) in [0, t_end].

    Args:
      T: number of discrete time indices
      t_end: maximum continuous time
    Returns:
      Tensor of shape [T] with monotonically increasing observation times.
    """
    # x_end = exp(-t_end)
    x_end = math.exp(-float(t_end))
    # f spans symmetrically around 0
    f_max = math.log(x_end / (1.0 - x_end))
    f_grid = torch.linspace(-f_max, f_max, steps=int(T), dtype=torch.float32)
    # x = sigmoid(f); t = -log(x)
    x_grid = torch.sigmoid(f_grid)
    obs_times = -torch.log(x_grid)
    return obs_times


def build_legacy_blackout_weights(observation_times: torch.Tensor, sampling_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute per-index weights matching the notebook: pt * (dt) / pi.

    Args:
      observation_times: tensor[T]
      sampling_prob: tensor[T] or None (defaults to uniform)
    Returns:
      weights: tensor[T] suitable to index by tIndex and broadcast across samples.
    """
    device = observation_times.device
    T = observation_times.shape[0]
    # Prepend t=0 as in notebook (eobservationTimes)
    eobs = torch.cat([torch.zeros(1, device=device, dtype=observation_times.dtype), observation_times])
    pt = torch.exp(-eobs[1:])  # pt = exp(-observationTimes)
    dt = (eobs[1:] - eobs[:-1])
    if sampling_prob is None:
        sampling_prob = torch.full((T,), 1.0 / T, device=device, dtype=observation_times.dtype)
    weights = pt * dt / sampling_prob
    return weights


def legacy_blackout_config_to_buffers(cfg: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience: build observation_times, sampling_prob (uniform), weights for CIFAR training.
    Reads scheduler settings from cfg dict with keys:
      cfg['scheduler']['T'], cfg['scheduler']['t_end']
    Returns tensors on the given device.
    """
    T = int(cfg.get('scheduler', {}).get('T', 1000))
    t_end = float(cfg.get('scheduler', {}).get('t_end', 15.0))
    obs = build_legacy_blackout_observation_times(T, t_end).to(device)
    pi = torch.full((T,), 1.0 / T, device=device, dtype=obs.dtype) 
    w = build_legacy_blackout_weights(obs, sampling_prob=pi)
    return obs, pi, w

def log_sigmoid(x):
    if x < -9:
        out = x
    elif x > 9:
        out = -np.exp(-x)
    else:
        out = -np.log(1 + np.exp(-x))
    return out

def infer_beta_end_from_logsnr(logsnr_end, beta_start, lbd, signal_stat, timesteps=1000):
    def logsnr_fn(beta_end):
        a, b = 1 - beta_start, 1 - beta_end
        fb = 0.5 * timesteps * ((b * math.log(b) - a * math.log(a)) / (b - a) - 1)
        target = logsnr_end - np.log(lbd) - np.log(signal_stat)
        return fb - target
    return scipy.optimize.fsolve(logsnr_fn, np.array(0.00015), xtol=1e-6, maxfev=1000)[0].item()  # noqa


def infer_lbd_from_logsnr(logsnr_start: float, signal_stat: float = 1.0) -> float:
    """Borrowed from JUMP (Chen et al. 2023)"""
    return math.exp(logsnr_start) / float(signal_stat)


def infer_alpha_from_beta(betas: torch.Tensor) -> torch.Tensor:
    """
    Given betas of shape (T,), compute alphas of shape (T+1,)
    such that:
        alpha[0] = 1
        alpha[t] = sqrt( prod_{s=1..t} (1 - beta_s) )
    """
    if not isinstance(betas, torch.Tensor):
        betas = torch.as_tensor(betas, dtype=torch.float64)

    alphas_sq = torch.cumprod(1.0 - betas, dim=0)

    # prepend alpha_0 = 1
    alphas_sq = torch.cat([torch.ones(1, dtype=alphas_sq.dtype, device=alphas_sq.device),
                           alphas_sq])

    return alphas_sq.sqrt()


def build_jump_linear_beta_schedule(
    T: int,
    beta_start: float = 1e-3,
    logsnr_start: float = 10.0,
    logsnr_end: float = -12.0,
    signal_stat: float = 1.0,
    lbd: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build a JUMP/DDPM-like linear beta schedule characterized by:
        - beta_start (default 0.001)
        - logsnr_end: desired logSNR at the final step T (default -12)
        - logsnr_start: logSNR at t = 0 (default 10)
        - signal_stat: typical signal scale (dataset dependent)

    Returns:
        betas:  (T,) tensor of betas
        alphas: (T+1,) tensor of alphas
        lbd:    scalar lambda used in the logSNR definition
    """
    # 1) infer λ from starting logSNR and signal scale
    if lbd is None:
        lbd = infer_lbd_from_logsnr(logsnr_start=logsnr_start, signal_stat=signal_stat)

    # 2) pick β_T so that terminal logSNR matches logsnr_end
    beta_end = infer_beta_end_from_logsnr(
        logsnr_end=logsnr_end,
        beta_start=beta_start,
        lbd=lbd,
        signal_stat=signal_stat,
        timesteps=T,
    )

    # 3) linear beta schedule as in DDPM
    betas = torch.linspace(beta_start, beta_end, steps=T, dtype=torch.float64)

    # 4) alpha_t from beta_t
    alphas = infer_alpha_from_beta(betas)

    # cast to requested dtype/device
    if device is not None:
        betas = betas.to(device)
        alphas = alphas.to(device)
    betas = betas.to(dtype=dtype)
    alphas = alphas.to(dtype=dtype)

    return betas, alphas, lbd



def positional_encoding(pos: torch.Tensor, dim: int = 4) -> torch.Tensor:
    """
    Sinusoidal positional encoding
    
    Args:
        pos: Position tensor
        dim: Encoding dimension
        
    Returns:
        Positional encoding tensor
    """
    encoding = torch.zeros(*pos.shape, dim)
    position = pos.unsqueeze(-1)
    
    div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                        -(np.log(10000.0) / dim))
    
    encoding[..., 0::2] = torch.sin(position * div_term)
    encoding[..., 1::2] = torch.cos(position * div_term)
    
    return encoding


def generate_batch_data_with_ancestry(
    batch_data: Tuple[torch.Tensor, ...],
    p_scheduler: callable = cos_p_scheduler,
    device: str = 'cuda',
    pos_encoding: str = 'sinusoidal',
    encoding_dim: int = 4,
    start_pos: int = 0
) -> Tuple[torch.Tensor, ...]:
    """
    Generate batch data for diffusion training with ancestry conditioning
    
    Args:
        batch_data: Tuple of (vals, true_pos, dataset_pos, valid_mask, ancestry)
        p_scheduler: Probability scheduler function
        device: Device to use
        pos_encoding: Type of positional encoding ('sinusoidal' or 'absolute')
        encoding_dim: Dimension for encoding
        start_pos: Starting position offset
        
    Returns:
        Tuple of (model_input, noised, level_sum, diff_batch, timesteps, valid_mask)
    """
    vals, true_pos, dataset_pos, valid_mask, ancestry = batch_data
    
    # Move to device
    vals = vals.to(device).float()
    true_pos = true_pos.to(device).float()
    dataset_pos = dataset_pos.to(device).float()
    valid_mask = valid_mask.to(device).float()
    ancestry = ancestry.to(device).float()
    
    batch_size, seq_len = vals.shape
    
    # Generate random timesteps for the batch
    timesteps = torch.rand(batch_size, device=device)

    # Calculate probability of keeping signal based on timestep
    p_t = p_scheduler(timesteps.view(batch_size, 1)).expand(vals.shape)

    # Apply binomial noise (blackout diffusion)
    noised = torch.binomial(count=vals, prob=p_t)

    # Calculate mean value for centering
    noised_normalized = (noised - noised.mean(dim=1, keepdim=True)) / (
        noised.std(dim=1, keepdim=True) + 1e-8
    )

    # Prepare positional encoding
    if pos_encoding == 'sinusoidal':
        positions = dataset_pos.squeeze(-1) + start_pos
        pos_channel = positional_encoding(positions, encoding_dim).to(device)
    else:  # absolute
        pos_channel = (dataset_pos + start_pos).expand(-1, seq_len, -1).to(device)

    # Prepare ancestry conditioning
    ancestry_channel = ancestry.unsqueeze(1).repeat(1, seq_len, 1)

    # Stack channels: [noised_normalized, pos_channel, ancestry_channel]
    model_input = torch.cat([
        noised_normalized.unsqueeze(2),
        pos_channel,
        ancestry_channel
    ], dim=2).permute(0, 2, 1)

    # Difference between original and noised data is our target
    diff_batch = vals - noised
    
    # Calculate level sum for sum loss
    level_sum = vals.sum(dim=1)

    return model_input, noised, level_sum, diff_batch, timesteps, valid_mask
