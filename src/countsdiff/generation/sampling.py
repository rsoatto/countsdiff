"""
Sampling utilities and data structures for SNP generation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Dict, Any
from dataclasses import dataclass
from collections.abc import Iterable

from ..training.trainer import CountsdiffTrainer
from ..training.utils import build_legacy_blackout_observation_times

# -----------------------------
# Sigma (remasking) schedules
# -----------------------------

def _random_round(x: torch.Tensor) -> torch.Tensor:
    """Randomized rounding of a float tensor to integers."""
    lower = torch.floor(x)
    prob = x - lower
    rand = torch.rand_like(x)
    return lower + (rand < prob).float()

def _sigma_max(p_t: torch.Tensor, p_s: torch.Tensor) -> torch.Tensor:
    """
    Compute sigma^max_t = min(1, (1 - p_s) / p_t)

    Args:
        p_t: p(t) values
        p_s: p(s) values (next time)
    Returns:
        Tensor of sigma^max values, clamped to [0,1]
    """
    sigma = (1.0 - p_s) / (p_t + 1e-12)
    return torch.clamp(sigma, min=0.0, max=1.0)


def _sigma_max_capped(p_t: torch.Tensor, p_s: torch.Tensor, eta_cap: float) -> torch.Tensor:
    """Capped schedule: min(eta_cap, sigma^max_t)."""
    return torch.clamp(_sigma_max(p_t, p_s), max=float(eta_cap))


def _sigma_rescaled(p_t: torch.Tensor, p_s: torch.Tensor, eta_rescale: float) -> torch.Tensor:
    """Rescaled schedule: eta_rescale * sigma^max_t, capped at 1."""
    return torch.clamp(_sigma_max(p_t, p_s) * float(eta_rescale), max=1.0)


def build_sigma_schedule(
    observation_times: Union[np.ndarray, torch.Tensor],
    p_scheduler,
    method: str = 'max',
    *,
    eta_cap: float = 0.005,
    eta_rescale: float = 0.005,
    device: Optional[Union[str, torch.device]] = None,
) -> List[torch.Tensor]:
    """
    Build a per-step sigma_t schedule from observation times using p_t as analogue of alpha_t.

    For i in [0..T-2], let t_i be observation_times[i] and s_i be observation_times[i+1].
    Returns a list of length T, where the last element is 0 (unused by the sampler loop).

    Supported methods:
      - 'max':        sigma_t = min(1, (1 - p_s) / p_t)
      - 'max_capped': sigma_t = min(eta_cap, sigma^max_t)
      - 'rescaled':   sigma_t = eta_rescale * sigma^max_t
    """
    if isinstance(observation_times, np.ndarray):
        ts = torch.from_numpy(observation_times.astype(np.float32))
    else:
        ts = observation_times.float().clone()
    if device is not None:
        ts = ts.to(device)

    T = ts.shape[0]
    sigmas: List[torch.Tensor] = []
    for i in range(T - 1):
        t = ts[i]
        s = ts[i + 1]
        p_t = p_scheduler(t)
        p_s = p_scheduler(s)
        # ensure tensor
        if not torch.is_tensor(p_t):
            p_t = torch.tensor(p_t, device=ts.device, dtype=torch.float32)
        if not torch.is_tensor(p_s):
            p_s = torch.tensor(p_s, device=ts.device, dtype=torch.float32)

        if method == 'max':
            sigma_i = _sigma_max(p_t, p_s)
        elif method == 'max_capped':
            sigma_i = _sigma_max_capped(p_t, p_s, eta_cap)
        elif method == 'rescaled':
            sigma_i = _sigma_rescaled(p_t, p_s, eta_rescale)
        else:
            raise ValueError(f"Unknown sigma schedule method: {method}")
        sigmas.append(sigma_i)

    # Append a terminal value for compatibility with code that expects len == T
    sigmas.append(torch.tensor(0.0, device=ts.device))
    return sigmas


def resolve_sigma_schedule(
    observation_times: Union[np.ndarray, torch.Tensor],
    p_scheduler,
    remasking_prob: Union[float, Iterable, torch.Tensor],
    *,
    sigma_method: Optional[str] = None,
    sigma_kwargs: Optional[Dict[str, Any]] = None,
    sigma_per_token: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Union[float, torch.Tensor]]:
    """
    Resolve the sigma/remasking schedule for the sampling loop.

    Priority:
      1) If sigma_per_token is provided (list[tensors] or tensor[T-1,...]/tensor[T,...]), return that.
      2) If sigma_method is provided, compute per-step schedule via build_sigma_schedule and return it.
      3) Fallback to remasking_prob (float or iterable) broadcast across steps.

    Returns a list of length len(observation_times). Elements may be floats or tensors
    broadcastable to the state shape used during sampling.
    """
    T = len(observation_times)

    # Case 1: explicit per-token schedule provided
    if sigma_per_token is not None:
        if isinstance(sigma_per_token, list):
            if len(sigma_per_token) == T - 1:
                return sigma_per_token + [0.0]
            elif len(sigma_per_token) == T:
                return sigma_per_token
            else:
                raise ValueError("sigma_per_token list must have length T or T-1")
        elif torch.is_tensor(sigma_per_token):
            if sigma_per_token.dim() == 0:
                # scalar tensor, just broadcast
                return [sigma_per_token.item() for _ in range(T)]
            # Expect shape [T-1, ...] or [T, ...]
            if sigma_per_token.shape[0] == T - 1:
                # pad a terminal zero slice
                pad = torch.zeros_like(sigma_per_token[0])
                return [sigma_per_token[i] for i in range(T - 1)] + [pad]
            elif sigma_per_token.shape[0] == T:
                return [sigma_per_token[i] for i in range(T)]
            else:
                raise ValueError("sigma_per_token tensor must have first dim T or T-1")
        else:
            raise TypeError("sigma_per_token must be list[tensor] or tensor")

    # Case 2: scheduled by method
    if sigma_method is not None:
        sigma_kwargs = sigma_kwargs or {}
        sigmas = build_sigma_schedule(
            p_scheduler=p_scheduler,
            observation_times=observation_times,
            method=sigma_method,
            device=device,
            **sigma_kwargs,
        )
        return sigmas

    # Case 3: fallback to provided remasking_prob
    if isinstance(remasking_prob, Iterable) and not torch.is_tensor(remasking_prob):
        remask_list = list(remasking_prob)
        if remask_list[0] < remask_list[-1]:
            print("Warning: remasking_prob schedule is increasing; likely defined for forward schedule; flipping to decreasing")
        if len(remask_list) not in (T - 1, T):
            raise ValueError("Iterable remasking_prob must have length T or T-1")
        if len(remask_list) == T - 1:
            remask_list = remask_list + [0.0]
        return torch.tensor(remask_list).to(device)
    elif torch.is_tensor(remasking_prob):
        if remasking_prob.dim() == 0:
            return [float(remasking_prob.item()) for _ in range(T)]
        if remasking_prob.shape[0] == T - 1:
            pad = torch.zeros_like(remasking_prob[0])
            return [remasking_prob[i] for i in range(T - 1)] + [pad]
        elif remasking_prob.shape[0] == T:
            return [remasking_prob[i] for i in range(T)]
        else:
            raise ValueError("Tensor remasking_prob must have first dim T or T-1")
    else:
        # scalar
        return torch.tensor([float(remasking_prob) for _ in range(T)]).to(device)


# Import positional_encoding from SNP_training (matching notebook)
def positional_encoding(pos, dim=4):
    """Simple sinusoidal positional encoding."""
    pe = torch.zeros(*(pos.shape[:-1]), dim).to(pos.device)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim)).to(pos.device)
    pe[..., 0::2] = torch.sin(pos * div_term)
    pe[..., 1::2] = torch.cos(pos * div_term)
    return pe


def create_model_input(state, true_pos, ancestry, encoding_dim):
    """Create model input exactly like in the notebook"""
    pos_channel = positional_encoding(true_pos, dim=encoding_dim)
    ancestry = torch.log(ancestry + 1e-8)
    ancestry_channel = ancestry
    normalized_state = (state - state.mean(dim=1, keepdim=True))/(state.std(dim=1, keepdim=True) + 1e-8)
    return torch.cat([normalized_state, pos_channel, ancestry_channel], dim=2).permute(0, 2, 1)


def prepare_generation_input(trainer, state: torch.Tensor, timestep) -> torch.Tensor:
    """
    Prepare model input for generation

    Args:
        state: Current state tensor

    Returns:
        Prepared model input tensor
    """
    n = state.shape[0]
    timestep = float(timestep)
    timestep_tensor = torch.full((n,), timestep, device=state.device, dtype=torch.float32)
    p_t = trainer.p_scheduler(timestep_tensor).squeeze()
    model_input = trainer.normalize_for_model(state, p_t)
    return model_input, timestep_tensor



def backward_step_with_remasking(
    state: torch.Tensor,
    predicted_diff: torch.Tensor,
    s: float,
    t: float,
    remasking_prob: torch.Tensor,
    p_scheduler,
    device: str = 'cuda',
    poisson_approximation=False
):
    sigma_t = remasking_prob
    p_t = p_scheduler(t).to(device)
    p_s = p_scheduler(s).to(device)
    beta_t = (p_s - (1 - sigma_t) * p_t) / (1 - p_t + 1e-12)
    # Binomial bridge sampling
    if poisson_approximation:
        rate_births = beta_t * predicted_diff
        rate_deaths = sigma_t * state
        births = torch.poisson(rate_births)
        deaths = torch.poisson(rate_deaths)
    else:
        births = torch.binomial(predicted_diff, beta_t)
        deaths = torch.binomial(state, sigma_t)

    return state + births - deaths



def backward_step_with_remasking_mixture(
    state: torch.Tensor,
    predicted_diff: torch.Tensor,
    s: float,
    t: float,
    remasking_prob: Union[float, torch.Tensor],
    scheduler,
    device: str = 'cuda'
):
    remask_mask = (torch.rand(state.shape, device=device) < remasking_prob).float()
    standard_mask = (1 - remask_mask)
    predicted_x0 = (predicted_diff + state)

    # Binomial bridge sampling
    p_additive = ((scheduler(s, 0) - scheduler(t, 0)) /
            (1 - scheduler(t, 0)))

    p_additive = torch.full(predicted_diff.shape, p_additive, device=device)
    additive = torch.binomial(predicted_diff, p_additive)
    state = (state + additive)
    p_resample = scheduler(s, 0)
    p_resample = torch.full(predicted_x0.shape, p_resample, device=device)
    resample = torch.binomial(predicted_x0, p_resample)
    state = (state * standard_mask + resample * remask_mask)
    return state
    

def generate_samples(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    observation_times: np.ndarray,
    labels: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    valid_mask: Optional[torch.Tensor],
    trainer: CountsdiffTrainer,
    guidance_scale: float = 0.0,
    remasking_prob: Union[float, Iterable, torch.Tensor] = 0.0,
    *,
    sigma_method: Optional[str] = None,
    sigma_kwargs: Optional[Dict[str, Any]] = None,
    sigma_per_token: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    random_rounding: bool = True,
    poisson_sampling: bool = False,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Generate samples with predictor-free guidance 
    NOTE: Implemented for images i.e. CIFAR-10
    TODO: Extend to SNPs
    Args:
        model: Trained model
        config: Configuration for this level
        initial_state: Initial state tensor
        positions: Position encodings
        valid_mask: Valid position mask
        observation_times: Time steps for generation
        device: Device to use
        guidance_scale: Scale for guidance
        remasking_prob: Probability of remasking
    Returns:
        Generated sample tensor
    """
    model.eval()
    state = initial_state.clone()
    # Resolve sigma (remasking) schedule: supports scalar, list, per-token, or method-based
    sigma_schedule = resolve_sigma_schedule(
        observation_times=observation_times,
        p_scheduler=trainer.p_scheduler,
        remasking_prob=remasking_prob,
        sigma_method=sigma_method,
        sigma_kwargs=sigma_kwargs,
        sigma_per_token=sigma_per_token,
        device=device,
    )
    with torch.no_grad():
        for i, (t, s) in enumerate(zip(observation_times[:-1], observation_times[1:])):
            # Normalize state
            model_input, time_tensor = prepare_generation_input(trainer, state, t)
            # Model prediction
            if trainer.conditional_training:
                full_uncond_mask = torch.ones(model_input.shape[0], dtype=torch.bool, device=model_input.device)
                output_unconditional = model(model_input, time_tensor, class_labels=labels,uncond_mask=full_uncond_mask, valid_mask=valid_mask, xt=state)
                predicted_diff_uncond_log = output_unconditional.log()
                output_conditional = model(model_input, time_tensor, class_labels=labels,uncond_mask=~full_uncond_mask, valid_mask=valid_mask, xt=state)
                predicted_diff_cond_log = output_conditional.log()

                log_predicted_diff_cond = guidance_scale * predicted_diff_cond_log + (1 - guidance_scale) * predicted_diff_uncond_log
                predicted_diff_guided = log_predicted_diff_cond.exp()
            else:
                if guidance_scale not in (0.0, None):
                    print("Warning: guidance_scale ignored in unconditional model")
                if labels not in ([], None) and i == 0:
                    print("Warning: labels ignored in unconditional model")
                output = model(model_input, time_tensor, valid_mask=valid_mask, xt=state)
                predicted_diff_guided = output

            if trainer.dataset_type in ('cifar10','celeba'):
                predicted_diff_guided = predicted_diff_guided.clamp(min=torch.zeros_like(state), max=255.-state)
                
            if random_rounding:
                predicted_diff_guided = _random_round(predicted_diff_guided).squeeze(-1)
            elif poisson_sampling:
                predicted_diff_guided = predicted_diff_guided.squeeze(-1)
            else:
                predicted_diff_guided = predicted_diff_guided.round().squeeze(-1)
                
            if predicted_diff_guided.min() < 0:
                import pdb; pdb.set_trace()

            state = backward_step_with_remasking(state, predicted_diff_guided, s, t, sigma_schedule[i], p_scheduler=trainer.p_scheduler, device=device, poisson_approximation=poisson_sampling)
            
            if valid_mask is not None:
                state = state * valid_mask.float()

            if i % 10 == 0 and verbose:
                print(f"Step {i+1}/{len(observation_times)-1}, max_val: {state.max().item()}, avg_val: {state.mean().item()}")

    return state

def generate_samples_jump(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    observation_times: np.ndarray,
    valid_mask: Optional[torch.Tensor],
    trainer: CountsdiffTrainer,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Generate samples with predictor-free guidance 
    NOTE: Implemented for images i.e. CIFAR-10
    TODO: Extend to SNPs
    Args:
        model: Trained model
        config: Configuration for this level
        initial_state: Initial state tensor
        positions: Position encodings
        valid_mask: Valid position mask
        observation_times: Time steps for generation
        device: Device to use
        guidance_scale: Scale for guidance
        remasking_prob: Probability of remasking
    Returns:
        Generated sample tensor
    """
    model.eval()
    
    assert trainer.pred_target == "x0", \
    "generate_samples_jump assumes the model was trained with pred_target='x0'"
    assert trainer.poisson_randomization, \
        "generate_samples_jump assumes poisson_randomization=True"
    
    state = initial_state.clone()
    # Resolve sigma (remasking) schedule: supports scalar, list, per-token, or method-based
    with torch.no_grad():
        for i, (t, s) in enumerate(zip(observation_times[:-1], observation_times[1:])):
            # Normalize state
            t = float(t)
            s = float(s)
            model_input, time_tensor = prepare_generation_input(trainer, state, t)
            # Model prediction
            output = model(model_input, time_tensor, valid_mask=valid_mask, xt=state, return_val="x0")
            x0_pred = output
            
            p_t = trainer.p_scheduler(torch.tensor(t, device=device)).squeeze()
            p_s = trainer.p_scheduler(torch.tensor(s, device=device)).squeeze()

            x0_pred = x0_pred.clamp(min=torch.zeros_like(state))
            
            rate = trainer.scheduler_config['lbd'] * (p_s - p_t) * x0_pred
            if rate.min() < 0:
                import pdb; pdb.set_trace()
            diff = torch.poisson(rate)

            state = state + diff
            
            if valid_mask is not None:
                state = state * valid_mask.float()

            if i % 10 == 0 and verbose:
                print(f"Step {i+1}/{len(observation_times)-1}, max_val: {state.max().item()}, avg_val: {state.mean().item()}")

    state = state / trainer.scheduler_config['lbd']
    return state


def impute_data(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    observation_times: np.ndarray,
    original_data: torch.Tensor,
    impute_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: Optional[Union[torch.Tensor, List[torch.Tensor]]],
    trainer: CountsdiffTrainer,
    guidance_scale: float = 0.0,
    remasking_prob: Union[float, Iterable, torch.Tensor] = 0.0,
    *,
    sigma_method: Optional[str] = None,
    sigma_kwargs: Optional[Dict[str, Any]] = None,
    sigma_per_token: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    random_rounding: bool = True,
    repaint_num_iters: int = 1,
    repaint_jump: int = 1,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    model.eval()
    state = initial_state.clone()
    sigma_schedule = resolve_sigma_schedule(
        observation_times=observation_times,
        p_scheduler=trainer.p_scheduler,
        remasking_prob=remasking_prob,
        sigma_method=sigma_method,
        sigma_kwargs=sigma_kwargs,
        sigma_per_token=sigma_per_token,
        device=device,
    )
    with torch.no_grad():
        ts = observation_times[:-1]
        ss = observation_times[1:]
        num_steps = len(ts)

        # RePaint with jump scheduling
        num_groups = int(np.ceil(num_steps / repaint_jump))
        for step_group in range(num_groups):
            start = step_group * repaint_jump
            end = min(start + repaint_jump, num_steps)

            # first_t is the highest-noise t in this group; current_s will be updated inside
            first_t = torch.tensor(ts[start], device=device)
            current_s = torch.tensor(ss[start], device=device)  # initialize; updated per j

            for it in range(repaint_num_iters):
                # Walk this jump window: t_j -> s_j for j in [start, end)
                for j in range(start, end):
                    t = ts[j]
                    s = ss[j]
                    model_input, time_tensor = prepare_generation_input(trainer, state, t)

                    if trainer.conditional_training:
                        full_uncond_mask = torch.ones(model_input.shape[0], dtype=torch.bool, device=model_input.device)
                        output_unconditional = model(model_input, time_tensor, class_labels=labels, uncond_mask=full_uncond_mask, valid_mask=valid_mask)
                        predicted_diff_uncond_log = F.softplus(output_unconditional).log()
                        output_conditional = model(model_input, time_tensor, class_labels=labels, uncond_mask=~full_uncond_mask, valid_mask=valid_mask)
                        predicted_diff_cond_log = F.softplus(output_conditional).log()
                        log_pred = guidance_scale * predicted_diff_cond_log + (1 - guidance_scale) * predicted_diff_uncond_log
                        predicted_diff_guided = log_pred.exp()
                    else:
                        if guidance_scale not in (0.0, None) and j == 0:
                            print("Warning: guidance_scale ignored in unconditional model")
                        if labels not in ([], None) and j == 0:
                            print("Warning: labels ignored in unconditional model")
                        output = model(model_input, time_tensor, valid_mask=valid_mask)
                        predicted_diff_guided = F.softplus(output)

                    if trainer.dataset_type in ('cifar10', 'celeba'):
                        predicted_diff_guided = predicted_diff_guided.clamp(min=torch.zeros_like(state), max=255. - state)

                    if random_rounding:
                        predicted_diff_guided = _random_round(predicted_diff_guided).squeeze(-1)
                    else:
                        predicted_diff_guided = predicted_diff_guided.round().squeeze(-1)

                    # Reverse step t -> s
                    state = backward_step_with_remasking(
                        state, predicted_diff_guided, s, t, sigma_schedule[j],
                        p_scheduler=trainer.p_scheduler, device=device
                    )

                    # Re-update known regions with original corrupted data at level s
                    corrupted_s, _ = trainer.corrupt_data(original_data, torch.tensor(s, device=device))
                    state[~impute_mask.bool()] = corrupted_s[~impute_mask.bool()]

                    current_s = torch.tensor(s, device=device)

                # If repainting, forward re-noise s -> first_t and project at t, then repeat
                if it < repaint_num_iters - 1:
                    # Forward re-noise unknown+known to t=first_t
                    state, _ = trainer.corrupt_data(state, t=first_t, current_t=current_s)
                    # PROJECT at level t (known region)
                    corrupted_t, _ = trainer.corrupt_data(original_data, t=first_t)
                    state[~impute_mask.bool()] = corrupted_t[~impute_mask.bool()]

            if verbose and (step_group % max(1, (100 // max(1, repaint_jump))) == 0):
                masked = state[impute_mask.bool()]
                print(f"Repaint step {step_group+1}/{num_groups}, max_val: {masked.max().item()}, avg_val: {masked.mean().item()}")

    return state
