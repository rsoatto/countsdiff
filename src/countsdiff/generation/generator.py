"""
Main generator class for SNP diffusion models
"""

import os
import pickle
import torch
import numpy as np
from typing import Dict, Iterable, List, Any, Optional, Union

from .sampling import generate_samples, generate_samples_jump
from ..training import trainer
from ..config.config import Config


class CountsdiffGenerator:
    """Generator for image-based data using trained blackout diffusion models"""
    def __init__(
        self,
        input_trainer: trainer.CountsdiffTrainer = None,
        run_id: Optional[str] = None,
        legacy_model: bool = False,
        config_override: Optional[Dict[str, Any]] = None,
        checkpoint = None,
        device: str = 'cuda'
    ):
        self.device = device
        if input_trainer is None:
            if run_id is None:
                raise ValueError("Either input_trainer or run_id must be provided")
            if config_override is not None:
                self.config = config_override
            else:
                self.config = Config.load_from_neptune(run_id, project_name='countsdiff-iclr/ICLR')
            self.run_id = run_id

            self.trainer = trainer.CountsdiffTrainer(self.config, run_id=run_id, legacy_model=legacy_model, eval_mode=True)
            self.trainer.setup_model()
            self.trainer.setup_data()
            self.load_checkpoint(checkpoint)
        else:
            self.trainer = input_trainer
            self.device = self.trainer.device
            self.config = self.trainer.config
        self.model = self.trainer.model.to(self.device)
        self.val_loader = self.trainer.val_loader
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_number: Optional[int] = None):
        if checkpoint_number is not None:
            filename = f"checkpoint_{checkpoint_number}.pth"
        else:
            best_filename = f'{self.trainer.dataset_type}_best.pth'
            if os.path.exists(os.path.join(self.trainer.checkpoint_dir, best_filename)):
                filename = best_filename
            else:
                filename = f'{self.trainer.dataset_type}_latest.pth'
        checkpoint_path = os.path.join(self.trainer.checkpoint_dir, filename)

        if os.path.exists(checkpoint_path):
            self.trainer.load_checkpoint(checkpoint_path)
            print(f"Loaded model checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def generate_samples(self, num_samples: int, n_steps: int = 1000, device='cuda', remasking_prob = 0.00, guidance_scale = 0.0, labels=None, valid_mask=None, batch_size=None, random_rounding=True, poisson_sampling: bool = False,
                        *, sigma_method: Optional[str] = None, sigma_kwargs: Optional[Dict[str, Any]] = None, sigma_per_token: Optional[torch.Tensor] = None, **kwargs) -> np.ndarray:
        """
        Generate image samples using the blackout diffusion model
        
        Args:
            num_samples: Number of samples to generate
            n_steps: Number of generation steps
            batch_size: Batch size for generation (if None, generates all at once)
            
        Returns:
            Generated samples as a numpy array
        """
        from .sampling import generate_samples

        if labels is None and self.trainer.conditional_training:
            print("Warning: Model was trained conditionally but no labels were provided for generation. Using val dataset labels.")
            batch = next(iter(torch.utils.data.DataLoader(self.trainer.val_loader.dataset, batch_size=num_samples, shuffle=True)))
            if len(self.trainer.val_loader.dataset) >= num_samples:
                labels = batch[1]
                if isinstance(labels, list):
                    labels = [label.to(device) for label in labels]
            else:
                # Need to repeat labels to match num_samples
                bootstrapped_indices = np.random.choice(len(self.trainer.val_loader.dataset), size=num_samples, replace=True)
                bootstrapped_indices = torch.from_numpy(bootstrapped_indices).long()
                if isinstance(batch[1], list):
                    labels = [label[bootstrapped_indices].to(device) for label in batch[1]]
                elif isinstance(batch[1], torch.Tensor):
                    labels = batch[1][bootstrapped_indices].to(device)

        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            self.device = device
            self.trainer.ema.store(self.model.parameters())
            self.trainer.ema.copy_to(self.model.parameters())
            
            if self.trainer.continuous_training == False:
                assert n_steps == self.trainer.T, "n_steps must match training n_steps for discrete training"
            
            observation_times = torch.from_numpy(np.linspace(1, 0, n_steps + 1).astype(np.float32))
            sample_image = self.trainer.val_loader.dataset[0]
            if isinstance(sample_image, (list, tuple)):
                sample_image = sample_image[0]
            img_shape = sample_image.shape
            
            # If no batch_size specified, generate all at once
            if batch_size is None:
                initial_state = torch.zeros((num_samples, *img_shape), device=device)
                if labels is not None:
                    if isinstance(labels, list) or isinstance(labels, tuple):
                        labels = [cat_labels.to(device) for cat_labels in labels]
                    elif isinstance(labels, torch.Tensor):
                        labels = labels.to(device)
                    else:
                        raise ValueError("Labels must be a torch Tensor or a list/tuple of Tensors")
                    
                if self.trainer.poisson_randomization:
                    samples = generate_samples_jump(
                        model=self.model,
                        initial_state=initial_state,
                        observation_times=observation_times,
                        valid_mask=valid_mask,
                        device=str(self.device),
                        trainer=self.trainer
                    )
                else:
                    samples = generate_samples(
                        model=self.model,
                        initial_state=initial_state,
                        remasking_prob=remasking_prob,
                        observation_times=observation_times,
                        labels=labels,
                        valid_mask=valid_mask,
                        guidance_scale=guidance_scale,
                        random_rounding=random_rounding,
                        poisson_sampling=poisson_sampling,
                        sigma_method=sigma_method,
                        sigma_kwargs=sigma_kwargs,
                        sigma_per_token=sigma_per_token,
                        device=str(self.device),
                        trainer=self.trainer
                    )
                    all_samples = samples
            else:
                # Batched generation
                all_samples = []
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    current_batch_size = end_idx - i
                    
                    # Create batch initial state
                    initial_state = torch.zeros((current_batch_size, *img_shape), device=device)
                    batch_labels = None
                    if labels is not None:
                        if isinstance(labels, list) or isinstance(labels, tuple):
                            batch_labels = [cat_labels[i:end_idx].to(device) for cat_labels in labels]
                        elif isinstance(labels, torch.Tensor):
                            batch_labels = labels[i:end_idx].to(device)
                    
                    # Generate batch
                    if self.trainer.poisson_randomization:
                        batch_samples = generate_samples_jump(
                            model=self.model,
                            initial_state=initial_state,
                            observation_times=observation_times,
                            valid_mask=valid_mask[i:end_idx] if valid_mask is not None else None,
                            device=str(self.device),
                            trainer=self.trainer
                        )
                    else:
                        batch_samples = generate_samples(
                            model=self.model,
                            initial_state=initial_state,
                            remasking_prob=remasking_prob,
                            observation_times=observation_times,
                            labels=batch_labels,
                            valid_mask=valid_mask[i:end_idx] if valid_mask is not None else None,
                            guidance_scale=guidance_scale,
                            random_rounding=random_rounding,
                            poisson_sampling=poisson_sampling,
                            sigma_method=sigma_method,
                            sigma_kwargs=sigma_kwargs,
                            sigma_per_token=sigma_per_token,
                            device=str(self.device),
                            trainer=self.trainer
                        )
                    
                    all_samples.append(batch_samples)
                
                # Stack all batches
                all_samples = torch.cat(all_samples, dim=0)
            
            self.trainer.ema.restore(self.model.parameters())
        self.model.train()
        return all_samples.cpu().numpy()


    
    def impute_data(self, counts, impute_mask: torch.tensor, n_steps: int = 1000, device='cuda', remasking_prob = 0.0, guidance_scale = 1.5, labels=None, valid_mask=None, batch_size=None, random_rounding=True, repaint_num_iters:int=1, repaint_jump:int=1,
                    *, sigma_method: Optional[str] = None, sigma_kwargs: Optional[Dict[str, Any]] = None, sigma_per_token: Optional[torch.Tensor] = None, verbose:bool=False) -> np.ndarray:
        """
        Generate image samples using the blackout diffusion model
        
        Args:
            num_samples: Number of samples to generate
            n_steps: Number of generation steps
            batch_size: Batch size for generation (if None, generates all at once)
            
        Returns:
            Generated samples as a numpy array
        """
        from .sampling import impute_data
        
        
        self.model.to(device)
        self.model.eval()
        if self.trainer.poisson_randomization:
            raise AssertionError("Imputation with poisson randomization not supported yet")
        with torch.no_grad():
            self.device = device
            self.trainer.ema.store(self.model.parameters())
            self.trainer.ema.copy_to(self.model.parameters())
            
            observation_times = torch.from_numpy(np.linspace(1, 0, n_steps).astype(np.float32))

            # Allow model to attend on imputed points
            valid_mask = torch.ones_like(counts, dtype=torch.bool) if valid_mask is None else (valid_mask + impute_mask).bool()
    
            # If no batch_size specified, generate all at once
            if batch_size is None:
                if labels is not None:
                    if isinstance(labels, list) or isinstance(labels, tuple):
                        labels = [cat_labels.to(device) for cat_labels in labels]
                    elif isinstance(labels, torch.Tensor):
                        labels = labels.to(device)
                    else:
                        raise ValueError("Labels must be a torch Tensor or a list/tuple of Tensors")
                initial_state = torch.zeros_like(counts)
                samples = impute_data(
                    model=self.model,
                    initial_state=initial_state,
                    original_data=counts.to(device).float(),
                    impute_mask=impute_mask.to(device).bool(),
                    valid_mask=valid_mask,
                    remasking_prob=remasking_prob,
                    observation_times=observation_times,
                    labels=labels,
                    guidance_scale=guidance_scale,
                    repaint_num_iters=repaint_num_iters,
                    repaint_jump=repaint_jump,
                    sigma_method=sigma_method,
                    sigma_kwargs=sigma_kwargs,
                    sigma_per_token=sigma_per_token,
                    random_rounding=random_rounding,
                    device=str(self.device),
                    trainer=self.trainer,
                    verbose=verbose
                )
                all_samples = samples
            else:
                # Batched data imputation
                all_samples = []
                    
                for i in range(0, counts.shape[0], batch_size):
                    end_idx = min(i + batch_size, counts.shape[0])
                    batch_labels = None
                    if labels is not None:
                        if isinstance(labels, list) or isinstance(labels, tuple):
                            batch_labels = [cat_labels[i:end_idx].to(device) for cat_labels in labels]
                        elif isinstance(labels, torch.Tensor):
                            batch_labels = labels[i:end_idx].to(device)
                    # Create batch initial state
                    initial_state = torch.zeros_like(counts[i:end_idx]).to(device)

                    # Slice batch tensors
                    counts_batch = counts[i:end_idx].to(device).float()
                    impute_mask_batch = impute_mask[i:end_idx].to(device).bool()
                    if valid_mask is not None:
                        valid_mask_batch = valid_mask[i:end_idx].to(device).bool()
                    else:
                        valid_mask_batch = torch.ones_like(counts_batch, dtype=torch.bool, device=device)
                    # Generate batch imputations
                    batch_samples = impute_data(
                        model=self.model,
                        initial_state=initial_state,
                        original_data=counts_batch,
                        impute_mask=impute_mask_batch,
                        valid_mask=valid_mask_batch,
                        remasking_prob=remasking_prob,
                        observation_times=observation_times,
                        labels=batch_labels,
                        guidance_scale=guidance_scale,
                        sigma_method=sigma_method,
                        sigma_kwargs=sigma_kwargs,
                        sigma_per_token=sigma_per_token,
                        random_rounding=random_rounding,
                        device=str(self.device),
                        trainer=self.trainer
                    )

                    all_samples.append(batch_samples)

                # Stack all batches
                all_samples = torch.cat(all_samples, dim=0)
            
            self.trainer.ema.restore(self.model.parameters())
        self.model.train()
        return all_samples.cpu().numpy()
    
    
class CountsdiffImputer:
    def __init__(self, input_trainer: trainer.CountsdiffTrainer = None,
        run_id: Optional[str] = None,
        legacy_model: bool = False,
        config_override: Optional[Dict[str, Any]] = None,
        checkpoint = None,
        n_steps: int = 1000, 
        device='cuda', 
        remasking_prob = 0.00, 
        guidance_scale = 0.0, 
        batch_size=None, 
        random_rounding=True,
        *, 
        sigma_method: Optional[str] = None, 
        sigma_kwargs: Optional[Dict[str, Any]] = None, 
        sigma_per_token: Optional[torch.Tensor] = None, 
        repaint_num_iters:int=1,
        repaint_jump:int=1,
        **kwargs):

        self.n_steps = n_steps
        self.device = device
        self.remasking_prob = remasking_prob
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.random_rounding = random_rounding
        self.sigma_method = sigma_method
        self.sigma_kwargs = sigma_kwargs
        self.sigma_per_token = sigma_per_token
        self.repaint_num_iters = repaint_num_iters
        self.repaint_jump = repaint_jump
        self.kwargs = kwargs
        self.generator = CountsdiffGenerator(input_trainer=input_trainer,
            run_id=run_id,
            legacy_model=legacy_model,
            config_override=config_override,
            checkpoint=checkpoint,
            device=device)

    def update_hyperparameters(self, n_steps: Optional[int] = None, device: Optional[str] = None, remasking_prob: Optional[float] = None, guidance_scale: Optional[float] = None, batch_size: Optional[int] = None, random_rounding: Optional[bool] = None, sigma_method: Optional[str] = None, sigma_kwargs: Optional[Dict[str, Any]] = None, sigma_per_token: Optional[torch.Tensor] = None, repaint_num_iters: Optional[int] = None, repaint_jump: Optional[int] = None):
        if n_steps is not None:
            self.n_steps = n_steps
            print(f"Updated n_steps to {n_steps}")
        if device is not None:
            self.device = device
            print(f"Updated device to {device}")
        if remasking_prob is not None:
            self.remasking_prob = remasking_prob
            print(f"Updated remasking_prob to {remasking_prob}")
        if guidance_scale is not None:
            self.guidance_scale = guidance_scale
            print(f"Updated guidance_scale to {guidance_scale}")
        if batch_size is not None:
            self.batch_size = batch_size
            print(f"Updated batch_size to {batch_size}")
        if random_rounding is not None:
            self.random_rounding = random_rounding
            print(f"Updated random_rounding to {random_rounding}")
        if sigma_method is not None:
            self.sigma_method = sigma_method
            print(f"Updated sigma_method to {sigma_method}")
        if sigma_kwargs is not None:
            self.sigma_kwargs = sigma_kwargs
            print(f"Updated sigma_kwargs to {sigma_kwargs}")
        if sigma_per_token is not None:
            self.sigma_per_token = sigma_per_token
            print(f"Updated sigma_per_token to {sigma_per_token}")
        if repaint_num_iters is not None:
            self.repaint_num_iters = repaint_num_iters
            print(f"Updated repaint_num_iters to {repaint_num_iters}")
        if repaint_jump is not None:
            self.repaint_jump = repaint_jump
            print(f"Updated repaint_jump to {repaint_jump}")


    def impute_data(self, counts=np.array, valid_mask=np.array, impute_mask=np.array, labels=Union[Iterable[np.array],np.array, None]) -> np.array:
        if isinstance(counts, np.ndarray):
            counts = torch.from_numpy(counts).to(self.device).float()
        if isinstance(impute_mask, np.ndarray):
            impute_mask = torch.from_numpy(impute_mask).to(self.device).bool()
        if isinstance(valid_mask, np.ndarray):
            valid_mask = torch.from_numpy(valid_mask).to(self.device).bool()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).to(self.device)
        elif isinstance(labels, list):
            if isinstance(labels[0], np.ndarray):
                labels = [torch.from_numpy(cat_labels).to(self.device) for cat_labels in labels]
                
        samples = self.generator.impute_data(
            counts=counts, 
            impute_mask=impute_mask, 
            n_steps=self.n_steps, 
            device=self.device, 
            remasking_prob=self.remasking_prob, 
            guidance_scale=self.guidance_scale, 
            labels=labels, 
            valid_mask=valid_mask, 
            batch_size=self.batch_size,
            random_rounding=self.random_rounding,
            sigma_method=self.sigma_method,
            sigma_kwargs=self.sigma_kwargs,
            sigma_per_token=self.sigma_per_token,
            repaint_num_iters=self.repaint_num_iters,
            repaint_jump=self.repaint_jump,
            **self.kwargs
        )
        
        return samples
        