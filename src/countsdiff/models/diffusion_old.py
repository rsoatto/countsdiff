"""
Blackout diffusion model implementation
"""

import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet2DModel, UNet1DModel
from typing import Optional, Tuple


class Countsdiff(nn.Module):
    """Blackout diffusion model for SNP generation"""
    
    def __init__(
        self,
        dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 2, 2),
        channels: int = 7,  # noised data + position + ancestry
        out_dim: int = 1,
        model_type: str = "1d",  # "1d" for SNP, "2d" for images
        **kwargs
    ):
        """
        Initialize blackout diffusion model
        
        Args:
            dim: Base dimension
            dim_mults: Dimension multipliers for U-Net
            channels: Number of input channels
            out_dim: Output dimension
            model_type: Type of model ("1d" for SNP, "2d" for images)
            **kwargs: Additional arguments for Unet/Unet1D
        """
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == "1d":
            # Use diffusers UNet1DModel for SNP data
            self.model = UNet1DModel(
                sample_size=128,  # sequence length
                in_channels=channels,
                out_channels=out_dim,
                layers_per_block=2,
                block_out_channels=[dim * mult for mult in dim_mults],
                down_block_types=["DownBlock1D"] * len(dim_mults),
                up_block_types=["UpBlock1D"] * len(dim_mults),
                **kwargs
            )
        elif model_type == "2d":
            # Use diffusers UNet2DModel for image data (CIFAR-10)
            # Configure attention blocks - use AttnDownBlock2D/AttnUpBlock2D for attention layers
            num_levels = len(dim_mults)
            attn_resolutions = kwargs.get('attn_resolutions', [16])  # Default attention at 16x16
            
            # Create block types - add attention where specified
            down_block_types = []
            up_block_types = []
            
            for i, mult in enumerate(dim_mults):
                current_resolution = 32 // (2 ** i)  # Resolution at this level
                if current_resolution in attn_resolutions:
                    down_block_types.append("AttnDownBlock2D")
                    up_block_types.insert(0, "AttnUpBlock2D")  # Reverse order for up blocks
                else:
                    down_block_types.append("DownBlock2D")
                    up_block_types.insert(0, "UpBlock2D")
            
            self.model = UNet2DModel(
                sample_size=32,  # CIFAR-10 image size
                in_channels=channels,
                out_channels=out_dim,
                layers_per_block=kwargs.get('num_res_blocks', 2),
                block_out_channels=[dim * mult for mult in dim_mults],
                down_block_types=tuple(down_block_types),
                up_block_types=tuple(up_block_types),
                attention_head_dim=kwargs.get('attention_head_dim', 8),
                dropout=kwargs.get('dropout', 0.0),
                norm_num_groups=32,
                resnet_time_scale_shift=kwargs.get('resnet_time_scale_shift', 'default'),
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        self.channels = channels
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, channels, seq_len] or [batch_size, channels, H, W]
            t: Timestep tensor [batch_size]
            
        Returns:
            Model output [batch_size, out_dim, seq_len] or [batch_size, out_dim, H, W]
        """
        # Diffusers models expect timesteps to be scaled to [0, 1000]
        timesteps = (t * 1000).long()
        return self.model(x, timesteps).sample
    
    def predict_noise(
        self, 
        noised_data: torch.Tensor, 
        timesteps: torch.Tensor,
        pos_encoding: Optional[torch.Tensor] = None,
        ancestry: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise given noised data and conditioning
        
        Args:
            noised_data: Noised SNP data or image data
            timesteps: Diffusion timesteps
            pos_encoding: Position encoding (for SNP data)
            ancestry: Ancestry conditioning (for SNP data)
            
        Returns:
            Predicted noise
        """
        if self.model_type == "1d":
            # For SNP data, concatenate conditioning information
            if pos_encoding is None or ancestry is None:
                raise ValueError("pos_encoding and ancestry required for 1D model")
            x = torch.cat([noised_data, pos_encoding, ancestry], dim=1)
        else:
            # For image data, use noised data directly
            x = noised_data
        
        # Forward pass
        return self.forward(x, timesteps)
    
    @staticmethod
    def cos_p_scheduler(t: torch.Tensor) -> torch.Tensor:
        """Cosine probability scheduler"""
        if len(t.shape) == 1:
            # For 1D (SNP) case
            t = t.reshape(-1, 1, 1)
        else:
            # For 2D (image) case
            t = t.reshape(-1, 1, 1, 1)
        return torch.cos(t * np.pi / 2) ** 2
    
    def add_noise(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        p_scheduler=None
    ) -> torch.Tensor:
        """
        Add binomial noise to data (blackout process)
        
        Args:
            x: Clean data
            t: Timesteps
            p_scheduler: Probability scheduler function
            
        Returns:
            Noised data
        """
        if p_scheduler is None:
            p_scheduler = self.cos_p_scheduler
        
        # Calculate survival probability
        p_t = p_scheduler(t).expand(*x.shape)
        
        # Apply binomial noise
        noised = torch.binomial(count=x, prob=p_t)
        
        return noised
    
    def compute_loss(
        self,
        x: torch.Tensor,
        pos_encoding: Optional[torch.Tensor] = None, 
        ancestry: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        weight_scheduler=None
    ) -> torch.Tensor:
        """
        Compute training loss
        
        Args:
            x: Clean data
            pos_encoding: Position encoding (for SNP data)
            ancestry: Ancestry conditioning (for SNP data)
            valid_mask: Valid position mask (for SNP data)
            reduction: Loss reduction method
            weight_scheduler: Weight scheduler function for time weighting
            
        Returns:
            Loss tensor
        """
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=x.device)
        
        # Add noise
        noised = self.add_noise(x, t)
        
        if self.model_type == "1d":
            # SNP data loss computation
            if pos_encoding is None or ancestry is None or valid_mask is None:
                raise ValueError("pos_encoding, ancestry, and valid_mask required for 1D model")
            
            seq_len = x.shape[1]
            
            # Normalize noised data
            noised_normalized = (noised - noised.mean(dim=1, keepdim=True)) / (noised.std(dim=1, keepdim=True) + 1e-8)
            
            # Prepare ancestry conditioning
            ancestry_expanded = ancestry.unsqueeze(1).repeat(1, seq_len, 1)
            
            # Create model input
            model_input = torch.cat([
                noised_normalized.unsqueeze(2),
                pos_encoding,
                ancestry_expanded
            ], dim=2).permute(0, 2, 1)
            
            # Predict birth rate using softplus
            predicted_rate = torch.softplus(self.forward(model_input, t).squeeze(1))
            
            # Target is the actual birth rate (difference between original and noised)
            target_rate = x - noised
            
            # Apply time weighting if provided
            if weight_scheduler is not None:
                weights = weight_scheduler(t).view(-1, 1)
            else:
                weights = 1.0
            
            # Compute loss using negative log-likelihood of Poisson process (only on valid positions)
            loss = weights * (predicted_rate - target_rate * torch.log(predicted_rate + 1e-8)) * valid_mask
            
            if reduction == 'mean':
                return loss.sum() / valid_mask.sum()
            elif reduction == 'sum':
                return loss.sum()
            else:
                return loss
        
        else:
            # Image data loss computation (CIFAR-10 style)
            # Normalize noised data
            width = 1.0
            mean_v = 255.0/2 * self.cos_p_scheduler(t)
            normalized_noised = (noised - mean_v) / width
            
            # Predict birth rate (expected noise)
            predicted_rate = torch.softplus(self.forward(normalized_noised, t))
            
            # Target is the actual birth rate (difference between original and noised)
            target_rate = x - noised
            
            # Apply time weighting if provided
            if weight_scheduler is not None:
                weights = weight_scheduler(t).view(-1, 1, 1, 1)
            else:
                weights = 1.0
            
            # Compute loss using negative log-likelihood of Poisson process
            loss = weights * (predicted_rate - target_rate * torch.log(predicted_rate + 1e-8))
            
            if reduction == 'mean':
                return loss.mean()
            elif reduction == 'sum':
                return loss.sum()
            else:
                return loss
