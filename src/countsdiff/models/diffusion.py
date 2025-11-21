"""
Blackout diffusion model implementation
"""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel, UNet1DModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from typing import Iterable, Optional, Tuple, Union


class Countsdiff(nn.Module):
    
    def __init__(
        self,
        model_type: str = "2d",  # "1d" for SNP, "2d" for images, "attention1d" for attention-based, i.e. rnaseq
        **kwargs
    ):
        """
        Initialize model
        
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
            raise ValueError("1D SNP model is not currently supported, please use attention1d model")
        elif model_type == "2d":
            # Use diffusers UNet2DModel for image data
            unet_signature = inspect.signature(UNet2DModel.__init__)
            if kwargs["num_classes"] is not None and kwargs["num_classes"] > 0:
                if "block_out_channels" not in kwargs or len(kwargs["block_out_channels"]) == 0:
                    kwargs["block_out_channels"] = [224, 448, 672, 896] # Default channels for UNet2DModel
                embed_dim = kwargs['block_out_channels'][0] * 4  # Use last block's out channels
                self.class_embedder = nn.Embedding(kwargs["num_classes"], embed_dim)
                self.num_classes = kwargs["num_classes"]
            # Multi-label conditioning (e.g., CelebA attributes)
            self.multi_label = bool(kwargs.get("multi_label", False))
            if self.multi_label:
                if "block_out_channels" not in kwargs or len(kwargs["block_out_channels"]) == 0:
                    kwargs["block_out_channels"] = [224, 448, 672, 896]
                embed_dim = kwargs['block_out_channels'][0] * 4
                self.attr_dim = int(kwargs.get("num_attributes", kwargs.get("label_dim", 40)))
                self.attr_mlp = nn.Sequential(
                    nn.Linear(self.attr_dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim)
                )
            valid_args = {k: v for k, v in kwargs.items() if k in unet_signature.parameters}
            self.model = UNet2DModel(
                **valid_args
            )
        elif model_type == "attention1d":
            attention_signature = inspect.signature(AttentionWrapper.__init__)
            valid_args = {k: v for k, v in kwargs.items() if k in attention_signature.parameters}
            self.model = AttentionWrapper(
                **valid_args
            )
            self.num_classes = kwargs.get("num_cell_types", None)
            self.num_batches = kwargs.get("num_batches", None)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        self.pred_target = kwargs.get("pred_target", "rate")
        print(f"Using prediction target: {self.pred_target}")
        if self.pred_target not in ["x0", "rate"]:
            raise ValueError(f"Unsupported pred_target: {self.pred_target}")
        
        
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: Union[torch.tensor, Iterable[torch.tensor]] = None,  uncond_mask = None, valid_mask = None, xt = None, return_val = "rate") -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, channels, seq_len] or [batch_size, channels, H, W]
            t: Timestep tensor [batch_size],
            class_labels: Optional class labels for conditioning
            uncond_mask: Optional mask to zero out class embeddings for unconditional guidance
            valid_mask: Optional mask for valid positions
            
        Returns:
            Model output [batch_size, out_dim, seq_len] or [batch_size, out_dim, H, W]
        """
        # Diffusers models expect timesteps to be scaled to [0, 1000]
        timesteps = (t * 1000).long()
        
        if self.model_type == "attention1d":
            pred = self.model(x, timesteps, class_labels=class_labels, valid_mask=valid_mask, uncond_mask=uncond_mask)
        else:
            class_embeddings = None
            if class_labels is not None:
                if getattr(self, "multi_label", False):
                    class_embeddings = self.attr_mlp(class_labels.float())
                elif hasattr(self, "class_embedder"):
                    class_embeddings = self.class_embedder(class_labels)
            if class_embeddings is not None and uncond_mask is not None:
                if torch.is_tensor(uncond_mask) and uncond_mask.dim() == 1 and uncond_mask.shape[0] == class_embeddings.shape[0]:
                    class_embeddings[uncond_mask] = 0.
                else:
                    class_embeddings = class_embeddings * (~uncond_mask).float()
            pred = self.model(x, timesteps, class_labels=class_embeddings).sample
        
        if return_val == "rate":
            if self.pred_target == "x0":
                assert xt is not None, "xt must be provided when pred_target is 'x0'"
                # Convert x0 prediction to rate prediction
                rate = F.softplus(pred - xt)
            else:
                rate = F.softplus(pred)
            return rate
        elif return_val == "x0":
            if self.pred_target == "rate":
                assert xt is not None, "xt must be provided when pred_target is 'rate'"
                # Convert rate prediction to x0 prediction
                return F.softplus(xt + pred)
            else:
                return F.softplus(pred)
        else:
            raise ValueError(f"Unsupported return_val: {return_val}")






class AttentionWrapper(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_genes: int, num_layers, embed_dim: int, num_heads: int, dropout: float, all_num_classes: Iterable[int]):
        super(AttentionWrapper, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=False, dim_feedforward=embed_dim * 4) # errors are raised incorrectly if batch_first=True
            for _ in range(num_layers)
        ])
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)
        self.gene_embeddings = nn.Embedding(num_genes, embed_dim)
        self.num_genes = num_genes
        self.embed_dim = embed_dim
        
        self.embedders = []
        for i, size in enumerate(all_num_classes):
            if size is not None and size > 0:
                embedder = nn.Embedding(size, embed_dim)
                setattr(self, f"label_embedder_{i}", embedder)
                self.embedders.append((f"label_embedder_{i}", embedder))
        self.all_num_classes = all_num_classes
        self.num_labels = len(self.embedders)
        
        #Following same timestep projection method as default diffusers
        self.time_proj = Timesteps(num_channels = embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(embed_dim, embed_dim)
        self.register_buffer("gene_idx", torch.arange(num_genes))

    def forward(self, x, timesteps, class_labels = Union[Iterable[torch.Tensor], None], valid_mask=None, uncond_mask=None):
        b, l = x.shape
        assert l == self.num_genes, f"Input shape {x.shape} does not match expected number of genes {self.num_genes}"
        
        # Process embeddings
        embeddings_list = []
        
        # Time embedding (always present)
        time_emb = self.time_embedding(self.time_proj(timesteps)).unsqueeze(1) # (batch_size, 1, embed_dim)
        embeddings_list.append(time_emb)
        
        
        for i, (name, embedder) in enumerate(self.embedders):
            label = class_labels[i] if i < len(class_labels) else None
            if label is not None:
                label_embs = embedder(label).unsqueeze(1)
                if uncond_mask is not None:
                    label_embs[uncond_mask] = 0.
                embeddings_list.append(label_embs)

        
        # Gene data embeddings
        data_emb = self.input_proj(x.unsqueeze(-1))  # (b, l, embed_dim) when input_dim=1
        gene_embeddings = self.gene_embeddings(self.gene_idx).unsqueeze(0) # (1, num_genes, embed_dim)
        gene_data = gene_embeddings + data_emb # (batch_size, num_genes, embed_dim)
        
        # Concatenate all embeddings
        x = torch.cat(embeddings_list + [gene_data], dim=1)
        
        # Create attention mask if needed
        if valid_mask is not None:
            # Pad valid_mask for the conditioning tokens
            n_conditioning_tokens = len(embeddings_list)
            conditioning_mask = torch.ones(b, n_conditioning_tokens, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([conditioning_mask, valid_mask], dim=1)
            # Transformer expects True for positions to IGNORE
            attention_mask = ~full_mask
        else:
            attention_mask = None
        
        for layer in self.layers:
            x = layer(x.transpose(0, 1), src_key_padding_mask=attention_mask).transpose(0, 1)
        # Return only gene predictions (skip conditioning tokens)
        n_conditioning_tokens = len(embeddings_list)
        return self.output_proj(x[:, n_conditioning_tokens:])