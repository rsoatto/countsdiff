"""
Main trainer class for SNP diffusion models
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Iterable, Optional, Tuple
import neptune
import datetime as dt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from neptune.utils import stringify_unsupported
import pandas as pd
import h5py
import anndata

from countsdiff.models.diffusion import Countsdiff
from countsdiff.models.ema import ExponentialMovingAverage
from countsdiff.data.datasets import create_cifar10_loaders, create_celeba_loaders
from countsdiff.training.utils import build_jump_linear_beta_schedule, legacy_blackout_config_to_buffers
from countsdiff.utils.metrics import scFID

# Neptune handler API token
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmJiYTkzOC00YTI1LTRhZmYtYmI0NS0zYjEzOGJlYjE1ZDkifQ=="

class CountsdiffTrainer:
    """Trainer for count diffusion models"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
        run_id: Optional[str] = None, #If resuming a specific run instead of starting fresh
        legacy_model: bool = False,
        eval_mode: bool = False,
    ):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            device: Device to use for training
        """
        if legacy_model:
            from ..models.diffusion_old import Countsdiff as CountsdiffOld
            self.Countsdiff = CountsdiffOld
        else:
            self.Countsdiff = Countsdiff
        self.config = config
        
        self.device = config.get('training', {}).get('device', device)
        self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')


         # Extract configuration
        self.training_config = config.get('training', {})
        self.model_config = config.get('model', {})
        self.data_config = config.get('data', {})
        self.scheduler_config = config.get('scheduler', {})
        self.generation_config = config.get('generation', {
            'num_samples': 5000,
            'n_steps': 1000,
            'device': 'cuda',
            'remasking_prob': 0.0,
            'sigma_method': None,
            'guidance_scale': 1.0,
        })
        # Ensure generation config is present for logging
        self.config['generation'] = self.generation_config

        self.p_uncond = self.model_config.get('p_uncond', None); self.model_config['p_uncond'] = self.p_uncond
        self.conditional_training = self.p_uncond is not None; self.model_config['conditional_training'] = self.conditional_training
        self.num_classes = self.model_config.get('num_classes', None) if self.conditional_training else None; self.model_config['num_classes'] = self.num_classes
        self.pred_target = self.model_config.get('pred_target', 'rate'); self.model_config['pred_target'] = self.pred_target

        # Normalize and persist scheduler name
        scheduler_name = (self.scheduler_config.get('scheduler', 'cosine')).lower()
        self.scheduler_config['scheduler'] = scheduler_name
        self.p_scheduler = getattr(self, f"{scheduler_name}_p_scheduler", None)
        # Persist resolved defaults to config as they are parsed
        self.continuous_training = self.scheduler_config.get('continuous', True)
        self.scheduler_config['continuous'] = self.continuous_training
        self.weight_scheduler = getattr(self, f"{scheduler_name}_weight_scheduler_{'continuous' if self.continuous_training else 'discrete'}", None)
        if self.p_scheduler is None:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        if self.weight_scheduler is None:
            raise ValueError(f"Unknown scheduler combination: {scheduler_name} and continuous is {self.continuous_training}")
        self.T = self.scheduler_config.get('T', 1000)
        self.scheduler_config['T'] = self.T
        self.poisson_randomization = self.scheduler_config.get('poisson_randomization', False)
        self.scheduler_config['poisson_randomization'] = self.poisson_randomization
        if not self.poisson_randomization:
            self.scheduler_config['lbd'] = 1.0
        else:
            self.scheduler_config['lbd'] = self.scheduler_config.get('lbd', None)

    
        self.batch_size = self.training_config.get('batch_size', 4096); self.training_config['batch_size'] = self.batch_size
        self.n_steps = self.training_config.get('n_steps', 5000); self.training_config['n_steps'] = self.n_steps
        self.lr = self.training_config.get('lr', 2e-5); self.training_config['lr'] = self.lr

        self.snapshot_freq = self.training_config.get('snapshot_freq', 10000); self.training_config['snapshot_freq'] = self.snapshot_freq
        self.snapshot_freq_preemption = self.training_config.get('snapshot_freq_preemption', 10000); self.training_config['snapshot_freq_preemption'] = self.snapshot_freq_preemption
        assert self.snapshot_freq_preemption <= self.snapshot_freq, "snapshot_freq_preemption should be less than or equal to snapshot_freq"
        assert self.snapshot_freq % self.snapshot_freq_preemption == 0, "snapshot_freq should be multiple of snapshot_freq_preemption"
        self.sum_lambda = self.training_config.get('sum_lambda', 1.0); self.training_config['sum_lambda'] = self.sum_lambda
        # Commonly used training defaults to ensure they appear in logged config
        if 'checkpoint_dir' not in self.training_config:
            self.training_config['checkpoint_dir'] = 'checkpoints'
        if 'random_seed' not in self.training_config:
            self.training_config['random_seed'] = 42
        if 'debug' not in self.training_config:
            self.training_config['debug'] = False

        # Data parameters
        self.dataset_type = self.data_config.get('dataset', 'snp'); self.data_config['dataset'] = self.dataset_type
        self.data_path = self.data_config.get('data_path', None); self.data_config['data_path'] = self.data_path
        if self.data_path is None:
            print("No data_path specified in data configuration")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.ema = None
        self.train_loader = None
        self.val_loader = None
        # Legacy blackout schedule buffers (CIFAR)
        self._inst_obs_times = None
        self._inst_sampling_prob = None
        self._inst_weights = None
        
        # Training state
        self.state = {
            'step': 0,
            'lossHistory': [],
            'blackoutLossHistory': [],
            'sumLossHistory': [],
            'evalLossHistory': [],
            'blackoutEvalLossHistory': [],
            'sumEvalLossHistory': []
        }

        
        print(f"Trainer initialized on device {self.device}")
        
        config = {
            'model': self.model_config,
            'data': self.data_config,
            'scheduler': self.scheduler_config,
            'training': self.training_config,
            'generation': self.generation_config,
        }
        
                # Initialize Neptune run if available
        if 'NEPTUNE_API_TOKEN' in os.environ:
            if run_id:
                # Resume from specific run
                self.run = neptune.init_run(
                    with_id=run_id,
                    api_token=NEPTUNE_API_TOKEN, #service account token for loading runs
                    project="countsdiff-iclr/ICLR",
                    source_files=["src/**.py"],
                    capture_hardware_metrics=False,
                    mode="read-only" if eval_mode else "async",
                )
                unique_run_name = self.run["config/run_name"].fetch()
                
                config["run_name"] = unique_run_name
                config = stringify_unsupported(config)
                if not eval_mode:
                    # Update config to include default values
                    del self.run["config"]
                    self.run["config"] = config
                print(f"Resuming Neptune run {run_id} with name {unique_run_name}")
            else:
                unique_run_name = f"countsdiff_{self.dataset_type}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.run = neptune.init_run(
                    name=unique_run_name,
                    api_token=NEPTUNE_API_TOKEN,
                    project="countsdiff-iclr/ICLR",
                    source_files=["src/**.py"],
                    capture_hardware_metrics=False,
                    mode = "debug" if self.training_config.get('debug', True) else "async",  # Use debug mode if specified
                    tags=config["training"].get("tags", None)
                )
                config["run_name"] = unique_run_name
                self.run["config"] = stringify_unsupported(config)
        else:
            print("Neptune API token not found in environment. Proceeding without experiment tracking.")
            unique_run_name = f"local_{self.dataset_type}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run = neptune.init_run(
                name=unique_run_name,
                project="countsdiff-iclr/ICLR",
                mode="offline"
            )
            config["run_name"] = unique_run_name
            self.run["config"] = config
                
        
        checkpoint_base_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = os.path.join(checkpoint_base_dir, unique_run_name)
        self.random_seed = self.training_config.get('random_seed', 42)
        
        self.setup_data()
        self.setup_model()
        if not eval_mode:
            self.setup_metrics()
       
    def cosine_p_scheduler(self, t: torch.Tensor) -> torch.Tensor:
        """Cosine probability scheduler"""
        return torch.cos(t * torch.pi / 2) ** 2

    def cosine_weight_scheduler_continuous(self, t: torch.Tensor) -> torch.Tensor:
        """Weights derived from cosine p_scheduler"""
        return torch.pi/2 * torch.sin(t * torch.pi)
    
    def jump_p_scheduler(self, t: torch.Tensor):
        """P scheduler borrowed from ddpm"""
        if self._p_schedule is None:
            raise ValueError("Jump p_schedule not initialized")
        num_steps = self._p_schedule.shape[0] - 1
        t_index = (t * num_steps).long()
        return self._p_schedule[t_index]

    def jump_weight_scheduler_discrete(self, t: torch.Tensor) -> torch.Tensor:
        """Jump weight scheduler"""
        return torch.ones_like(t)

    def blackout_p_scheduler(self, t: torch.Tensor, t_T: torch.Tensor = torch.Tensor([15.])) -> torch.Tensor:
        """Legacy blackout scheduler"""
        t_T = t_T.to(t.device)
        return torch.nn.functional.sigmoid((2*t - 1) * torch.log(torch.exp(-t_T)/(1 - torch.exp(-t_T))))

    def blackout_weight_scheduler_discrete(self, t: torch.Tensor) -> torch.Tensor:
        """Blackout weight scheduler"""
        if self._inst_weights is None:
            raise ValueError("Legacy blackout weights not initialized")
        num_steps = self._inst_weights.shape[0]
        t_index = (t * num_steps).long()
        return self._inst_weights[t_index]

    def blackout_weight_scheduler_continuous(self, t: torch.Tensor,  t_T: torch.Tensor = torch.Tensor([15.])) -> torch.Tensor:
        """Difference becomes derivative as we go into continuous time"""
        p_t = self.blackout_p_scheduler(t, t_T)
        t_T = t_T.to(p_t.device)
        return -2 * p_t * (1 - p_t) * torch.log(torch.exp(-t_T)/(1 - torch.exp(-t_T)))

    def setup_data(self) -> None:
        """Setup data loaders"""
        if self.dataset_type == 'cifar10':
            # Setup CIFAR-10 data loaders
            self.train_loader, self.val_loader = create_cifar10_loaders(
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                centered=self.data_config.get('centered', True),
                random_flip=self.data_config.get('random_flip', True),
                return_labels=self.conditional_training
            )
            print(f"Loaded CIFAR-10 dataset")
            # Build legacy blackout schedule buffers if requested
            if self.scheduler_config.get('scheduler', '') in ['blackout']:
                self._inst_obs_times, self._inst_sampling_prob, self._inst_weights = \
                    legacy_blackout_config_to_buffers(self.config, self.device)
            if self.scheduler_config.get('scheduler', '') in ['jump']:
                # For jump scheduler, no legacy buffers needed
                _, self._p_schedule, lmd = build_jump_linear_beta_schedule(
                    T=self.scheduler_config.get('T', 1000),
                    beta_start=self.scheduler_config.get('beta_start', 1e-3),
                    logsnr_start=self.scheduler_config.get('logsnr_start', 10.0),
                    logsnr_end=self.scheduler_config.get('logsnr_end', -12.0),
                    signal_stat=self.scheduler_config.get('signal_stat', 1.0),
                    lbd=self.scheduler_config.get('lbd', None),
                    device=self.device
                )
                self.scheduler_config['lbd'] = lmd
                if self.poisson_randomization:
                    print("Jump scheduler initialized with lbd =", lmd)
                else:
                    print("Binomial JUMP scheduler initialized (no poisson randomization)")
        elif self.dataset_type == 'celeba':
            # Setup CelebA data loaders (same image pipeline)
            self.train_loader, self.val_loader = create_celeba_loaders(
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
                image_size=self.data_config.get('image_size', 64),
                centered=self.data_config.get('centered', False),
                random_flip=self.data_config.get('random_flip', True),
                return_labels=self.conditional_training
            )
            print(f"Loaded CelebA dataset")
            if self.scheduler_config.get('scheduler', '') in ['blackout']:
                self._inst_obs_times, self._inst_sampling_prob, self._inst_weights = \
                    legacy_blackout_config_to_buffers(self.config, self.device)
        elif self.dataset_type == 'scrna':
            # Setup scRNA-seq data loaders
            self.setup_scrna_data()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        print(f"Created data loaders: {len(self.train_loader)} train batches, "
              f"{len(self.val_loader)} val batches")

    def setup_model(self) -> None:
        """Setup model, optimizer, and EMA"""
        if self.dataset_type in ('cifar10', 'celeba'):
            # CIFAR-10 model setup
            self.model = self.Countsdiff(
                **self.model_config,
            ).to(self.device)
            print(f"{self.dataset_type} model setup complete")
            
        elif self.dataset_type == 'scrna':
            # scRNA-seq model setup
            model_config = self.model_config.copy()
            model_config.update({
                'num_genes': self.num_genes,
                'all_num_classes': self.all_num_classes,
                'model_type': 'attention1d'
            })
            
            self.model = self.Countsdiff(**model_config).to(self.device)
            print(f"scRNA-seq attention model setup complete")
            
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        # Setup optimizer and EMA
        ema_rate = self.training_config.get('ema_rate', 0.9999)
        
        optimizer_params = {
            'lr': self.lr,
            'betas': (self.training_config.get('beta1', 0.9), 0.999),
            'eps': self.training_config.get('eps', 1e-8),
            'weight_decay': self.training_config.get('weight_decay', 0)
        }
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_rate)
        
        # Learning rate warmup setup
        self.warmup_steps = self.training_config.get('warmup', 0)
        if self.warmup_steps > 0:
            print(f"Using learning rate warmup for {self.warmup_steps} steps")

    def setup_metrics(self) -> None:
        """Setup metrics for validation"""
        if self.dataset_type in ('cifar10', 'celeba'):
            print("Initializing FID and IS metrics for image dataset")
            self.fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False).to(self.device).eval()
            self.is_score = InceptionScore(normalize=False).to(self.device).eval()
        elif self.dataset_type == 'scrna':
            print("Initializing scFID")
            gene_names = self.val_dataset.gene_names
            counts = self.val_dataset.counts.cpu().numpy()
            missingness_mask = self.val_dataset.missingness_mask.cpu().numpy()
            
            condition_values = {}
            for condition in self.val_dataset.condition_keys:
                condition_values[condition] = getattr(self.val_dataset, f"{condition}_values")
                
            # 2. Prepare observation (obs) and variable (var) metadata
            # .obs stores metadata for each cell (row)
            self.obs_df = pd.DataFrame({
                **condition_values
            })
            for column_name in self.obs_df.columns:
                if column_name not in self.data_config.get('condition_keys', []):
                    self.obs_df = self.obs_df.drop(columns=[column_name])

            # .var stores metadata for each gene (column)
            var_df = pd.DataFrame(index=gene_names)
            # 3. Create the AnnData object
            adata2 = anndata.AnnData(
                X=counts,
                obs=self.obs_df,
                var=var_df,
                layers={'missingness_mask': missingness_mask}
            )
            adata_covariates = {
                key: adata2.obs[key].unique().tolist() for key in adata2.obs.keys()
            }
            
            model_filename = "data/dnadiff/2024-02-12-scvi-homo-sapiens/scvi.model"
            if not os.path.exists(model_filename):
                raise FileNotFoundError(f"Missing SCVI Model. Please download and place in {model_filename}")
            self.scfid = scFID(gene_names = gene_names, categorical_covariates= adata_covariates, feature_model_path=model_filename)
            
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self._fid_real_initialized = False
        print("Metrics setup complete")

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_state_dict': self.ema.state_dict() if hasattr(self.ema, 'state_dict') else None,
            'config': self.config,
            **self.state
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint"""
        if os.path.exists(path):
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint.get('ema_state_dict') and hasattr(self.ema, 'load_state_dict'):
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
            
            # Load training state
            for key in self.state.keys():
                if key in checkpoint:
                    self.state[key] = checkpoint[key]

    


    def normalize_for_model(self, data, p_t):
        n = data.shape[0]
        if self.dataset_type in ('cifar10', 'celeba') :
            width = 255.0
            mean_v = (255.0/2 * p_t).reshape((n, 1, 1, 1))
            normalized = ((data - mean_v) / width).to(torch.float32)
        elif self.dataset_type == 'scrna':
            normalized = torch.log1p(data)
        else:
            print(f"Normalization for {self.dataset_type} not implemented")
            normalized = data
        return normalized


    def corrupt_data(self, input_counts: torch.Tensor, t: torch.Tensor, current_t: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if current_t is None:
            current_t = torch.zeros_like(t)
        assert torch.all(t >= current_t), "t must be greater than or equal to current_t"
        p_scheduler = self.p_scheduler
        counts_shape = input_counts.shape
        p_t = p_scheduler(t).reshape(-1, *[1 for _ in range(len(counts_shape) - 1)])
        p_current = p_scheduler(current_t).reshape(-1, *[1 for _ in range(len(counts_shape) - 1)])
        corrupted_counts = torch.binomial(input_counts, p_t/p_current)
        return corrupted_counts, p_t
    
    def corrupt_data_old(self, original_counts: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_scheduler = self.p_scheduler
        counts_shape = original_counts.shape
        p_t = p_scheduler(t).reshape(-1, *[1 for _ in range(len(counts_shape) - 1)])
        corrupted_counts = torch.binomial(original_counts, p_t)
        return corrupted_counts, p_t

    def generate_cifar10_batch_data(self, batch):
        fixed_timepoints = (self.continuous_training == False)
        if self.conditional_training:
            images, labels = batch
            labels = labels.to(self.device).long()
        else:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch
                labels = None

        with torch.no_grad():

            n, nc, nx, ny = images.shape  # [batch, channels, height, width]
    
            # Scale to [0, 255] range
            if images.max() <= 1.0:
                img_batch_gpu = (255 * images).to(self.device).float()
            else:
                img_batch_gpu = images.to(self.device).float()

            images_original = img_batch_gpu.clone()          
            if self.poisson_randomization:
                assert self.scheduler_config.get('lbd', None) is not None, "lbd must be set for poisson randomization"
                # JUMP process of taking Poisson (lambda alpha_t x_0) is equivalent to first taking Poisson(x_0) then applying Binomial(alpha_t) by thinning
                img_batch_gpu = torch.poisson(img_batch_gpu * self.scheduler_config.get('lbd'))
            

            # Sample timesteps and reshape for broadcasting
            if fixed_timepoints:
                k = torch.randint(low=1, high=self.T + 1, size=(n,), device=self.device)
                timesteps = k.float() / self.T 
            else:
                timesteps = torch.from_numpy(np.random.uniform(0, 1, size=(n))).to(self.device).float()
            nt, p_t = self.corrupt_data(img_batch_gpu, timesteps)
            birth_rate_batch = img_batch_gpu - nt

            # Normalize
            model_input = self.normalize_for_model(nt, p_t)

            return model_input, birth_rate_batch, nt, labels, timesteps, images_original

    
    def train_step_cifar10(self, batch: torch.Tensor) -> float:
        """Execute one CIFAR-10 training step - matching reference exactly"""
        # Generate noised batch data
        self.model.train()
        model_input, birth_rate, xt, labels, timesteps, images_original = self.generate_cifar10_batch_data(batch)
        # Zero gradients
        self.optimizer.zero_grad()
        
        if self.conditional_training:
            uncond_mask = (torch.rand(model_input.shape[0], device=self.device) < self.p_uncond).bool() # 1 for unconditional
        else:
            uncond_mask = None
        
        if self.poisson_randomization:
            assert self.scheduler_config.get('lbd', None) is not None, "lbd must be set for poisson randomization"
            assert self.pred_target == 'x0', "With poisson randomization, only x0 prediction is supported"
            x0_pred = self.model(model_input, timesteps, class_labels=labels, uncond_mask=uncond_mask, xt=xt, return_val="x0")
            weights = self.weight_scheduler(timesteps).view(-1, 1, 1, 1).detach()
            eps = self.training_config.get('eps', 1e-8)
            loss = torch.mean(weights * (x0_pred - images_original * torch.log(x0_pred + eps)))
            
        else:
            predicted_rate = self.model(model_input, timesteps, class_labels=labels, uncond_mask=uncond_mask, xt=xt)
            with torch.no_grad():
                weights = self.weight_scheduler(timesteps).view(-1, 1, 1, 1).detach()
            eps = self.training_config.get('eps', 1e-8)
            loss = torch.mean(weights * (predicted_rate - birth_rate * torch.log(predicted_rate + eps)))

            # Backward pass
        loss.backward()
            

        self.ema.update(self.model.parameters())
        
        # Gradient clipping if specified  
        gradient_clip = self.training_config.get('gradient_clip', None)
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate_cifar10(self) -> float:
        """Run CIFAR-10 validation - matching reference exactly"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            # Use EMA parameters like reference
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            
            for batch in self.val_loader:
                # Generate noised batch data
                model_input, birth_rate, xt, labels, timesteps, images_original = self.generate_cifar10_batch_data(batch)
   
                if self.poisson_randomization:
                    assert self.scheduler_config.get('lbd', None) is not None, "lbd must be set for poisson randomization"
                    assert self.pred_target == 'x0', "With poisson randomization, only x0 prediction is supported"
                    x0_pred = self.model(model_input, timesteps, class_labels=labels, xt=xt, return_val="x0")
                    weights = self.weight_scheduler(timesteps).view(-1, 1, 1, 1).detach()
                    eps = self.training_config.get('eps', 1e-8)
                    loss = torch.mean(weights * (x0_pred - images_original * torch.log(x0_pred + eps)))
                    val_losses.append(loss.item())
                else:
                    predicted_rate = self.model(model_input, timesteps, class_labels=labels, xt=xt)
                    
                    weights = self.weight_scheduler(timesteps).view(-1, 1, 1, 1).detach()
                    eps = self.training_config.get('eps', 1e-8)
                    loss = torch.mean(weights * (predicted_rate - birth_rate * torch.log(predicted_rate + eps)))
                    val_losses.append(loss.item())
            
            # Restore original parameters
            self.ema.restore(self.model.parameters())
        
        self.model.train()
        return np.mean(val_losses)
    
    def _to_uint8_255(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure images are uint8 in [0,255] without needless round-trips."""
        if x.dtype == torch.uint8:
            return x
        # if already [0,1], scale; otherwise clamp to [0,255]
        if x.max() <= 1.0:
            x = (x * 255.0).clamp(0, 255)
        else:
            x = x.clamp(0, 255)
        return x.to(torch.uint8)
    
    def fid_is_calculate_and_log(
        self,
        generator,
        num_samples: int = 5000,
        n_steps: int = 1000,
        device: str = 'cuda',
        remasking_prob: float = 0.0,
        guidance_scale: float = 0.0,
        log_to_neptune: bool = True,
        batch_size: int = 500,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[float, float, float]:
        """Generates samples and computes FID/IS. Returns (FID, IS_mean, IS_std)."""

        print(f"Generating {num_samples} samples")
        with torch.inference_mode():
            gen_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=num_samples,
            shuffle=True
        )
            gen_batch = next(iter(gen_loader))
            gen = generator.generate_samples(
                num_samples=num_samples,
                labels=gen_batch[1],
                n_steps=n_steps,
                device=device,
                remasking_prob=remasking_prob,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                **kwargs
            )
            gen = torch.from_numpy(gen) if not isinstance(gen, torch.Tensor) else gen
            gen_u8 = self._to_uint8_255(gen)

        # ensure metrics are on GPU
        self.fid = self.fid.to(self.device).eval()
        self.is_score = self.is_score.to(self.device).eval()

        # One-time: feed real images to FID (persists because reset_real_features=False)
        if not self._fid_real_initialized:
            print("Feeding real images to FID (one-time init)")
            with torch.inference_mode():
                for batch in self.val_loader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    else:
                        x = batch
                    x_u8 = self._to_uint8_255(x)
                    self.fid.update(x_u8.to(self.device, non_blocking=True), real=True)
            self._fid_real_initialized = True

        # DataLoader for generated samples (CPU→GPU pinned)
        gen_loader = DataLoader(
            TensorDataset(gen_u8.cpu()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

        with torch.inference_mode():
            # update with generated only
            for (g_u8,) in gen_loader:
                g_u8 = g_u8.to(self.device, non_blocking=True)
                self.fid.update(g_u8, real=False)
                self.is_score.update(g_u8)

            fid_value = float(self.fid.compute().item())
            is_mean, is_std = self.is_score.compute()
            is_mean, is_std = float(is_mean), float(is_std)

        self.fid.reset()
        self.is_score.reset()

        if log_to_neptune and getattr(self, "run", None) is not None:
            print(f"Logging FID: {fid_value}, IS(mean): {is_mean}, IS(std): {is_std} to Neptune")
            self.run["val/fid"].append(value=fid_value, step=self.state['step'])
            self.run["val/is_mean"].append(value=is_mean, step=self.state['step'])
            self.run["val/is_std"].append(value=is_std, step=self.state['step'])
        else:
            print(f"FID: {fid_value}, IS(mean): {is_mean}, IS(std): {is_std}")

        return fid_value, is_mean, is_std
    
    def scfid_calculate_and_log(
        self,
        generator,
        num_samples: int = 5000,
        n_steps: int = 50,
        device: str = 'cuda',
        remasking_prob: float = 0.0,
        guidance_scale: float = 1.0,
        log_to_neptune: bool = True,
        batch_size: int = 500,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[float, float, float]:
        """Generates samples and computes FID/IS. Returns (FID, IS_mean, IS_std)."""
        
        print(f"Generating {num_samples} samples")
        with torch.inference_mode():
            gen_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=num_samples,
            shuffle=True
        )
            gen_batch = next(iter(gen_loader))
            valid_mask=(~gen_batch[2]).to(self.device).bool()
            gen = generator.generate_samples(
                num_samples=num_samples,
                labels=gen_batch[1],
                n_steps=n_steps,
                device=device,
                valid_mask=valid_mask,
                remasking_prob=remasking_prob,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                **kwargs
            )
            gen = torch.from_numpy(gen) if not isinstance(gen, torch.Tensor) else gen
                # ensure metrics are on GPU
        self.scfid = self.scfid.to(self.device).eval()
        full_dataset = self.train_loader.dataset
        train_data = next(iter(DataLoader(
            self.train_loader.dataset,
            batch_size=min(num_samples, len(self.train_loader.dataset)),
            shuffle=False,
        )))
        train_counts=train_data[0]
        train_labels = train_data[1] if self.conditional_training else None
        covariate_df = full_dataset.build_covariate_df(train_labels)
        self.scfid.reset()
        self.scfid.update(train_counts.cpu().numpy(), covariate_df, True)
        val_labels = [cat_label.to(self.device) for cat_label in gen_batch[1]]
        val_covariate_df = full_dataset.build_covariate_df(val_labels)
        self.scfid.update(gen.cpu().numpy(), val_covariate_df, False)

        fid_score = float(self.scfid.compute().item())

        self.scfid.reset()

        if log_to_neptune and getattr(self, "run", None) is not None:
            print(f"Logging FID: {fid_score} to Neptune")
            self.run["val/fid"].append(value=fid_score, step=self.state['step'])
        else:
            print(f"FID: {fid_score}")

        return fid_score

        
    
    def setup_scrna_data(self) -> None:
        """Setup single-cell RNA-seq data loaders"""
        from ..data.process_scrna import SingleCellDataset

        self.train_dataset = SingleCellDataset(self.data_path, split = "train", condition_keys=self.data_config.get('condition_keys', []))
        self.val_dataset = SingleCellDataset(self.data_path, split = "val", condition_keys = self.data_config.get('condition_keys', []))
        for key in self.train_dataset.condition_keys:
            print(f"Categorical key '{key}' has {self.train_dataset.get_num_classes(key)} classes")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Store dataset info for model setup
        self.num_genes = self.train_dataset.counts.shape[1]
        self.all_num_classes = [self.train_dataset.get_num_classes(key) for key in self.train_dataset.condition_keys]
        
        print(f"Created data loaders: {len(self.train_loader)} train batches, "
              f"{len(self.val_loader)} val batches")

    def generate_scrna_batch_data(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, Iterable[torch.Tensor], torch.Tensor]:
        """Generate noised batch data for scRNA-seq"""
        with torch.no_grad():
            counts, labels, missing_mask = batch
            valid_mask = ~missing_mask
            # Move to device
            counts = counts.to(self.device).float()
            valid_mask = valid_mask.to(self.device).bool()
            labels = [lbl.to(self.device).long() for lbl in labels]
            
            
            # Sample timesteps
            batch_size = counts.shape[0]
            t = torch.rand(batch_size, device=self.device)
            
            noised_counts, p_t = self.corrupt_data(counts, t)
            
            
            # Calculate birth rate (difference)
            birth_rate = counts - noised_counts
        
            model_input = self.normalize_for_model(noised_counts, p_t)
            
            return (
                model_input,  # model input
                birth_rate,  # target birth rate
                noised_counts,
                labels,
                t,
                valid_mask
            )

    def train_step_scrna(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """Execute one scRNA-seq training step"""
        # Generate noised batch data
        self.model.train()

        normalized_noised, birth_rate, xt, labels, timesteps, valid_mask = \
            self.generate_scrna_batch_data(batch)
         
        if self.conditional_training:
            uncond_mask = (torch.rand(normalized_noised.shape[0], device=self.device) < self.p_uncond).bool() # 1 for unconditional
        else:
            uncond_mask = None
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        predicted_rate = self.model(
            normalized_noised,
            timesteps,
            class_labels=labels,
            valid_mask=valid_mask,
            uncond_mask=uncond_mask,
            xt=xt
        ).squeeze(-1)

        # Apply time weighting
        weights = self.weight_scheduler(timesteps).view(-1, 1)
        
        # Compute loss (similar to CIFAR-10 but adapted for scRNA-seq)
        # Only compute loss on valid genes
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        per_gene = (predicted_rate - birth_rate * torch.log(predicted_rate + epsilon)) * valid_mask.float()
        valid_counts = valid_mask.float().sum(dim=1).clamp_min(1.0)  # (B,)
        per_sample = per_gene.sum(dim=1) / valid_counts              # (B,)
        loss = torch.mean(weights.squeeze(1) * per_sample)

        # Backward pass
        loss.backward()
        
        # Update EMA
        self.ema.update(self.model.parameters())
        
        # Gradient clipping if specified
        gradient_clip = self.training_config.get('gradient_clip', None)
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
        
        self.optimizer.step()
        
        return loss.item()

    def validate_scrna(self) -> float:
        """Run scRNA-seq validation"""
        self.model.eval()
        val_losses = []
        print("Running scRNA-seq validation...")
        
        with torch.no_grad():
            # Use EMA parameters
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            
            for batch in self.val_loader:
                # Generate noised batch data
                normalized_noised, birth_rate, xt, labels, timesteps, valid_mask = \
                    self.generate_scrna_batch_data(batch)
                # Forward pass
                predicted_rate = self.model(
                    normalized_noised,
                    timesteps,
                    class_labels=labels,
                    valid_mask=valid_mask,
                    xt=xt
                ).squeeze(-1)
                
                # Apply time weighting
                weights = self.weight_scheduler(timesteps).view(-1, 1)
                
                # Compute loss (similar to CIFAR-10 but adapted for scRNA-seq)
                # Only compute loss on valid genes
                # Avoid log(0) by adding small epsilon
                epsilon = 1e-8
                per_gene = (predicted_rate - birth_rate * torch.log(predicted_rate + epsilon)) * valid_mask.float()
                valid_counts = valid_mask.float().sum(dim=1).clamp_min(1.0)  # (B,)
                per_sample = per_gene.sum(dim=1) / valid_counts              # (B,)
                loss = torch.mean(weights.squeeze(1) * per_sample)
                val_losses.append(loss.item())
                if self.training_config.get('debug', False):
                    break # Only one batch in debug mode
            
            # Restore original parameters
            self.ema.restore(self.model.parameters())
        
        self.model.train()
        return np.mean(val_losses)

    
    def update_learning_rate(self, step: int) -> None:
        """Update learning rate with warmup"""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup
            warmup_lr = self.lr * (step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # Use base learning rate after warmup
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
    
    def train(self) -> None:
        """Main training loop"""
        
        seed = self.random_seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        from ..generation import CountsdiffGenerator
        # Try to load existing checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.dataset_type}_latest.pth')

        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            
        generator = CountsdiffGenerator(input_trainer=self)
        
        # Set up iterators
        train_iter = iter(self.train_loader)
        
        # Training loop
        steps_per_epoch = len(self.train_loader)
        estimated_epochs = self.n_steps / steps_per_epoch
        print(f"Training for {self.n_steps} steps (approximately {estimated_epochs:.2f} epochs)")
        
        self.model.train()
        for step in tqdm(range(self.state['step'], self.n_steps)):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Update learning rate
            self.update_learning_rate(step)
            
            # Training step - different for each dataset type
            if self.dataset_type in ('cifar10', 'celeba'):
                loss = self.train_step_cifar10(batch)
                self.state['lossHistory'].append(loss)
                self.run["train/loss"].append(value=loss, step=step)
                # Evaluate periodically
                log_freq = self.training_config.get('log_freq', 100)
                eval_freq = self.training_config.get('eval_freq', 5000)
                if step % log_freq == 0:
                    val_loss = self.validate_cifar10()
                    self.state['evalLossHistory'].append(val_loss)
                    self.run["val/loss"].append(value=val_loss, step=step)
                    current_epoch_estimate = step / steps_per_epoch
                    log_message = f"Step {step}/{self.n_steps}, Loss: {loss:.6f}, Val Loss: {val_loss:.6f}"
                    if step % eval_freq == 0 and step >= self.training_config.get('start_eval', 10000):
                        fid_score, is_mean, is_std = self.fid_is_calculate_and_log(generator, **self.generation_config)
                        log_message += f"\n FID: {fid_score}, IS(mean): {is_mean}, IS(std): {is_std}"
                    print(log_message)
                    
            elif self.dataset_type == 'scrna':
                loss = self.train_step_scrna(batch)
                self.state['lossHistory'].append(loss)
                self.run["train/loss"].append(value=loss, step=step)
                
                # Evaluate periodically
                log_freq = self.training_config.get('log_freq', 100)
                if step % log_freq == 0:
                    val_loss = self.validate_scrna()
                    self.state['evalLossHistory'].append(val_loss)
                    self.run["val/loss"].append(value=val_loss, step=step)
                    current_epoch_estimate = step / steps_per_epoch
                    print(f"Step {step}/{self.n_steps} (≈ epoch {current_epoch_estimate:.2f}/{estimated_epochs:.2f}), "
                          f"Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")
                eval_freq = self.training_config.get('eval_freq', 5000)
                if step % eval_freq == 0 and step >= self.training_config.get('start_eval', 10000):
                    fid_score = self.scfid_calculate_and_log(generator, **self.generation_config)
                    print(f"Step {step}, scFID: {fid_score}")
                          
            else:
                raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
            # Save checkpoint
            if step % self.snapshot_freq_preemption == 0:
                self.save_checkpoint(checkpoint_path)
                
                if step % self.snapshot_freq == 0:
                    numbered_path = os.path.join(self.checkpoint_dir, f'checkpoint_{step//self.snapshot_freq}.pth')
                    self.save_checkpoint(numbered_path)

            if step % 1000 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            # Update step counter
            self.state['step'] = step + 1
        
        # Final checkpoint save
        print("Training completed!")
        self.save_checkpoint(checkpoint_path)
        step = self.state['step']
        print(f"Step {step}/{self.n_steps} (≈ epoch {current_epoch_estimate:.2f}/{estimated_epochs:.2f}), "
                f"Train loss: {loss:.4f}"
                f"Val loss: {val_loss:.4f}")
        
        # Save checkpoint
        if step % self.snapshot_freq == 0 and step > 0:
            self.state['step'] = step
            self.save_checkpoint(os.path.join(self.checkpoint_dir, f'{step}.pth'))
            self.save_checkpoint(checkpoint_path)  # Override latest
        
        # Final save
        self.state['step'] = self.n_steps
        self.save_checkpoint(os.path.join(self.checkpoint_dir, f'final.pth'))
        
        self.run.stop()
        
        print(f"Training complete after {self.n_steps} steps!")
    
