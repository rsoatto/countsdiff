"""
Dataset classes for SNP diffusion training
"""

import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any, Optional


class SNPDataset(Dataset):
    """Dataset for hierarchical SNP data"""
    
    def __init__(self, samples: List[Tuple], max_seq_len: int = 128):
        """
        Initialize SNP dataset
        
        Args:
            samples: List of (vals, true_pos, dataset_pos, valid_mask, ancestry) tuples
            max_seq_len: Maximum sequence length
        """
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """Get a sample by index"""
        vals, true_pos, dataset_pos, valid_mask, ancestry = self.samples[idx]
        return vals, true_pos, dataset_pos, valid_mask, ancestry

    @classmethod
    def from_pickle(cls, pickle_path: str, max_seq_len: int = 128) -> 'SNPDataset':
        """Load dataset from pickle file"""
        with open(pickle_path, 'rb') as f:
            samples = pickle.load(f)
        return cls(samples, max_seq_len)


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function to handle batch preparation for SNP data
    
    Args:
        batch: List of (vals, true_pos, dataset_pos, valid_mask, ancestry) tuples
        
    Returns:
        Batched tensors: (vals, true_pos, dataset_pos, valid_mask, ancestry)
    """
    vals, true_pos, dataset_pos, valid_mask, ancestry = zip(*batch)
    
    vals = torch.from_numpy(np.stack(vals))
    true_pos = torch.from_numpy(np.stack(true_pos)).unsqueeze(-1)
    dataset_pos = torch.from_numpy(np.stack(dataset_pos)).unsqueeze(-1)
    valid_mask = torch.from_numpy(np.stack(valid_mask))
    ancestry = torch.from_numpy(np.stack(ancestry))
    
    return vals, true_pos, dataset_pos, valid_mask, ancestry


def create_data_loaders(
    data_path: str,
    batch_size: int = 4096,
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_seq_len: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        data_path: Path to pickled data file
        batch_size: Batch size
        train_split: Fraction of data to use for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset
    dataset = SNPDataset.from_pickle(data_path, max_seq_len)
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


class CIFAR10Dataset(Dataset):
    """Dataset wrapper for CIFAR-10 with normalization matching NCSN++"""
    
    # Pickle-safe transform helpers (avoid lambdas inside __init__)
    @staticmethod
    def _scale_to_255(x: torch.Tensor) -> torch.Tensor:
        return x * 255.0

    @staticmethod
    def _center_127_5(x: torch.Tensor) -> torch.Tensor:
        return x - 127.5
    
    def __init__(self, train: bool = True, transform=None, centered: bool = False, random_flip: bool = True, return_labels: bool = False):
        """
        Initialize CIFAR-10 dataset
        
        Args:
            train: Whether to load training or test set
            transform: Optional transform to apply
            centered: Whether to center data around 0 (NCSN++ uses False)
            random_flip: Whether to apply random horizontal flips (NCSN++ uses True)
            return_labels: Whether to return labels along with images
        """
        if transform is None:
            # Build transform list matching NCSN++ settings
            transform_list = []
            
            # Add random flip for training data if specified
            if train and random_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            
            # Convert to tensor (gives [0, 1] range)
            transform_list.append(transforms.ToTensor())
            
            # Scale to [0, 255] range (as in original blackout code)
            transform_list.append(transforms.Lambda(CIFAR10Dataset._scale_to_255))
            
            # Center around 0 if specified (NCSN++ uses centered=False)
            if centered:
                transform_list.append(transforms.Lambda(CIFAR10Dataset._center_127_5))
            
            transform = transforms.Compose(transform_list)
        
        self.return_labels = return_labels
        
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data/dnadiff', 
            train=train, 
            download=True, 
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.return_labels:
            return image, label
        return image


def create_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    centered: bool = False,
    random_flip: bool = True,
    return_labels: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test data loaders matching NCSN++ settings
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        centered: Whether to center data around 0 (NCSN++ uses False)
        random_flip: Whether to apply random horizontal flips (NCSN++ uses True)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets with NCSN++ settings
    train_dataset = CIFAR10Dataset(train=True, centered=centered, random_flip=random_flip, return_labels=return_labels)
    test_dataset = CIFAR10Dataset(train=False, centered=centered, random_flip=False, return_labels=return_labels)  # No flip for test
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


class CelebADataset(Dataset):
    """Dataset wrapper for CelebA with preprocessing similar to CIFAR path."""
    def __init__(self, split: str = 'train', transform=None, image_size: int = 64, centered: bool = False, random_flip: bool = True, return_labels: bool = True):
        if transform is None:
            tfs = []
            # Crop to face region then resize
            tfs.extend([
                transforms.CenterCrop(178),
                transforms.Resize(image_size),
            ])
            if split == 'train' and random_flip:
                tfs.append(transforms.RandomHorizontalFlip())
            tfs.append(transforms.ToTensor())
            tfs.append(transforms.Lambda(CIFAR10Dataset._scale_to_255))
            if centered:
                tfs.append(transforms.Lambda(CIFAR10Dataset._center_127_5))
            transform = transforms.Compose(tfs)

        self.return_labels = return_labels
        self.split = split
        self.dataset = torchvision.datasets.CelebA(
            root='./data/dnadiff',
            split=split,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, attr = self.dataset[idx]
        if self.return_labels:
            # Multi-label attributes; not used in current conditional pipeline
            return image, attr
        return image


def create_celeba_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 64,
    centered: bool = False,
    random_flip: bool = True,
    return_labels: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create CelebA train and validation loaders."""
    train_dataset = CelebADataset(split='train', image_size=image_size, centered=centered, random_flip=random_flip, return_labels=return_labels)
    val_dataset = CelebADataset(split='valid', image_size=image_size, centered=centered, random_flip=False, return_labels=return_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader
