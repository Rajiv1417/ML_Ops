"""
Dataset loader for weld defect images.

Handles loading, preprocessing, and batching of weld inspection images.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, List
import numpy as np


class WeldDefectDataset(Dataset):
    """PyTorch dataset for weld defect images."""
    
    def __init__(self, root_dir: Path, split: str = 'train', transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing train/val/test splits
            split: One of 'train', 'val', 'test'
            transform: Optional image transformations
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image paths and labels from disk."""
        # Handle both 'val' and 'valid' directory names
        if self.split == 'val':
            split_dir = self.root_dir / 'val'
            if not split_dir.exists():
                split_dir = self.root_dir / 'valid'
        else:
            split_dir = self.root_dir / self.split
        
        # Load normal images (label 0)
        normal_dir = split_dir / "normal"
        if normal_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in sorted(normal_dir.glob(ext)):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
        
        # Load defect images (label 1)
        defect_dir = split_dir / "defect"
        if defect_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in sorted(defect_dir.glob(ext)):
                    self.image_paths.append(img_path)
                    self.labels.append(1)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of sample
        
        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_data_transforms() -> dict:
    """
    Get data transformation pipelines.
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    transforms_dict = {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }
    return transforms_dict


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, val, and test sets.
    
    Args:
        dataset_dir: Root directory of dataset
        batch_size: Batch size for all loaders
        num_workers: Number of worker threads (0 for CPU)
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    transforms_dict = get_data_transforms()
    
    # Create datasets
    train_dataset = WeldDefectDataset(dataset_dir, split='train', transform=transforms_dict['train'])
    val_dataset = WeldDefectDataset(dataset_dir, split='val', transform=transforms_dict['val'])
    test_dataset = WeldDefectDataset(dataset_dir, split='test', transform=transforms_dict['val'])
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_stats(dataset_dir: Path) -> dict:
    """
    Get statistics about dataset.
    
    Args:
        dataset_dir: Root directory of dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'train': {'normal': 0, 'defect': 0},
        'val': {'normal': 0, 'defect': 0},
        'test': {'normal': 0, 'defect': 0},
    }
    
    dataset_dir = Path(dataset_dir)
    
    # Check train split
    train_dir = dataset_dir / 'train'
    if train_dir.exists():
        normal_dir = train_dir / 'normal'
        defect_dir = train_dir / 'defect'
        if normal_dir.exists():
            stats['train']['normal'] = len(list(normal_dir.glob('*.png')))
        if defect_dir.exists():
            stats['train']['defect'] = len(list(defect_dir.glob('*.png')))
    
    # Check validation split (try 'val' first, then 'valid')
    val_dir = dataset_dir / 'val'
    if not val_dir.exists():
        val_dir = dataset_dir / 'valid'
    
    if val_dir.exists():
        normal_dir = val_dir / 'normal'
        defect_dir = val_dir / 'defect'
        if normal_dir.exists():
            stats['val']['normal'] = len(list(normal_dir.glob('*.png')))
        if defect_dir.exists():
            stats['val']['defect'] = len(list(defect_dir.glob('*.png')))
    
    # Check test split
    test_dir = dataset_dir / 'test'
    if test_dir.exists():
        normal_dir = test_dir / 'normal'
        defect_dir = test_dir / 'defect'
        if normal_dir.exists():
            stats['test']['normal'] = len(list(normal_dir.glob('*.png')))
        if defect_dir.exists():
            stats['test']['defect'] = len(list(defect_dir.glob('*.png')))
    
    return stats
