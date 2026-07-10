"""
Model trainer with start/stop/resume capabilities.

Handles training loop, checkpointing, and pause/resume functionality.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime
from tqdm import tqdm
import traceback

from trainer.model_builder import create_model, compute_model_size_mb, count_model_parameters
from config import training_config, CHECKPOINTS_DIR


class TrainingState:
    """Manages training state and progress."""
    
    def __init__(self):
        self.current_epoch = 0
        self.total_epochs = 0
        self.is_training = False
        self.is_paused = False
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.timestamp = None
        self.checkpoint_path = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'is_training': self.is_training,
            'is_paused': self.is_paused,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'timestamp': self.timestamp,
            'checkpoint_path': self.checkpoint_path,
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'TrainingState':
        """Create from dictionary."""
        state = TrainingState()
        state.current_epoch = data.get('current_epoch', 0)
        state.total_epochs = data.get('total_epochs', 0)
        state.is_training = data.get('is_training', False)
        state.is_paused = data.get('is_paused', False)
        state.train_losses = data.get('train_losses', [])
        state.val_losses = data.get('val_losses', [])
        state.train_accuracies = data.get('train_accuracies', [])
        state.val_accuracies = data.get('val_accuracies', [])
        state.best_val_accuracy = data.get('best_val_accuracy', 0.0)
        state.timestamp = data.get('timestamp')
        state.checkpoint_path = data.get('checkpoint_path')
        return state


class WeldDefectTrainer:
    """Trainer for weld defect detection model."""
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = 'cpu',
        checkpoint_dir: Path = CHECKPOINTS_DIR
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model (or None to create default MobileNetV2)
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Model
        if model is None:
            self.model = create_model(num_classes=2, device=device)
        else:
            self.model = model.to(device)
        
        # Training state
        self.state = TrainingState()
        self.state.timestamp = datetime.now().isoformat()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(train_loader)
        print(f"    Processing {num_batches} batches of training data...")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress indicator
            if (batch_idx + 1) % max(1, num_batches // 5) == 0:
                current_accuracy = correct / total
                avg_loss_so_far = total_loss / (batch_idx + 1)
                progress_pct = ((batch_idx + 1) / num_batches) * 100
                print(f"      Batch {batch_idx + 1}/{num_batches} ({progress_pct:.0f}%) - "
                      f"Loss: {avg_loss_so_far:.4f}, Accuracy: {current_accuracy:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = len(val_loader)
        print(f"    Processing {num_batches} batches of validation data...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Progress indicator
                if (batch_idx + 1) % max(1, num_batches // 3) == 0:
                    current_accuracy = correct / total
                    avg_loss_so_far = total_loss / (batch_idx + 1)
                    progress_pct = ((batch_idx + 1) / num_batches) * 100
                    print(f"      Batch {batch_idx + 1}/{num_batches} ({progress_pct:.0f}%) - "
                          f"Loss: {avg_loss_so_far:.4f}, Accuracy: {current_accuracy:.4f}")
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 0.001,
        checkpoint_name: Optional[str] = None
    ) -> Dict:
        """
        Train model from scratch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            checkpoint_name: Name for saving checkpoint (or None to auto-generate)
        
        Returns:
            Dictionary with training results
        """
        # Setup
        self.state.total_epochs = num_epochs
        self.state.is_training = True
        self.state.train_losses = []
        self.state.val_losses = []
        self.state.train_accuracies = []
        self.state.val_accuracies = []
        
        print("\n" + "="*80)
        print("TRAINING SESSION STARTED")
        print("="*80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Training batch size: {train_loader.batch_size}")
        print(f"Validation batch size: {val_loader.batch_size}")
        print(f"Device: {self.device}")
        print("="*80 + "\n")
        
        # Calculate class weights to handle class imbalance
        # Count samples in each class
        class_counts = torch.zeros(2)
        for images, labels in train_loader:
            class_counts += torch.bincount(labels, minlength=2).float()
        
        # Weight each class inversely proportional to its frequency
        # Minority class gets higher weight
        total_samples = class_counts.sum()
        class_weights = total_samples / (2 * class_counts)
        class_weights = class_weights.to(self.device)
        
        print(f"Class distribution in training data:")
        print(f"  Class 0 (normal): {int(class_counts[0])} samples (weight: {class_weights[0].item():.3f})")
        print(f"  Class 1 (defect): {int(class_counts[1])} samples (weight: {class_weights[1].item():.3f})")
        print(f"Using weighted CrossEntropyLoss to balance classes\n")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        try:
            for epoch in range(num_epochs):
                self.state.current_epoch = epoch + 1
                
                print(f"\n[EPOCH {epoch + 1}/{num_epochs}]")
                print("-" * 80)
                
                # Train
                print(f"  Training phase...")
                train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
                self.state.train_losses.append(train_loss)
                self.state.train_accuracies.append(train_acc)
                print(f"  Training complete - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                
                # Validate
                print(f"  Validation phase...")
                val_loss, val_acc = self.validate(val_loader, criterion)
                self.state.val_losses.append(val_loss)
                self.state.val_accuracies.append(val_acc)
                print(f"  Validation complete - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # Track best accuracy
                if val_acc > self.state.best_val_accuracy:
                    self.state.best_val_accuracy = val_acc
                    print(f"  *** NEW BEST VALIDATION ACCURACY: {val_acc:.4f} ({val_acc*100:.2f}%) ***")
                
                print(f"  Epoch summary:")
                print(f"    Training:   Loss={train_loss:.4f}  Accuracy={train_acc:.4f} ({train_acc*100:.2f}%)")
                print(f"    Validation: Loss={val_loss:.4f}  Accuracy={val_acc:.4f} ({val_acc*100:.2f}%)")
                print(f"    Best so far: {self.state.best_val_accuracy:.4f} ({self.state.best_val_accuracy*100:.2f}%)")
            
            print("\n" + "="*80)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Final best validation accuracy: {self.state.best_val_accuracy:.4f} ({self.state.best_val_accuracy*100:.2f}%)")
            
            # Save final checkpoint
            if checkpoint_name is None:
                checkpoint_name = f"model_{self.state.timestamp.replace(':', '-')}.pt"
            
            print(f"Saving checkpoint: {checkpoint_name}")
            checkpoint_path = self.save_checkpoint(checkpoint_name)
            self.state.checkpoint_path = str(checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
            print("="*80 + "\n")
            
            self.state.is_training = False
            
            return {
                'success': True,
                'epoch': self.state.current_epoch,
                'best_val_accuracy': self.state.best_val_accuracy,
                'checkpoint_path': checkpoint_path,
                'train_losses': self.state.train_losses,
                'val_losses': self.state.val_losses,
                'train_accuracies': self.state.train_accuracies,
                'val_accuracies': self.state.val_accuracies,
            }
        
        except Exception as e:
            self.state.is_training = False
            print(f"Error during training: {str(e)}")
            print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'epoch': self.state.current_epoch,
            }
    
    def save_checkpoint(self, name: str) -> Path:
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint filename
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / name
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'training_state': self.state.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'model_size_mb': compute_model_size_mb(self.model),
            'num_params': count_model_parameters(self.model),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.state = TrainingState.from_dict(checkpoint['training_state'])
            
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
            return True
        
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False
    
    def get_checkpoints(self) -> list:
        """
        List available checkpoints.
        
        Returns:
            List of checkpoint filenames
        """
        checkpoints = sorted(self.checkpoint_dir.glob('*.pt'))
        return [cp.name for cp in checkpoints]
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model stats
        """
        return {
            'model_size_mb': compute_model_size_mb(self.model),
            'num_parameters': count_model_parameters(self.model),
            'model_name': type(self.model).__name__,
            'num_classes': 2,
        }
