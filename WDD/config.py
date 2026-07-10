"""
Configuration file for Welding Defect Detection.

Centralizes all configurable parameters for training, optimization, and UI.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Literal

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "weld_defects"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ==================== TRAINING CONFIG ====================
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model
    model_name: str = "mobilenetv2"
    num_classes: int = 2  # Normal vs Defect
    input_size: int = 224
    
    # Dataset
    train_size: int = 560  # 70% of 800
    val_size: int = 120    # 15% of 800
    test_size: int = 120   # 15% of 800
    
    # Training hyperparameters
    default_epochs: int = 10
    default_batch_size: int = 16
    default_learning_rate: float = 0.001
    
    # Ranges for UI sliders
    min_epochs: int = 1
    max_epochs: int = 20
    batch_sizes: list = None
    learning_rates: list = None
    
    # Device (CPU only)
    device: Literal["cpu"] = "cpu"
    
    # Checkpointing
    save_interval: int = 2  # Save every N epochs
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32, 64]
        if self.learning_rates is None:
            self.learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]


# ==================== DATA GENERATION CONFIG ====================
@dataclass
class DataGenerationConfig:
    """Configuration for synthetic weld defect data generation."""
    
    # Dataset composition
    num_defect_images: int = 400
    num_normal_images: int = 400
    total_images: int = 800
    
    # Image properties
    image_size: int = 224
    image_format: str = "png"
    
    # Defect types and probabilities
    defect_types: list = None
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        if self.defect_types is None:
            self.defect_types = [
                "crack",
                "porosity",
                "undercut",
                "spatter",
                "incomplete_fusion"
            ]


# ==================== OPTIMIZATION CONFIG ====================
@dataclass
class OptimizationConfig:
    """Configuration for model optimization (compression, pruning, quantization)."""
    
    # Slider ranges
    compression_range: tuple = (0.0, 1.0)  # 0% to 100% compression
    pruning_range: tuple = (0.0, 0.95)     # 0% to 95% pruning
    quantization_bits: list = None
    
    # Optimization impacts (scaling factors)
    compression_accuracy_loss: float = 0.05  # Per 10% compression
    pruning_accuracy_loss: float = 0.02      # Per 10% pruning
    quantization_accuracy_loss: float = 0.01 # Per bit reduction
    
    compression_size_reduction: float = 0.8  # 80% size reduction per unit
    pruning_size_reduction: float = 0.7
    quantization_size_reduction: float = 0.5
    
    def __post_init__(self):
        if self.quantization_bits is None:
            self.quantization_bits = [32, 16, 8, 4]


# ==================== INSTANTIATE CONFIGS ====================
training_config = TrainingConfig()
data_generation_config = DataGenerationConfig()
optimization_config = OptimizationConfig()
