"""
Model builders for weld defect classification.

Supports multiple architectures: MobileNetV2, EfficientNet-B0, ShuffleNet V2
Lightweight, CPU-friendly models optimized for edge deployment.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from torchvision import models


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture for image classification.
    
    Lightweight CNN optimized for mobile and edge devices.
    """
    
    def __init__(self, num_classes: int = 2, width_mult: float = 1.0):
        """
        Initialize MobileNetV2.
        
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier for model scaling (e.g., 0.5 for half-width)
        """
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        
        # Inverted residual block configuration
        # [t, c, n, s] where t=expansion, c=channels, n=num_blocks, s=stride
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # First layer: regular convolution
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult) if width_mult > 0.1 else 1280
        
        features = [
            self._conv_bn(3, input_channel, 2)
        ]
        
        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        
        # Last convolution block
        features.append(self._conv_bn(input_channel, last_channel, 1))
        
        self.features = nn.Sequential(*features)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _conv_bn(self, inp: int, oup: int, stride: int) -> nn.Sequential:
        """Create a conv-batch norm block."""
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_size_mb(self) -> float:
        """
        Calculate model size in MB.
        
        Returns:
            Model size in megabytes
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        # Convert bytes to MB
        return param_size / (1024 * 1024)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InvertedResidual(nn.Module):
    """Inverted residual block (bottleneck) used in MobileNetV2."""
    
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        """
        Initialize inverted residual block.
        
        Args:
            inp: Input channels
            oup: Output channels
            stride: Stride for depthwise convolution
            expand_ratio: Expansion factor for hidden dimension
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(self._conv_1x1_bn(inp, hidden_dim))
        
        layers.extend([
            # Depthwise convolution
            self._conv_dw(hidden_dim, hidden_dim, stride),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _conv_1x1_bn(self, inp: int, oup: int) -> nn.Sequential:
        """1x1 convolution with batch norm."""
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    
    def _conv_dw(self, inp: int, oup: int, stride: int) -> nn.Sequential:
        """Depthwise convolution with batch norm."""
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def create_model(
    num_classes: int = 2,
    model_name: str = "MobileNetV2",
    width_mult: float = 1.0,
    device: str = 'cpu'
) -> nn.Module:
    """
    Create and return a model for weld defect classification.
    
    Args:
        num_classes: Number of output classes (default: 2 for normal/defect)
        model_name: Model architecture - "MobileNetV2", "EfficientNet-B0", or "ShuffleNet V2"
        width_mult: Width multiplier for scaling (only used for MobileNetV2)
        device: Device to move model to ('cpu' or 'cuda')
    
    Returns:
        Model on specified device
    
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "MobileNetV2":
        model = MobileNetV2(num_classes=num_classes, width_mult=width_mult)
    
    elif model_name == "EfficientNet-B0":
        # Use torchvision's EfficientNet-B0 with pretrained weights
        model = models.efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
        # Replace final classification layer
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name == "ShuffleNet V2":
        # Use torchvision's ShuffleNet V2
        model = models.shufflenet_v2_x1_0(weights="ShuffleNet_V2_X1_0_Weights.DEFAULT")
        # Replace final classification layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: 'MobileNetV2', 'EfficientNet-B0', 'ShuffleNet V2'"
        )
    
    model = model.to(device)
    return model


def get_available_models() -> list:
    """Return list of available model names."""
    return ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model architecture.
    
    Args:
        model_name: Model name
    
    Returns:
        Dictionary with model metadata
    """
    info_map = {
        "MobileNetV2": {
            "size_mb": 5.5,
            "num_params": 3500000,
            "description": "Lightweight inverted residuals"
        },
        "EfficientNet-B0": {
            "size_mb": 5.3,
            "num_params": 4000000,
            "description": "Compound scaling design"
        },
        "ShuffleNet V2": {
            "size_mb": 2.3,
            "num_params": 2300000,
            "description": "Group convolutions for mobile"
        }
    }
    return info_map.get(model_name, {"size_mb": 0, "num_params": 0, "description": "Unknown"})


def compute_model_size_mb(model: nn.Module) -> float:
    """
    Compute the size of a PyTorch model in megabytes.
    
    Works with any model type (MobileNetV2, EfficientNet-B0, ShuffleNet V2, etc.).
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Convert bytes to MB
    return param_size / (1024 * 1024)


def count_model_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in a PyTorch model.
    
    Works with any model type (MobileNetV2, EfficientNet-B0, ShuffleNet V2, etc.).
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
