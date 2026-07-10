"""
Model naming convention utilities for consistent checkpoint naming across all phases.

Naming format: ModelName_DatasetName_EpochCount_Compression_PrunningRatio_Quantization_Phase.pt

Examples:
- Phase 1 (baseline): mobilenetv2_OpenSourceData01_5ep_baseline.pt
- Phase 2 (optimized): mobilenetv2_OpenSourceData01_5ep_0.3comp_0.2prune_8q.pt
"""

from pathlib import Path
from typing import Optional, Tuple


class ModelNaming:
    """Utilities for consistent model naming across phases."""
    
    @staticmethod
    def generate_baseline_name(
        model_name: str,
        dataset_name: str,
        epochs: int
    ) -> str:
        """
        Generate baseline model name (Phase 1).
        
        Format: ModelName_DatasetName_EpochCount_baseline.pt
        
        Args:
            model_name: Model architecture (e.g., "mobilenetv2")
            dataset_name: Dataset name (e.g., "OpenSourceData01")
            epochs: Number of training epochs
        
        Returns:
            Checkpoint filename string
        """
        # Clean dataset name (remove path, special chars)
        dataset_clean = Path(dataset_name).name.replace(" ", "_").lower()
        model_clean = model_name.lower().replace(" ", "_")
        
        return f"{model_clean}_{dataset_clean}_{epochs}ep_baseline.pt"
    
    @staticmethod
    def generate_optimized_name(
        model_name: str,
        dataset_name: str,
        epochs: int,
        compression: float = 0.0,
        pruning: float = 0.0,
        quantization_bits: int = 32
    ) -> str:
        """
        Generate optimized model name (Phase 2).
        
        Format: ModelName_DatasetName_EpochCount_Compression_Pruning_Quantization.pt
        
        Args:
            model_name: Model architecture
            dataset_name: Dataset name
            epochs: Number of training epochs
            compression: Compression ratio [0, 1]
            pruning: Pruning ratio [0, 1]
            quantization_bits: Quantization bits (4, 8, 16, 32)
        
        Returns:
            Checkpoint filename string
        """
        # Clean names
        dataset_clean = Path(dataset_name).name.replace(" ", "_").lower()
        model_clean = model_name.lower().replace(" ", "_")
        
        # Format optimization parameters (remove trailing zeros)
        comp_str = f"{compression:.1f}".rstrip('0').rstrip('.')
        prune_str = f"{pruning:.1f}".rstrip('0').rstrip('.')
        
        return f"{model_clean}_{dataset_clean}_{epochs}ep_{comp_str}c_{prune_str}p_{quantization_bits}q.pt"
    
    @staticmethod
    def parse_model_name(filename: str) -> dict:
        """
        Parse model name to extract training parameters.
        
        Handles models with underscores in name (e.g., efficientnet_b0, shufflenet_v2).
        Format: {model}_{dataset}_{epochs}ep[_{compression}c_{pruning}p_{quantization}q][_baseline|_optimized].pt
        
        Args:
            filename: Model checkpoint filename
        
        Returns:
            Dictionary with extracted parameters
        """
        name = filename.replace(".pt", "")
        parts = name.split("_")
        
        result = {
            'filename': filename,
            'model_name': "unknown",
            'dataset_name': "unknown",
            'epochs': None,
            'compression': 0.0,
            'pruning': 0.0,
            'quantization_bits': 32,
            'is_baseline': False,
            'is_optimized': False
        }
        
        # Find the epoch marker (format: "{digits}ep")
        epoch_idx = None
        for i, part in enumerate(parts):
            if 'ep' in part and part[0].isdigit():
                try:
                    result['epochs'] = int(part.replace('ep', ''))
                    epoch_idx = i
                    break
                except (ValueError, IndexError):
                    pass
        
        if epoch_idx is None:
            # Fallback if no epoch found
            return result
        
        # Dataset is the part just before epochs
        if epoch_idx > 0:
            result['dataset_name'] = parts[epoch_idx - 1]
        
        # Model name is everything before the dataset
        if epoch_idx > 1:
            result['model_name'] = "_".join(parts[:epoch_idx - 1])
        elif epoch_idx == 1:
            result['model_name'] = parts[0]
        
        # Extract optimization parameters (after epoch index)
        for i in range(epoch_idx + 1, len(parts)):
            part = parts[i]
            
            # Check if baseline
            if part == 'baseline':
                result['is_baseline'] = True
            elif part == 'optimized':
                result['is_optimized'] = True
            
            # Extract compression
            if 'c' in part and part[0].isdigit():
                try:
                    result['compression'] = float(part.replace('c', ''))
                except ValueError:
                    pass
            
            # Extract pruning
            if 'p' in part and part[0].isdigit():
                try:
                    result['pruning'] = float(part.replace('p', ''))
                except ValueError:
                    pass
            
            # Extract quantization
            if 'q' in part and part[0].isdigit():
                try:
                    result['quantization_bits'] = int(part.replace('q', ''))
                except ValueError:
                    pass
        
        # Update is_optimized based on compression/pruning/quantization
        if (result['compression'] > 0 or 
            result['pruning'] > 0 or 
            result['quantization_bits'] < 32):
            result['is_optimized'] = True
        
        return result
    
    @staticmethod
    def get_friendly_name(filename: str) -> str:
        """
        Get human-friendly display name for model.
        
        Args:
            filename: Model checkpoint filename
        
        Returns:
            Friendly display name
        """
        info = ModelNaming.parse_model_name(filename)
        
        if info['is_baseline']:
            return f"📦 Baseline: {info['model_name'].upper()} ({info['dataset_name']}) - {info['epochs']} epochs"
        elif info['is_optimized']:
            comp_pct = int(info['compression'] * 100) if info['compression'] > 0 else 0
            prune_pct = int(info['pruning'] * 100) if info['pruning'] > 0 else 0
            quant_str = f"{info['quantization_bits']}-bit" if info['quantization_bits'] < 32 else ""
            
            optimizations = []
            if comp_pct > 0:
                optimizations.append(f"{comp_pct}% compression")
            if prune_pct > 0:
                optimizations.append(f"{prune_pct}% pruning")
            if quant_str:
                optimizations.append(quant_str)
            
            opt_str = ", ".join(optimizations) if optimizations else "minimal"
            return f"⚡ Optimized: {info['model_name'].upper()} ({info['dataset_name']}) - {opt_str}"
        else:
            return f"📄 {info['model_name'].upper()} ({info['dataset_name']}) - {info['epochs']} epochs"
    
    @staticmethod
    def get_model_info_display(filename: str) -> str:
        """
        Get detailed information string for model.
        
        Args:
            filename: Model checkpoint filename
        
        Returns:
            Formatted info string
        """
        info = ModelNaming.parse_model_name(filename)
        
        lines = [
            f"**Model:** {info['model_name'].upper()}",
            f"**Dataset:** {info['dataset_name']}",
            f"**Epochs:** {info['epochs']}",
        ]
        
        if info['is_baseline']:
            lines.append(f"**Status:** ✅ Baseline (unoptimized)")
        elif info['is_optimized']:
            lines.append(f"**Status:** ⚡ Optimized")
            if info['compression'] > 0:
                lines.append(f"**Compression:** {info['compression']*100:.0f}%")
            if info['pruning'] > 0:
                lines.append(f"**Pruning:** {info['pruning']*100:.0f}%")
            if info['quantization_bits'] < 32:
                lines.append(f"**Quantization:** {info['quantization_bits']}-bit")
        
        return "\n".join(lines)
