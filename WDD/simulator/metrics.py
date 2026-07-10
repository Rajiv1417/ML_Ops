"""
Metrics calculations for model optimization.

Calculates accuracy, latency, throughput, sparsity, and other metrics
based on optimization parameters.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    
    accuracy: float  # 0-1
    model_size_mb: float
    latency_ms: float  # Inference time per image
    throughput_fps: float  # Images per second
    sparsity: float  # 0-1 (fraction of zero weights)
    compression_ratio: float  # Original size / compressed size


class MetricsCalculator:
    """Calculates model metrics based on optimization parameters."""
    
    # Baseline metrics for MobileNetV2 on CPU
    BASELINE_ACCURACY = 0.92
    BASELINE_MODEL_SIZE_MB = 14.0
    BASELINE_LATENCY_MS = 45.0  # CPU inference time
    BASELINE_THROUGHPUT_FPS = 1000.0 / BASELINE_LATENCY_MS  # ~22 FPS
    
    def __init__(self):
        """Initialize calculator with baseline metrics."""
        self.baseline = ModelMetrics(
            accuracy=self.BASELINE_ACCURACY,
            model_size_mb=self.BASELINE_MODEL_SIZE_MB,
            latency_ms=self.BASELINE_LATENCY_MS,
            throughput_fps=self.BASELINE_THROUGHPUT_FPS,
            sparsity=0.0,
            compression_ratio=1.0
        )
    
    def calculate_metrics(
        self,
        compression: float,  # 0-1 (0% to 100%)
        pruning: float,      # 0-1 (0% to 100%)
        quantization_bits: int,  # 4, 8, 16, 32
        target_accuracy: float = None  # 0.5-1.0, if set, overrides calculated accuracy
    ) -> ModelMetrics:
        """
        Calculate optimized metrics based on compression, pruning, and quantization.
        
        Args:
            compression: Compression ratio (0-1)
            pruning: Fraction of weights to prune (0-1)
            quantization_bits: Bits per weight (4, 8, 16, 32)
            target_accuracy: Target accuracy to use (0.5-1.0). If provided, overrides calculated accuracy.
        
        Returns:
            ModelMetrics with optimized values
        """
        # ==================== ACCURACY IMPACT ====================
        if target_accuracy is not None:
            # Use target accuracy directly (student has direct control)
            accuracy = max(0.5, min(1.0, target_accuracy))
        else:
            # Calculate accuracy based on optimization techniques
            # Compression degrades accuracy over time (diminishing returns)
            compression_accuracy_loss = compression * compression * 0.25  # Quadratic decay
            
            # Pruning preserves accuracy initially, then degrades
            pruning_threshold = 0.3
            if pruning < pruning_threshold:
                pruning_accuracy_loss = (pruning / pruning_threshold) * 0.05
            else:
                pruning_accuracy_loss = 0.05 + ((pruning - pruning_threshold) / (1 - pruning_threshold)) * 0.15
            
            # Quantization reduces accuracy slightly
            quantization_bits_normalized = (32 - quantization_bits) / 32
            quantization_accuracy_loss = quantization_bits_normalized * 0.1
            
            # Combined accuracy impact
            total_accuracy_loss = compression_accuracy_loss + pruning_accuracy_loss + quantization_accuracy_loss
            accuracy = max(0.5, self.baseline.accuracy - total_accuracy_loss)  # Floor at 50%
        
        # ==================== MODEL SIZE ====================
        # Compression: ~0.8 reduction per unit
        compression_size_factor = (1 - compression * 0.8)
        
        # Pruning: ~0.7 reduction per unit (fewer weights = smaller)
        pruning_size_factor = (1 - pruning * 0.7)
        
        # Quantization: bits reduction (32->16 is ~50%, 32->8 is ~75%, etc.)
        quantization_factor = quantization_bits / 32.0
        
        # Combined size
        model_size_mb = self.baseline.model_size_mb * compression_size_factor * pruning_size_factor * quantization_factor
        model_size_mb = max(0.5, model_size_mb)  # Floor at 0.5 MB
        
        # ==================== LATENCY ====================
        # Compression reduces computation
        compression_latency_factor = 1 - compression * 0.5  # 50% speedup at max compression
        
        # Pruning reduces computation (sparse matrix ops are faster)
        pruning_latency_factor = 1 - pruning * 0.6  # 60% speedup at max pruning
        
        # Quantization speeds up computation significantly on CPU
        quantization_latency_factor = (quantization_bits / 32.0) ** 0.8  # Sublinear improvement
        
        # Combined latency
        latency_ms = (self.baseline.latency_ms * 
                     compression_latency_factor * 
                     pruning_latency_factor * 
                     quantization_latency_factor)
        latency_ms = max(5.0, latency_ms)  # Floor at 5ms
        
        # ==================== THROUGHPUT ====================
        # Inverse of latency
        throughput_fps = 1000.0 / latency_ms
        
        # ==================== SPARSITY ====================
        # Sparsity is the fraction of zero weights (from pruning)
        # More pruning = more sparse
        sparsity = pruning
        
        # ==================== COMPRESSION RATIO ====================
        compression_ratio = self.baseline.model_size_mb / model_size_mb if model_size_mb > 0 else 1.0
        
        return ModelMetrics(
            accuracy=accuracy,
            model_size_mb=model_size_mb,
            latency_ms=latency_ms,
            throughput_fps=throughput_fps,
            sparsity=sparsity,
            compression_ratio=compression_ratio
        )
    
    def get_accuracy_impact(
        self,
        compression: float,
        pruning: float,
        quantization_bits: int
    ) -> Tuple[float, str]:
        """
        Calculate accuracy loss and provide interpretation.
        
        Returns:
            Tuple of (accuracy_loss, interpretation)
        """
        metrics = self.calculate_metrics(compression, pruning, quantization_bits)
        accuracy_loss = self.baseline.accuracy - metrics.accuracy
        accuracy_loss_pct = accuracy_loss * 100
        
        if accuracy_loss_pct < 2:
            interpretation = "✅ Minimal accuracy loss"
        elif accuracy_loss_pct < 5:
            interpretation = "⚠️ Slight accuracy loss"
        elif accuracy_loss_pct < 10:
            interpretation = "⚠️ Moderate accuracy loss"
        else:
            interpretation = "❌ High accuracy loss"
        
        return accuracy_loss_pct, interpretation
    
    def get_speedup(
        self,
        compression: float,
        pruning: float,
        quantization_bits: int
    ) -> float:
        """Calculate inference speedup compared to baseline."""
        metrics = self.calculate_metrics(compression, pruning, quantization_bits)
        return self.baseline.latency_ms / metrics.latency_ms
    
    def get_size_reduction(
        self,
        compression: float,
        pruning: float,
        quantization_bits: int
    ) -> float:
        """Calculate size reduction percentage."""
        metrics = self.calculate_metrics(compression, pruning, quantization_bits)
        reduction = (self.baseline.model_size_mb - metrics.model_size_mb) / self.baseline.model_size_mb * 100
        return max(0, reduction)


# Global calculator instance
metrics_calculator = MetricsCalculator()
