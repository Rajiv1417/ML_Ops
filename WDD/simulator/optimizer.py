"""
Core optimization engine for model compression, pruning, and quantization.

Provides the logic for applying multiple optimization techniques and
calculating their combined effects.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from simulator.metrics import ModelMetrics, metrics_calculator


@dataclass
class OptimizationState:
    """Current state of model optimization."""
    
    compression: float = 0.0  # 0-1
    pruning: float = 0.0      # 0-1
    quantization_bits: int = 32
    target_accuracy: float = 1.0  # 0.5-1.0 (50%-100%)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'compression': self.compression,
            'pruning': self.pruning,
            'quantization_bits': self.quantization_bits,
            'target_accuracy': self.target_accuracy
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'OptimizationState':
        """Create from dictionary."""
        return OptimizationState(
            compression=data.get('compression', 0.0),
            pruning=data.get('pruning', 0.0),
            quantization_bits=data.get('quantization_bits', 32),
            target_accuracy=data.get('target_accuracy', 1.0)
        )


@dataclass
class OptimizationHistory:
    """Tracks optimization history for visualization."""
    
    compressions: List[float]
    prunings: List[float]
    quantizations: List[int]
    accuracies: List[float]
    sizes_mb: List[float]
    latencies_ms: List[float]
    
    def add_step(self, state: OptimizationState, metrics: ModelMetrics):
        """Add a step to history."""
        self.compressions.append(state.compression)
        self.prunings.append(state.pruning)
        self.quantizations.append(state.quantization_bits)
        self.accuracies.append(metrics.accuracy)
        self.sizes_mb.append(metrics.model_size_mb)
        self.latencies_ms.append(metrics.latency_ms)
    
    def clear(self):
        """Clear history."""
        self.compressions.clear()
        self.prunings.clear()
        self.quantizations.clear()
        self.accuracies.clear()
        self.sizes_mb.clear()
        self.latencies_ms.clear()


class OptimizationEngine:
    """Core optimization engine."""
    
    def __init__(self):
        """Initialize optimization engine."""
        self.current_state = OptimizationState()
        self.current_metrics = None
        self.history = OptimizationHistory([], [], [], [], [], [])
        self._update_metrics()
    
    def _update_metrics(self):
        """Update current metrics based on state."""
        self.current_metrics = metrics_calculator.calculate_metrics(
            compression=self.current_state.compression,
            pruning=self.current_state.pruning,
            quantization_bits=self.current_state.quantization_bits,
            target_accuracy=self.current_state.target_accuracy
        )
    
    def set_compression(self, value: float) -> ModelMetrics:
        """
        Set compression level.
        
        Args:
            value: Compression level 0-1 (0% to 100%)
        
        Returns:
            Updated metrics
        """
        self.current_state.compression = max(0.0, min(1.0, value))
        self._update_metrics()
        return self.current_metrics
    
    def set_pruning(self, value: float) -> ModelMetrics:
        """
        Set pruning level.
        
        Args:
            value: Pruning level 0-0.95 (0% to 95%)
        
        Returns:
            Updated metrics
        """
        self.current_state.pruning = max(0.0, min(0.95, value))
        self._update_metrics()
        return self.current_metrics
    
    def set_quantization_bits(self, bits: int) -> ModelMetrics:
        """
        Set quantization bits.
        
        Args:
            bits: Number of bits (4, 8, 16, or 32)
        
        Returns:
            Updated metrics
        """
        valid_bits = [4, 8, 16, 32]
        if bits in valid_bits:
            self.current_state.quantization_bits = bits
            self._update_metrics()
        return self.current_metrics
    
    def apply_optimization(
        self,
        compression: float,
        pruning: float,
        quantization_bits: int,
        target_accuracy: float = 1.0
    ) -> ModelMetrics:
        """
        Apply all optimizations at once.
        
        Args:
            compression: Compression level 0-1
            pruning: Pruning level 0-0.95
            quantization_bits: Quantization bits (4, 8, 16, 32)
            target_accuracy: Target accuracy to maintain (0.5-1.0)
        
        Returns:
            Updated metrics
        """
        self.set_compression(compression)
        self.set_pruning(pruning)
        self.set_quantization_bits(quantization_bits)
        self.current_state.target_accuracy = max(0.5, min(1.0, target_accuracy))
        
        # Recalculate metrics with target accuracy constraint
        self.current_metrics = metrics_calculator.calculate_metrics(
            compression=self.current_state.compression,
            pruning=self.current_state.pruning,
            quantization_bits=self.current_state.quantization_bits,
            target_accuracy=self.current_state.target_accuracy
        )
        return self.current_metrics
    
    def record_step(self):
        """Record current state to history."""
        if self.current_metrics:
            self.history.add_step(self.current_state, self.current_metrics)
    
    def reset(self):
        """Reset to baseline (no optimization)."""
        self.current_state = OptimizationState()
        self.history.clear()
        self._update_metrics()
    
    def get_current_state(self) -> OptimizationState:
        """Get current optimization state."""
        return self.current_state
    
    def get_current_metrics(self) -> ModelMetrics:
        """Get current metrics."""
        return self.current_metrics
    
    def get_history(self) -> OptimizationHistory:
        """Get optimization history."""
        return self.history
    
    def get_optimization_summary(self) -> Dict:
        """
        Get summary of current optimization.
        
        Returns:
            Dictionary with optimization summary
        """
        baseline = metrics_calculator.baseline
        
        accuracy_loss = baseline.accuracy - self.current_metrics.accuracy
        size_reduction = (baseline.model_size_mb - self.current_metrics.model_size_mb) / baseline.model_size_mb * 100
        speedup = baseline.latency_ms / self.current_metrics.latency_ms
        
        return {
            'baseline_accuracy': baseline.accuracy,
            'current_accuracy': self.current_metrics.accuracy,
            'accuracy_loss': accuracy_loss,
            'accuracy_loss_pct': accuracy_loss * 100,
            
            'baseline_size_mb': baseline.model_size_mb,
            'current_size_mb': self.current_metrics.model_size_mb,
            'size_reduction_pct': size_reduction,
            
            'baseline_latency_ms': baseline.latency_ms,
            'current_latency_ms': self.current_metrics.latency_ms,
            'speedup': speedup,
            
            'baseline_throughput_fps': baseline.throughput_fps,
            'current_throughput_fps': self.current_metrics.throughput_fps,
            
            'sparsity': self.current_metrics.sparsity,
            'compression_ratio': self.current_metrics.compression_ratio,
            
            'state': self.current_state.to_dict()
        }
    
    def recommend_optimization(self) -> Dict:
        """
        Provide optimization recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            'compression': self._recommend_compression(),
            'pruning': self._recommend_pruning(),
            'quantization': self._recommend_quantization(),
        }
        return recommendations
    
    def _recommend_compression(self) -> str:
        """Recommend compression level."""
        if self.current_state.compression < 0.3:
            return "💡 Try increasing compression for faster inference"
        elif self.current_state.compression < 0.7:
            return "💡 Compression is moderate. Can go higher for more speedup."
        else:
            return "⚠️ High compression. Watch accuracy loss carefully."
    
    def _recommend_pruning(self) -> str:
        """Recommend pruning level."""
        if self.current_state.pruning < 0.2:
            return "💡 Pruning is conservative. Try 20-30% for better compression."
        elif self.current_state.pruning < 0.5:
            return "💡 Pruning level is good. More is possible with care."
        elif self.current_state.pruning < 0.8:
            return "⚠️ High pruning. Model sparsity is increasing significantly."
        else:
            return "❌ Very high pruning. Accuracy degradation likely."
    
    def _recommend_quantization(self) -> str:
        """Recommend quantization bits."""
        bits = self.current_state.quantization_bits
        if bits == 32:
            return "💡 Quantization not applied. Try 8-bit for 75% size reduction."
        elif bits == 16:
            return "💡 16-bit quantization is good. Can try 8-bit for more savings."
        elif bits == 8:
            return "💡 8-bit is a good balance of size and accuracy."
        elif bits == 4:
            return "⚠️ 4-bit quantization is aggressive. Check accuracy!"
        
        return ""
    
    def get_tradeoff_analysis(self) -> Dict:
        """
        Analyze trade-offs in current optimization.
        
        Returns:
            Dictionary with trade-off analysis
        """
        baseline = metrics_calculator.baseline
        
        return {
            'accuracy_vs_size': {
                'baseline_accuracy': baseline.accuracy,
                'baseline_size': baseline.model_size_mb,
                'current_accuracy': self.current_metrics.accuracy,
                'current_size': self.current_metrics.model_size_mb,
                'accuracy_per_mb': self.current_metrics.accuracy / self.current_metrics.model_size_mb if self.current_metrics.model_size_mb > 0 else 0,
            },
            'accuracy_vs_latency': {
                'baseline_accuracy': baseline.accuracy,
                'baseline_latency': baseline.latency_ms,
                'current_accuracy': self.current_metrics.accuracy,
                'current_latency': self.current_metrics.latency_ms,
            },
            'size_vs_latency': {
                'baseline_size': baseline.model_size_mb,
                'baseline_latency': baseline.latency_ms,
                'current_size': self.current_metrics.model_size_mb,
                'current_latency': self.current_metrics.latency_ms,
            }
        }


# Global optimizer instance
optimizer = OptimizationEngine()
