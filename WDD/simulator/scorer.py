"""
Deployment scoring system.

Validates model against deployment constraints and provides deployment readiness score.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from simulator.metrics import ModelMetrics
from simulator.profiles import DeploymentProfile


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    
    constraint_name: str
    required_value: float
    current_value: float
    unit: str
    severity: str  # 'error', 'warning', 'info'
    recommendation: str


@dataclass
class DeploymentScore:
    """Overall deployment readiness score."""
    
    total_score: float  # 0-100
    accuracy_score: float
    size_score: float
    latency_score: float
    throughput_score: float
    
    is_deployable: bool  # All hard constraints met?
    violations: List[ConstraintViolation]
    warnings: List[str]
    recommendations: List[str]


class DeploymentScorer:
    """Scores model against deployment constraints."""
    
    def __init__(self):
        """Initialize scorer."""
        pass
    
    def score(self, metrics: ModelMetrics, profile: DeploymentProfile) -> DeploymentScore:
        """
        Score model against deployment constraints.
        
        Args:
            metrics: Current model metrics
            profile: Deployment profile for weld inspection
        
        Returns:
            DeploymentScore with detailed feedback
        """
        violations = []
        warnings = []
        recommendations = []
        
        # ==================== CHECK HARD CONSTRAINTS ====================
        
        # Accuracy constraint
        accuracy_pass = metrics.accuracy >= profile.min_accuracy
        if not accuracy_pass:
            violations.append(ConstraintViolation(
                constraint_name="Minimum Accuracy",
                required_value=profile.min_accuracy,
                current_value=metrics.accuracy,
                unit="",
                severity="error",
                recommendation=f"Reduce optimization (compression/pruning) to improve accuracy to {profile.min_accuracy:.1%}"
            ))
        
        # Model size constraint
        size_pass = metrics.model_size_mb <= profile.max_model_size_mb
        if not size_pass:
            violations.append(ConstraintViolation(
                constraint_name="Maximum Model Size",
                required_value=profile.max_model_size_mb,
                current_value=metrics.model_size_mb,
                unit="MB",
                severity="error",
                recommendation=f"Increase optimization (compression/pruning/quantization) to reduce size to {profile.max_model_size_mb:.1f} MB"
            ))
        
        # Latency constraint
        latency_pass = metrics.latency_ms <= profile.max_latency_ms
        if not latency_pass:
            violations.append(ConstraintViolation(
                constraint_name="Maximum Latency",
                required_value=profile.max_latency_ms,
                current_value=metrics.latency_ms,
                unit="ms",
                severity="error",
                recommendation=f"Reduce latency to {profile.max_latency_ms:.0f}ms by applying more optimization"
            ))
        
        # ==================== CHECK SOFT CONSTRAINTS ====================
        
        # Preferred accuracy
        if metrics.accuracy < profile.preferred_accuracy:
            gap = (profile.preferred_accuracy - metrics.accuracy) * 100
            warnings.append(f"Accuracy {gap:.1f}% below preferred level ({profile.preferred_accuracy:.1%})")
        
        # Preferred throughput
        if metrics.throughput_fps < profile.preferred_throughput_fps * 0.8:  # 20% tolerance
            warnings.append(f"Throughput below preferred level ({metrics.throughput_fps:.1f} vs {profile.preferred_throughput_fps:.1f} FPS)")
        
        # ==================== CALCULATE SCORES ====================
        
        # Accuracy score: 0-100 (100 at preferred, scales down to min)
        if metrics.accuracy >= profile.preferred_accuracy:
            accuracy_score = 100.0
        elif metrics.accuracy >= profile.min_accuracy:
            ratio = (metrics.accuracy - profile.min_accuracy) / (profile.preferred_accuracy - profile.min_accuracy)
            accuracy_score = 50.0 + (ratio * 50.0)
        else:
            accuracy_score = max(0, (metrics.accuracy / profile.min_accuracy) * 50.0)
        
        # Size score: how close to limit
        if metrics.model_size_mb <= profile.max_model_size_mb:
            size_ratio = 1.0 - (metrics.model_size_mb / profile.max_model_size_mb)
            size_score = min(100, 50 + (size_ratio * 50))
        else:
            excess = (metrics.model_size_mb - profile.max_model_size_mb) / profile.max_model_size_mb
            size_score = max(0, 50 - (excess * 50))
        
        # Latency score: how close to limit
        if metrics.latency_ms <= profile.max_latency_ms:
            latency_ratio = 1.0 - (metrics.latency_ms / profile.max_latency_ms)
            latency_score = min(100, 50 + (latency_ratio * 50))
        else:
            excess = (metrics.latency_ms - profile.max_latency_ms) / profile.max_latency_ms
            latency_score = max(0, 50 - (excess * 50))
        
        # Throughput score: bonus for high throughput
        throughput_score = min(100, (metrics.throughput_fps / profile.preferred_throughput_fps) * 100)
        
        # ==================== GENERATE RECOMMENDATIONS ====================
        
        if accuracy_pass and size_pass and latency_pass:
            recommendations.append("✅ Model meets all hard constraints - ready for deployment!")
            
            # If any soft constraints not met, suggest improvements
            if warnings:
                recommendations.append("💡 Consider further optimization for better performance")
        else:
            if not accuracy_pass:
                recommendations.append("❌ Accuracy too low - reduce optimization to preserve model quality")
            if not size_pass:
                recommendations.append("❌ Model too large - increase compression/pruning/quantization")
            if not latency_pass:
                recommendations.append("❌ Inference too slow - more aggressive optimization needed")
        
        # ==================== OVERALL SCORE ====================
        
        # Weighted average of component scores
        # Hard constraints have more weight if violated
        weights = [0.4, 0.25, 0.25, 0.1]  # accuracy, size, latency, throughput
        
        if accuracy_pass and size_pass and latency_pass:
            # All hard constraints met - focus on quality
            total_score = (accuracy_score * weights[0] + 
                         size_score * weights[1] + 
                         latency_score * weights[2] + 
                         throughput_score * weights[3])
        else:
            # Hard constraints violated - penalize heavily
            constraint_failures = sum([not accuracy_pass, not size_pass, not latency_pass])
            total_score = (accuracy_score * weights[0] + 
                         size_score * weights[1] + 
                         latency_score * weights[2] + 
                         throughput_score * weights[3]) * (1.0 - constraint_failures * 0.3)
        
        total_score = max(0, min(100, total_score))
        
        is_deployable = accuracy_pass and size_pass and latency_pass
        
        return DeploymentScore(
            total_score=total_score,
            accuracy_score=accuracy_score,
            size_score=size_score,
            latency_score=latency_score,
            throughput_score=throughput_score,
            is_deployable=is_deployable,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def get_next_step(self, score: DeploymentScore) -> str:
        """Get recommended next optimization step."""
        if score.is_deployable:
            return "✅ Ready for deployment!"
        
        # Find the biggest constraint violation
        if not score.violations:
            return "🔍 Checking constraints..."
        
        worst_violation = max(score.violations, key=lambda v: (v.severity == 'error', abs(v.required_value - v.current_value)))
        
        return f"⚠️ Focus on: {worst_violation.constraint_name} ({worst_violation.recommendation})"
    
    def get_constraint_progress(self, metrics: ModelMetrics, profile: DeploymentProfile) -> Dict[str, Dict]:
        """
        Get progress towards each constraint.
        
        Returns:
            Dictionary with progress for each constraint
        """
        return {
            'accuracy': {
                'current': metrics.accuracy,
                'required': profile.min_accuracy,
                'preferred': profile.preferred_accuracy,
                'met': metrics.accuracy >= profile.min_accuracy,
                'progress_pct': min(100, (metrics.accuracy / profile.min_accuracy) * 100) if profile.min_accuracy > 0 else 100
            },
            'size': {
                'current': metrics.model_size_mb,
                'required': profile.max_model_size_mb,
                'met': metrics.model_size_mb <= profile.max_model_size_mb,
                'progress_pct': max(0, 100 - (metrics.model_size_mb / profile.max_model_size_mb * 100))
            },
            'latency': {
                'current': metrics.latency_ms,
                'required': profile.max_latency_ms,
                'met': metrics.latency_ms <= profile.max_latency_ms,
                'progress_pct': max(0, 100 - (metrics.latency_ms / profile.max_latency_ms * 100))
            }
        }


# Global scorer instance
scorer = DeploymentScorer()
