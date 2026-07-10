"""
Quick test and demo script for the optimization engine.

Run this to verify all components are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulator.optimizer import OptimizationEngine
from simulator.scorer import DeploymentScorer
from simulator.profiles import ProfileManager
from simulator.metrics import metrics_calculator


def demo_optimization():
    """Demo the optimization engine."""
    print("=" * 80)
    print("Welding Defect Detection - Phase 2 Demo")
    print("=" * 80)
    print()
    
    # Initialize components
    optimizer = OptimizationEngine()
    scorer = DeploymentScorer()
    profiles = ProfileManager.get_profiles()
    
    # ==================== BASELINE ====================
    print("📊 BASELINE METRICS")
    print("-" * 80)
    baseline = metrics_calculator.baseline
    print(f"  Accuracy:    {baseline.accuracy:.1%}")
    print(f"  Model Size:  {baseline.model_size_mb:.1f} MB")
    print(f"  Latency:     {baseline.latency_ms:.1f} ms")
    print(f"  Throughput:  {baseline.throughput_fps:.1f} FPS")
    print()
    
    # ==================== OPTIMIZATION EXAMPLE 1: CONSERVATIVE ====================
    print("🔧 OPTIMIZATION #1: Conservative (light compression)")
    print("-" * 80)
    optimizer.reset()
    metrics1 = optimizer.apply_optimization(
        compression=0.2,  # 20% compression
        pruning=0.1,      # 10% pruning
        quantization_bits=16  # 16-bit quantization
    )
    
    print(f"  Settings: 20% compression, 10% pruning, 16-bit quantization")
    print(f"  Accuracy:    {metrics1.accuracy:.1%} ({(metrics1.accuracy - baseline.accuracy)*100:+.2f}%)")
    print(f"  Model Size:  {metrics1.model_size_mb:.1f} MB (-{(1 - metrics1.model_size_mb/baseline.model_size_mb)*100:.1f}%)")
    print(f"  Latency:     {metrics1.latency_ms:.1f} ms ({baseline.latency_ms/metrics1.latency_ms:.1f}x faster)")
    print(f"  Speedup:     {baseline.latency_ms/metrics1.latency_ms:.1f}x")
    print()
    
    # Score against Weld Vision profile
    weld_vision = profiles[0]
    score1 = scorer.score(metrics1, weld_vision)
    print(f"  Deployment Score (Weld Vision): {score1.total_score:.0f}/100")
    print(f"  Deployable: {'✅ YES' if score1.is_deployable else '❌ NO'}")
    if score1.violations:
        for v in score1.violations:
            print(f"    - {v.constraint_name}: {v.current_value:.1f} vs {v.required_value:.1f} {v.unit}")
    print()
    
    # ==================== OPTIMIZATION EXAMPLE 2: AGGRESSIVE ====================
    print("🔧 OPTIMIZATION #2: Aggressive (maximum compression)")
    print("-" * 80)
    optimizer.reset()
    metrics2 = optimizer.apply_optimization(
        compression=0.8,  # 80% compression
        pruning=0.7,      # 70% pruning
        quantization_bits=4  # 4-bit quantization
    )
    
    print(f"  Settings: 80% compression, 70% pruning, 4-bit quantization")
    print(f"  Accuracy:    {metrics2.accuracy:.1%} ({(metrics2.accuracy - baseline.accuracy)*100:+.2f}%)")
    print(f"  Model Size:  {metrics2.model_size_mb:.1f} MB (-{(1 - metrics2.model_size_mb/baseline.model_size_mb)*100:.1f}%)")
    print(f"  Latency:     {metrics2.latency_ms:.1f} ms ({baseline.latency_ms/metrics2.latency_ms:.1f}x faster)")
    print(f"  Sparsity:    {metrics2.sparsity:.1%}")
    print()
    
    # Score against Weld Vision profile
    score2 = scorer.score(metrics2, weld_vision)
    print(f"  Deployment Score (Weld Vision): {score2.total_score:.0f}/100")
    print(f"  Deployable: {'✅ YES' if score2.is_deployable else '❌ NO'}")
    if score2.violations:
        for v in score2.violations:
            print(f"    - {v.constraint_name}: {v.current_value:.1f} vs {v.required_value:.1f} {v.unit}")
    print()
    
    # ==================== OPTIMIZATION EXAMPLE 3: BALANCED ====================
    print("🔧 OPTIMIZATION #3: Balanced (middle ground)")
    print("-" * 80)
    optimizer.reset()
    metrics3 = optimizer.apply_optimization(
        compression=0.4,  # 40% compression
        pruning=0.3,      # 30% pruning
        quantization_bits=8  # 8-bit quantization
    )
    
    print(f"  Settings: 40% compression, 30% pruning, 8-bit quantization")
    print(f"  Accuracy:    {metrics3.accuracy:.1%} ({(metrics3.accuracy - baseline.accuracy)*100:+.2f}%)")
    print(f"  Model Size:  {metrics3.model_size_mb:.1f} MB (-{(1 - metrics3.model_size_mb/baseline.model_size_mb)*100:.1f}%)")
    print(f"  Latency:     {metrics3.latency_ms:.1f} ms ({baseline.latency_ms/metrics3.latency_ms:.1f}x faster)")
    print()
    
    # Score against all profiles
    print("  Deployment Scores across all profiles:")
    for profile in profiles:
        score = scorer.score(metrics3, profile)
        status = "✅ PASS" if score.is_deployable else "❌ FAIL"
        print(f"    - {profile.name:20s}: {score.total_score:5.0f}/100 {status}")
    print()
    
    # ==================== DEPLOYMENT PROFILES ====================
    print("🎯 DEPLOYMENT PROFILES")
    print("-" * 80)
    for i, profile in enumerate(profiles, 1):
        print(f"{i}. {profile.name}")
        print(f"   Description: {profile.description}")
        print(f"   Constraints: Max {profile.max_model_size_mb:.0f}MB | "
              f"Max {profile.max_latency_ms:.0f}ms | Min {profile.min_accuracy:.0%} accuracy")
        print()
    
    # ==================== RECOMMENDATIONS ====================
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("-" * 80)
    optimizer.reset()
    optimizer.apply_optimization(0.3, 0.2, 8)
    recommendations = optimizer.recommend_optimization()
    print(f"  Compression: {recommendations['compression']}")
    print(f"  Pruning:     {recommendations['pruning']}")
    print(f"  Quantization: {recommendations['quantization']}")
    print()
    
    print("=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    demo_optimization()
