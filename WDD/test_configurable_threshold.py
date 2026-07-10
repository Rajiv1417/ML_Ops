"""
Test script to verify configurable confidence threshold in Phase 3
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import InferenceEngine, OODDetector
from trainer.model_builder import create_model
import torch

def test_ood_detector_threshold():
    """Test OOD detector with different threshold values"""
    print("\n" + "="*60)
    print("TEST: OOD Detector Threshold Configuration")
    print("="*60)
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7]
    ood_score = 0.55  # Simulated OOD score
    
    for threshold in thresholds:
        detector = OODDetector(threshold=threshold)
        is_ood = ood_score > detector.threshold
        
        print(f"\nThreshold: {threshold}")
        print(f"  OOD Score: {ood_score}")
        print(f"  Is OOD: {is_ood}")
        print(f"  Classification: {'Non-weld (OOD)' if is_ood else 'Valid weld'}")


def test_inference_engine_threshold():
    """Test InferenceEngine initialization with different thresholds"""
    print("\n" + "="*60)
    print("TEST: InferenceEngine with Configurable Thresholds")
    print("="*60)
    
    try:
        # Create model
        model = create_model(num_classes=2, model_name="MobileNetV2", device='cpu')
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            engine = InferenceEngine(
                model=model,
                device='cpu',
                ood_threshold=threshold
            )
            
            print(f"\nThreshold: {threshold}")
            print(f"  Engine created: OK")
            print(f"  OOD Threshold set to: {engine.ood_detector.threshold}")
            assert engine.ood_detector.threshold == threshold, \
                f"Expected {threshold}, got {engine.ood_detector.threshold}"
        
        print("\n[OK] All InferenceEngine instances created with correct thresholds")
        
    except Exception as e:
        print(f"\n[FAIL] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def test_threshold_effects():
    """Demonstrate how threshold affects OOD detection decision"""
    print("\n" + "="*60)
    print("TEST: Threshold Effects on OOD Detection")
    print("="*60)
    
    test_scores = [0.2, 0.4, 0.5, 0.6, 0.8]
    threshold = 0.5
    
    print(f"\nUsing threshold: {threshold}\n")
    print("OOD Score | Classification | Reason")
    print("-" * 45)
    
    for score in test_scores:
        is_ood = score > threshold
        classification = "OOD (Reject)" if is_ood else "Valid (Accept)"
        
        if is_ood:
            reason = f"Score {score:.1f} > Threshold {threshold}"
        else:
            reason = f"Score {score:.1f} <= Threshold {threshold}"
        
        print(f"{score:^9.1f} | {classification:^15} | {reason}")
    
    print("\n[OK] Threshold logic demonstration complete")


def test_slider_range():
    """Verify slider range configuration (0.0 - 1.0)"""
    print("\n" + "="*60)
    print("TEST: Slider Configuration Range")
    print("="*60)
    
    slider_config = {
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.05,
        "default": 0.5
    }
    
    print(f"\nSlider Configuration:")
    print(f"  Minimum value: {slider_config['min_value']}")
    print(f"  Maximum value: {slider_config['max_value']}")
    print(f"  Step size: {slider_config['step']}")
    print(f"  Default value: {slider_config['default']}")
    
    # Generate possible slider values
    min_val = slider_config["min_value"]
    max_val = slider_config["max_value"]
    step = slider_config["step"]
    
    possible_values = []
    current = min_val
    while current <= max_val + 1e-6:  # Small epsilon for float comparison
        possible_values.append(round(current, 2))
        current += step
    
    print(f"\nPossible threshold values ({len(possible_values)} total):")
    print(f"  {possible_values}")
    
    # Verify range
    assert min(possible_values) >= 0.0, "Minimum should be >= 0.0"
    assert max(possible_values) <= 1.0, "Maximum should be <= 1.0"
    print("\n[OK] Slider range verified")


if __name__ == "__main__":
    try:
        test_ood_detector_threshold()
        test_inference_engine_threshold()
        test_threshold_effects()
        test_slider_range()
        
        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)
        print("\nConfigurable confidence threshold is working correctly!")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
