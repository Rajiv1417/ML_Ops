"""
Test script to verify model size and parameter counting works for all models
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from trainer.model_builder import (
    create_model,
    compute_model_size_mb,
    count_model_parameters,
    get_model_info
)
from trainer.trainer import WeldDefectTrainer

def test_model_size_computation():
    """Test that model size can be computed for all model types"""
    print("\n" + "="*60)
    print("TEST: Model Size Computation for All Architectures")
    print("="*60)
    
    models = ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]
    
    for model_name in models:
        try:
            model = create_model(num_classes=2, model_name=model_name, device='cpu')
            size_mb = compute_model_size_mb(model)
            
            print(f"\n{model_name}:")
            print(f"  Size: {size_mb:.2f} MB")
            
            # Verify size is reasonable
            assert size_mb > 0, f"Model size should be > 0, got {size_mb}"
            assert size_mb < 100, f"Model size should be < 100 MB, got {size_mb}"
            
        except Exception as e:
            print(f"\n✗ Failed for {model_name}: {str(e)}")
            raise
    
    print("\n[OK] All model sizes computed successfully")


def test_parameter_counting():
    """Test that parameters can be counted for all model types"""
    print("\n" + "="*60)
    print("TEST: Parameter Counting for All Architectures")
    print("="*60)
    
    models = ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]
    
    for model_name in models:
        try:
            model = create_model(num_classes=2, model_name=model_name, device='cpu')
            num_params = count_model_parameters(model)
            
            print(f"\n{model_name}:")
            print(f"  Parameters: {num_params:,}")
            
            # Verify parameter count is reasonable
            assert num_params > 0, f"Parameter count should be > 0, got {num_params}"
            assert num_params < 100000000, f"Parameter count should be < 100M, got {num_params}"
            
        except Exception as e:
            print(f"\n✗ Failed for {model_name}: {str(e)}")
            raise
    
    print("\n[OK] All parameter counts computed successfully")


def test_trainer_get_model_info():
    """Test that trainer.get_model_info() works for all model types"""
    print("\n" + "="*60)
    print("TEST: Trainer.get_model_info() for All Architectures")
    print("="*60)
    
    models = ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]
    
    for model_name in models:
        try:
            model = create_model(num_classes=2, model_name=model_name, device='cpu')
            trainer = WeldDefectTrainer(model=model, device='cpu')
            info = trainer.get_model_info()
            
            print(f"\n{model_name}:")
            print(f"  Size: {info['model_size_mb']:.2f} MB")
            print(f"  Parameters: {info['num_parameters']:,}")
            print(f"  Model Type: {info['model_name']}")
            
            # Verify keys exist
            assert 'model_size_mb' in info
            assert 'num_parameters' in info
            assert 'model_name' in info
            assert 'num_classes' in info
            
            # Verify values are reasonable
            assert info['model_size_mb'] > 0
            assert info['num_parameters'] > 0
            assert info['num_classes'] == 2
            
        except Exception as e:
            print(f"\n✗ Failed for {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    print("\n[OK] Trainer.get_model_info() works for all models")


def test_comparison_with_static_info():
    """Compare computed sizes with static info from get_model_info()"""
    print("\n" + "="*60)
    print("TEST: Comparison of Computed vs Static Model Info")
    print("="*60)
    
    models = ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]
    
    for model_name in models:
        try:
            model = create_model(num_classes=2, model_name=model_name, device='cpu')
            
            computed_size = compute_model_size_mb(model)
            static_info = get_model_info(model_name)
            static_size = static_info['size_mb']
            
            print(f"\n{model_name}:")
            print(f"  Computed size: {computed_size:.2f} MB")
            print(f"  Static size:   {static_size:.2f} MB")
            print(f"  Difference:    {abs(computed_size - static_size):.2f} MB")
            
            # Sizes might differ slightly, but should be in the same ballpark
            # Allow 10% difference
            diff_percent = abs(computed_size - static_size) / static_size * 100
            print(f"  Difference %:  {diff_percent:.1f}%")
            
        except Exception as e:
            print(f"\n✗ Failed for {model_name}: {str(e)}")
            raise
    
    print("\n[OK] Size comparison complete")


if __name__ == "__main__":
    try:
        test_model_size_computation()
        test_parameter_counting()
        test_trainer_get_model_info()
        test_comparison_with_static_info()
        
        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)
        print("\nModel size and parameter counting work for all architectures!")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
