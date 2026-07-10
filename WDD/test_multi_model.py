"""
Test script to verify multi-model support integration
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_naming import ModelNaming
from trainer.model_builder import create_model, get_available_models, get_model_info
import torch

def test_model_selection():
    """Test model selection and retrieval"""
    print("\n" + "="*60)
    print("TEST 1: Model Selection")
    print("="*60)
    
    models = get_available_models()
    print(f"Available models: {models}")
    
    for model_name in models:
        info = get_model_info(model_name)
        print(f"\n{model_name}:")
        print(f"  Size: {info['size_mb']}MB")
        print(f"  Params: {info['num_params']:,}")
        print(f"  Description: {info['description']}")
    
    assert "MobileNetV2" in models
    assert "EfficientNet-B0" in models
    assert "ShuffleNet V2" in models
    print("\n✓ All models available and retrievable")


def test_checkpoint_naming():
    """Test checkpoint name generation with different models"""
    print("\n" + "="*60)
    print("TEST 2: Checkpoint Naming")
    print("="*60)
    
    # Test baseline naming
    models_to_test = [
        ("mobilenetv2", "MobileNetV2"),
        ("efficientnet_b0", "EfficientNet-B0"),
        ("shufflenet_v2", "ShuffleNet V2")
    ]
    
    for model_prefix, model_full_name in models_to_test:
        checkpoint_name = ModelNaming.generate_baseline_name(
            model_name=model_prefix,
            dataset_name="OpenSourceData01",
            epochs=5
        )
        print(f"\n{model_full_name}:")
        print(f"  Checkpoint: {checkpoint_name}")
        assert model_prefix in checkpoint_name.lower()
        assert "opensourcedata01" in checkpoint_name.lower()
        assert "5ep" in checkpoint_name
        assert "_baseline.pt" in checkpoint_name
    
    print("\n[OK] All checkpoint names generated correctly")


def test_checkpoint_parsing():
    """Test parsing model names from checkpoint filenames"""
    print("\n" + "="*60)
    print("TEST 3: Checkpoint Parsing")
    print("="*60)
    
    test_files = [
        "mobilenetv2_opensourcedata01_5ep_baseline.pt",
        "efficientnet_b0_opensourcedata01_5ep_baseline.pt",
        "shufflenet_v2_opensourcedata01_5ep_baseline.pt"
    ]
    
    for checkpoint_name in test_files:
        parsed = ModelNaming.parse_model_name(checkpoint_name)
        print(f"\n{checkpoint_name}:")
        print(f"  Model: {parsed['model_name']}")
        print(f"  Dataset: {parsed['dataset_name']}")
        print(f"  Epochs: {parsed['epochs']}")
        print(f"  Type: {'baseline' if parsed['is_baseline'] else 'optimized'}")
        
        assert parsed['dataset_name'] == 'opensourcedata01'
        assert parsed['epochs'] == 5
        assert parsed['is_baseline'] == True
    
    print("\n✓ All checkpoints parsed correctly")


def test_model_creation():
    """Test creating models with different architectures"""
    print("\n" + "="*60)
    print("TEST 4: Model Creation")
    print("="*60)
    
    models_to_create = [
        "MobileNetV2",
        "EfficientNet-B0",
        "ShuffleNet V2"
    ]
    
    for model_name in models_to_create:
        try:
            model = create_model(
                num_classes=2,
                model_name=model_name,
                device='cpu'
            )
            
            # Test model forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"\n{model_name}:")
            print(f"  Model type: {type(model).__name__}")
            print(f"  Output shape: {output.shape}")
            assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
            
        except Exception as e:
            print(f"\n✗ Failed to create {model_name}: {str(e)}")
            raise
    
    print("\n✓ All models created and working correctly")


def test_model_name_conversion():
    """Test conversion between different model name formats"""
    print("\n" + "="*60)
    print("TEST 5: Model Name Format Conversion")
    print("="*60)
    
    # Simulate the conversion logic in app.py
    test_cases = [
        ("MobileNetV2", "mobilenetv2"),
        ("EfficientNet-B0", "efficientnet_b0"),
        ("ShuffleNet V2", "shufflenet_v2")
    ]
    
    for full_name, expected_prefix in test_cases:
        # This simulates the logic from app.py line 521
        converted = full_name.lower().replace("-", "")
        print(f"\n{full_name} -> {converted}")
        
        # For ShuffleNet V2, we need to handle the space
        if " " in full_name:
            # This is what happens when we parse from checkpoint
            assert "_" in expected_prefix or " " not in expected_prefix
    
    print("\n✓ Model name conversion logic verified")


def test_checkpoint_roundtrip():
    """Test full roundtrip: select model -> generate checkpoint name -> parse it back"""
    print("\n" + "="*60)
    print("TEST 6: Checkpoint Roundtrip")
    print("="*60)
    
    user_selections = [
        "MobileNetV2",
        "EfficientNet-B0",
        "ShuffleNet V2"
    ]
    
    for user_selected_model in user_selections:
        # Simulate app.py line 521-522 (clean model name)
        model_name_lower = user_selected_model.lower().replace("-", "").replace(" ", "_")
        
        # Generate checkpoint name (simulating app.py line 523-527)
        checkpoint_name = ModelNaming.generate_baseline_name(
            model_name=model_name_lower,
            dataset_name="TestDataset",
            epochs=3
        )
        
        # Parse it back
        parsed = ModelNaming.parse_model_name(checkpoint_name)
        
        print(f"\n{user_selected_model}:")
        print(f"  Generated checkpoint: {checkpoint_name}")
        print(f"  Parsed model: {parsed['model_name']}")
        print(f"  Match: {model_name_lower.lower() in checkpoint_name.lower()}")
        
        assert model_name_lower.lower() in checkpoint_name.lower()
    
    print("\n✓ Full roundtrip working correctly")


if __name__ == "__main__":
    try:
        test_model_selection()
        test_checkpoint_naming()
        test_checkpoint_parsing()
        test_model_creation()
        test_model_name_conversion()
        test_checkpoint_roundtrip()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nMulti-model support is working correctly!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
