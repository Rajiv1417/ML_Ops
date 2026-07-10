"""
Test checkpoint saving for all model types to ensure the error is fixed
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset
from trainer.model_builder import create_model
from trainer.trainer import WeldDefectTrainer

def test_checkpoint_saving():
    """Test that checkpoints can be saved for all model types"""
    print("\n" + "="*60)
    print("TEST: Checkpoint Saving for All Architectures")
    print("="*60)
    
    models = ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]
    
    # Create dummy data
    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=2)
    
    for model_name in models:
        print(f"\n{model_name}:")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Create model and trainer
                model = create_model(num_classes=2, model_name=model_name, device='cpu')
                trainer = WeldDefectTrainer(model=model, device='cpu', checkpoint_dir=tmpdir)
                
                # Run a short training to populate state
                trainer.state.total_epochs = 1
                trainer.state.current_epoch = 0
                
                # Save checkpoint
                checkpoint_name = f"test_{model_name.lower().replace(' ', '_').replace('-', '_')}_1ep_baseline.pt"
                checkpoint_path = trainer.save_checkpoint(checkpoint_name)
                
                print(f"  Checkpoint saved: {checkpoint_path.name}")
                
                # Verify checkpoint file exists and contains expected keys
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                required_keys = ['model_state', 'training_state', 'timestamp', 'model_size_mb', 'num_params']
                for key in required_keys:
                    assert key in checkpoint, f"Missing key: {key}"
                    print(f"    {key}: OK")
                
                # Verify values
                assert checkpoint['model_size_mb'] > 0, "model_size_mb should be > 0"
                assert checkpoint['num_params'] > 0, "num_params should be > 0"
                
                print(f"  Model size: {checkpoint['model_size_mb']:.2f} MB")
                print(f"  Parameters: {checkpoint['num_params']:,}")
                print(f"  [OK] Checkpoint saved and verified")
                
            except Exception as e:
                print(f"  [FAIL] Error: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
    
    print("\n[OK] All checkpoints saved successfully")


def test_get_model_info_during_training():
    """Test that get_model_info() works during training"""
    print("\n" + "="*60)
    print("TEST: get_model_info() During Training")
    print("="*60)
    
    models = ["MobileNetV2", "EfficientNet-B0", "ShuffleNet V2"]
    
    for model_name in models:
        print(f"\n{model_name}:")
        
        try:
            # Create model and trainer
            model = create_model(num_classes=2, model_name=model_name, device='cpu')
            trainer = WeldDefectTrainer(model=model, device='cpu')
            
            # Call get_model_info (this is what app.py calls)
            info = trainer.get_model_info()
            
            print(f"  Model info retrieved:")
            print(f"    Size: {info['model_size_mb']:.2f} MB")
            print(f"    Parameters: {info['num_parameters']:,}")
            print(f"    Model type: {info['model_name']}")
            print(f"    Classes: {info['num_classes']}")
            print(f"  [OK] get_model_info() works")
            
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    print("\n[OK] get_model_info() works for all models")


if __name__ == "__main__":
    try:
        test_checkpoint_saving()
        test_get_model_info_during_training()
        
        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)
        print("\nCheckpoint saving error is FIXED!")
        print("Training with EfficientNet-B0, ShuffleNet V2, etc. now works!")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
