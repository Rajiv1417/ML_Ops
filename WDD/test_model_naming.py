#!/usr/bin/env python3
"""
Test script for model naming convention validation.
"""

from model_naming import ModelNaming


def test_baseline_naming():
    """Test baseline model naming."""
    print("=" * 60)
    print("TEST 1: Baseline Model Naming (Phase 1)")
    print("=" * 60)
    
    name = ModelNaming.generate_baseline_name(
        model_name="mobilenetv2",
        dataset_name="OpenSourceData01",
        epochs=5
    )
    print(f"✓ Generated name: {name}")
    assert name == "mobilenetv2_opensourcedata01_5ep_baseline.pt", f"Unexpected format: {name}"
    
    # Test parsing
    info = ModelNaming.parse_model_name(name)
    print(f"✓ Parsed info: {info}")
    assert info['is_baseline'] == True
    assert info['epochs'] == 5
    assert info['compression'] == 0.0
    
    # Test friendly name
    friendly = ModelNaming.get_friendly_name(name)
    print(f"✓ Friendly name: {friendly}")
    assert "📦 Baseline" in friendly
    
    print("✅ Baseline naming test PASSED\n")


def test_optimized_naming():
    """Test optimized model naming."""
    print("=" * 60)
    print("TEST 2: Optimized Model Naming (Phase 2)")
    print("=" * 60)
    
    name = ModelNaming.generate_optimized_name(
        model_name="mobilenetv2",
        dataset_name="OpenSourceData01",
        epochs=5,
        compression=0.3,
        pruning=0.2,
        quantization_bits=8
    )
    print(f"✓ Generated name: {name}")
    assert name == "mobilenetv2_opensourcedata01_5ep_0.3c_0.2p_8q.pt"
    
    # Test parsing
    info = ModelNaming.parse_model_name(name)
    print(f"✓ Parsed info: {info}")
    assert info['is_optimized'] == True
    assert info['compression'] == 0.3
    assert info['pruning'] == 0.2
    assert info['quantization_bits'] == 8
    
    # Test friendly name
    friendly = ModelNaming.get_friendly_name(name)
    print(f"✓ Friendly name: {friendly}")
    assert "⚡ Optimized" in friendly
    assert "30% compression" in friendly
    assert "20% pruning" in friendly
    
    print("✅ Optimized naming test PASSED\n")


def test_model_info_display():
    """Test model info display."""
    print("=" * 60)
    print("TEST 3: Model Info Display")
    print("=" * 60)
    
    # Baseline
    baseline_name = ModelNaming.generate_baseline_name("mobilenetv2", "OpenSourceData01", 10)
    info_display = ModelNaming.get_model_info_display(baseline_name)
    print(f"✓ Baseline info display:\n{info_display}\n")
    assert "Baseline" in info_display
    assert "10" in info_display
    
    # Optimized
    opt_name = ModelNaming.generate_optimized_name(
        "mobilenetv2", "OpenSourceData01", 10, 
        compression=0.5, pruning=0.25, quantization_bits=16
    )
    info_display = ModelNaming.get_model_info_display(opt_name)
    print(f"✓ Optimized info display:\n{info_display}\n")
    assert "Optimized" in info_display
    assert "50%" in info_display
    
    print("✅ Model info display test PASSED\n")


def test_parsing_edge_cases():
    """Test parsing edge cases."""
    print("=" * 60)
    print("TEST 4: Parsing Edge Cases")
    print("=" * 60)
    
    # Minimal optimization
    name = ModelNaming.generate_optimized_name(
        "mobilenetv2", "Data", 3, compression=0, pruning=0, quantization_bits=32
    )
    print(f"✓ Minimal optimization: {name}")
    info = ModelNaming.parse_model_name(name)
    assert info['is_optimized'] == False, "Minimal opt should not be flagged as optimized"
    
    # Partial optimization (only compression)
    name = ModelNaming.generate_optimized_name(
        "mobilenetv2", "Data", 3, compression=0.2, pruning=0, quantization_bits=32
    )
    print(f"✓ Partial optimization (compression only): {name}")
    info = ModelNaming.parse_model_name(name)
    assert info['is_optimized'] == True
    assert info['compression'] == 0.2
    
    print("✅ Edge case parsing test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MODEL NAMING CONVENTION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_baseline_naming()
        test_optimized_naming()
        test_model_info_display()
        test_parsing_edge_cases()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nModel naming convention is working correctly across all phases:")
        print("  ✓ Phase 1 baseline: mobilenetv2_datasource_Nep_baseline.pt")
        print("  ✓ Phase 2 optimized: mobilenetv2_datasource_Nep_Xc_Yp_Zq.pt")
        print("  ✓ Phase 3 parsing and friendly display names")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
