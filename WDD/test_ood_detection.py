"""Test OOD detection on valid weld images"""
import numpy as np
from pathlib import Path
from PIL import Image
from inference import OODDetector

# Test OOD detection on real weld images
print("=" * 80)
print("TESTING OOD DETECTION ON VALID WELD IMAGES")
print("=" * 80)

ood_detector = OODDetector()

# Real weld images that SHOULD pass OOD detection
real_weld_images = [
    "data/OpenSourceData01/test/normal/carbon-steel-good-welds_28_png_jpg.rf.0d13b8fd981d7a68d163528d61a2e534.jpg",
    "data/OpenSourceData01/test/defect/bad_weld_vid166_jpeg_jpg.rf.525052424b574a43de777660f6cb2d10.jpg",
    "data/weld_defects/test/normal/normal_0340.png",
    "data/weld_defects/test/defect/defect_0340.png",
]

print("\nTesting real weld images:")
print("-" * 80)

for img_path in real_weld_images:
    img_path = Path(img_path)
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img) / 255.0
        
        is_weld, ood_score, reasons = ood_detector.is_valid_weld_image(img_array)
        
        status = "✓ PASS" if is_weld else "✗ FAIL"
        print(f"{status}: {img_path.name}")
        print(f"       OOD Score: {ood_score:.3f}, Is Weld: {is_weld}")
        print(f"       Reasons: {reasons}")
        print()
    else:
        print(f"✗ File not found: {img_path}")

print("\n" + "=" * 80)
print("SUMMARY OF OOD DETECTION ISSUE")
print("=" * 80)
print("""
If valid weld images are being rejected as OOD:

1. The OOD detection thresholds may be too strict
2. The heuristics may not work well with the actual images
3. Need to analyze edge detection, texture variance, brightness patterns, etc.

Possible fixes:
1. Lower the ood_threshold parameter
2. Adjust individual heuristic weights
3. Use learned thresholds based on training data instead of fixed values
4. Add image normalization/preprocessing before OOD detection
""")
