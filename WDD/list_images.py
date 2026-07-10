from pathlib import Path

print("=" * 80)
print("DATASET 1: OpenSourceData01 - Real Weld Images")
print("=" * 80)

opensourcedata = Path("data/OpenSourceData01")
if opensourcedata.exists():
    # Normal welds
    normal_dir = opensourcedata / "test" / "normal"
    if normal_dir.exists():
        normal_images = sorted(list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.png")))
        print(f"\n✓ Valid Weld Images (Normal) - {len(normal_images)} available:")
        for i, img in enumerate(normal_images[:5], 1):
            print(f"  {i}. data/OpenSourceData01/test/normal/{img.name}")
    
    # Defective welds
    defect_dir = opensourcedata / "test" / "defect"
    if defect_dir.exists():
        defect_images = sorted(list(defect_dir.glob("*.jpg")) + list(defect_dir.glob("*.jpeg")) + list(defect_dir.glob("*.png")))
        print(f"\n✓ Valid Weld Images (Defective) - {len(defect_images)} available:")
        for i, img in enumerate(defect_images[:5], 1):
            print(f"  {i}. data/OpenSourceData01/test/defect/{img.name}")

print("\n" + "=" * 80)
print("DATASET 2: weld_defects - Synthetic Weld Images")
print("=" * 80)

weld_defects = Path("data/weld_defects")
if weld_defects.exists():
    # Normal welds
    normal_dir = weld_defects / "test" / "normal"
    if normal_dir.exists():
        normal_images = sorted(list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.png")))
        print(f"\n✓ Valid Weld Images (Normal) - {len(normal_images)} available:")
        for i, img in enumerate(normal_images[:5], 1):
            print(f"  {i}. data/weld_defects/test/normal/{img.name}")
    
    # Defective welds
    defect_dir = weld_defects / "test" / "defect"
    if defect_dir.exists():
        defect_images = sorted(list(defect_dir.glob("*.jpg")) + list(defect_dir.glob("*.jpeg")) + list(defect_dir.glob("*.png")))
        print(f"\n✓ Valid Weld Images (Defective) - {len(defect_images)} available:")
        for i, img in enumerate(defect_images[:5], 1):
            print(f"  {i}. data/weld_defects/test/defect/{img.name}")

print("\n" + "=" * 80)
print("BASELINE MODELS AVAILABLE")
print("=" * 80)

from config import CHECKPOINTS_DIR
checkpoints = sorted([f.name for f in CHECKPOINTS_DIR.glob("*baseline.pt")])
for i, cp in enumerate(checkpoints, 1):
    print(f"{i}. {cp}")

print("\n" + "=" * 80)
print("NON-WELD IMAGES (for OOD testing)")
print("=" * 80)
print("\nTo test OOD detection, you need to provide external non-weld images.")
print("The OOD detection is failing because it's too strict or misconfigured.")
print("\nThe issue is likely in the OOD detection heuristics that need adjustment.")
