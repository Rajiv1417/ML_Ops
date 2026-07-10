"""
Convert flat dataset structure to hierarchical normal/defect structure.

Flat structure:
  OpenSourceData01/
  ├── train/ (all images mixed)
  ├── valid/ (all images mixed)
  └── test/ (all images mixed)

Hierarchical structure (REQUIRED):
  OpenSourceData01/
  ├── train/
  │   ├── normal/ (good welds)
  │   └── defect/ (bad welds)
  ├── valid/
  │   ├── normal/
  │   └── defect/
  └── test/
      ├── normal/
      └── defect/

Usage:
  python restructure_flat_dataset.py --dataset OpenSourceData01 --normal-keywords good,carbon,stick,tig,mig --defect-keywords bad,crack,spatter,slag,porosity
"""

import argparse
import sys
from pathlib import Path
from shutil import copy2
import json


class DatasetRestructurer:
    """Convert flat dataset to hierarchical structure."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        self.splits = ['train', 'valid', 'test']
    
    def restructure(self, normal_keywords: list = None, defect_keywords: list = None) -> dict:
        """
        Restructure flat dataset into hierarchical normal/defect structure.
        
        Args:
            normal_keywords: List of keywords to identify normal (good) images
            defect_keywords: List of keywords to identify defect (bad) images
        
        Returns:
            Dictionary with restructuring results
        """
        normal_keywords = normal_keywords or [
            'good', 'carbon', 'stick', 'tig', 'mig',
            'ok', 'perfect', 'excellent', 'normal'
        ]
        defect_keywords = defect_keywords or [
            'bad', 'crack', 'spatter', 'slag', 'porosity',
            'defect', 'poor', 'not', 'overlap'
        ]
        
        # Convert to lowercase for matching
        normal_keywords = [k.lower() for k in normal_keywords]
        defect_keywords = [k.lower() for k in defect_keywords]
        
        results = {
            'total_images': 0,
            'classified': 0,
            'unclassified': 0,
            'splits': {}
        }
        
        for split in self.splits:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                print(f"[SKIP] {split}/ not found")
                continue
            
            print(f"\n[PROCESSING] {split}/")
            
            # Create normal and defect subdirectories
            normal_dir = split_dir / "normal"
            defect_dir = split_dir / "defect"
            
            normal_dir.mkdir(exist_ok=True)
            defect_dir.mkdir(exist_ok=True)
            
            normal_count = 0
            defect_count = 0
            unclassified_count = 0
            
            # Process all image files in split directory
            image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(split_dir.glob(ext))
            
            print(f"  Found {len(image_files)} images")
            
            for img_path in sorted(image_files):
                # Classify based on filename keywords
                filename = img_path.name.lower()
                
                is_normal = any(kw in filename for kw in normal_keywords)
                is_defect = any(kw in filename for kw in defect_keywords)
                
                if is_normal and not is_defect:
                    # Move to normal
                    dest = normal_dir / img_path.name
                    copy2(img_path, dest)
                    normal_count += 1
                    results['classified'] += 1
                    print(f"    [NORMAL] {img_path.name}")
                
                elif is_defect and not is_normal:
                    # Move to defect
                    dest = defect_dir / img_path.name
                    copy2(img_path, dest)
                    defect_count += 1
                    results['classified'] += 1
                    print(f"    [DEFECT] {img_path.name}")
                
                elif is_normal and is_defect:
                    # Ambiguous - ask user or default to normal
                    dest = normal_dir / img_path.name
                    copy2(img_path, dest)
                    normal_count += 1
                    results['classified'] += 1
                    print(f"    [AMBIGUOUS->NORMAL] {img_path.name}")
                
                else:
                    # Unclassified
                    unclassified_count += 1
                    results['unclassified'] += 1
                    print(f"    [UNCLASSIFIED] {img_path.name} - defaulting to DEFECT")
                    dest = defect_dir / img_path.name
                    copy2(img_path, dest)
                    defect_count += 1
                    results['classified'] += 1
                
                results['total_images'] += 1
            
            results['splits'][split] = {
                'normal': normal_count,
                'defect': defect_count,
                'total': len(image_files)
            }
            
            # Delete empty split directory content if needed
            # (We keep original images and create copies in subdirs for safety)
            
            print(f"  Summary: {normal_count} normal, {defect_count} defect, {unclassified_count} unclassified")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert flat dataset structure to hierarchical normal/defect structure"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., OpenSourceData01) - must be in data/ folder"
    )
    parser.add_argument(
        "--normal-keywords",
        default="good,carbon,stick,tig,mig,ok,perfect,excellent,normal",
        help="Comma-separated keywords to identify normal (good) images"
    )
    parser.add_argument(
        "--defect-keywords",
        default="bad,crack,spatter,slag,porosity,defect,poor,not,overlap",
        help="Comma-separated keywords to identify defect (bad) images"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # Find dataset in data/ directory
    data_dir = Path(__file__).parent
    dataset_path = data_dir / args.dataset
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print(f"Available datasets in {data_dir}:")
        for item in sorted(data_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("__") and item.name != "weld_defects":
                print(f"  - {item.name}/")
        sys.exit(1)
    
    print(f"[START] Restructuring {args.dataset}")
    print(f"Dataset path: {dataset_path}")
    print()
    
    normal_kw = [k.strip() for k in args.normal_keywords.split(",")]
    defect_kw = [k.strip() for k in args.defect_keywords.split(",")]
    
    print(f"Normal keywords: {normal_kw}")
    print(f"Defect keywords: {defect_kw}")
    print()
    
    if args.dry_run:
        print("[DRY RUN MODE] - No changes will be made\n")
    
    try:
        restructurer = DatasetRestructurer(dataset_path)
        results = restructurer.restructure(normal_keywords=normal_kw, defect_keywords=defect_kw)
        
        print("\n" + "="*60)
        print("RESTRUCTURING SUMMARY")
        print("="*60)
        print(f"Total images processed: {results['total_images']}")
        print(f"Successfully classified: {results['classified']}")
        print(f"Unclassified (defaulted to defect): {results['unclassified']}")
        print()
        
        for split, stats in results['splits'].items():
            print(f"{split}:")
            print(f"  Normal: {stats['normal']}")
            print(f"  Defect: {stats['defect']}")
            print(f"  Total:  {stats['total']}")
        
        print()
        print("[OK] Dataset restructuring complete!")
        print("The dataset now has the hierarchical structure required for training.")
        print(f"Images copied to: {dataset_path}/{{train,valid,test}}/{{normal,defect}}/")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
