"""
Data source manager for external and synthetic datasets.

Handles discovery and loading of available datasets from the data/ directory.
Supports both synthetic generated data and external open-source datasets.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from config import DATA_DIR, DATASET_DIR


class DataSourceManager:
    """Manages available data sources (synthetic and external)."""
    
    SYNTHETIC_NAME = "[SYNTHETIC] Generated Weld Defects"
    REQUIRED_SPLITS = {'train', 'valid', 'test'}  # or 'val' for some
    
    @staticmethod
    def is_hierarchical_dataset(dataset_path: Path) -> bool:
        """Check if dataset has required hierarchical structure (normal/defect subdirs)."""
        train_dir = dataset_path / "train"
        if not train_dir.exists():
            return False
        has_normal = (train_dir / "normal").exists()
        has_defect = (train_dir / "defect").exists()
        return has_normal or has_defect
    
    @staticmethod
    def is_flat_dataset(dataset_path: Path) -> bool:
        """Check if dataset has flat structure (images directly in splits, no class subdirs)."""
        train_dir = dataset_path / "train"
        if not train_dir.exists():
            return False
        # If there are image files directly in train/, it's flat
        for file_type in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            if list(train_dir.glob(file_type)):
                # Found images at root level - check that normal/defect don't exist
                if not (train_dir / "normal").exists() and not (train_dir / "defect").exists():
                    return True
        return False
    
    @staticmethod
    def discover_external_datasets() -> List[Tuple[str, Path]]:
        """
        Discover external datasets in data/ directory.
        
        An external dataset must have hierarchical structure:
        - data/DatasetName/
          - train/normal/ and train/defect/
          - valid/normal/ and valid/defect/ (or val/)
          - test/normal/ and test/defect/
        
        Returns:
            List of tuples: [(display_name, dataset_path), ...]
            Sorted by name, excluding synthetic dataset folder
        """
        datasets = []
        
        if not DATA_DIR.exists():
            return datasets
        
        for item in sorted(DATA_DIR.iterdir()):
            if not item.is_dir():
                continue
            
            # Skip synthetic dataset folder
            if item.name == "weld_defects":
                continue
            
            # Skip Python files/cache
            if item.name.startswith("__") or item.name.endswith(".py"):
                continue
            
            # Check if folder has required splits
            has_train = (item / "train").exists()
            has_valid = (item / "valid").exists() or (item / "val").exists()
            has_test = (item / "test").exists()
            
            if has_train and has_valid and has_test:
                # IMPORTANT: Only include if hierarchical (has normal/defect subdirs)
                if DataSourceManager.is_hierarchical_dataset(item):
                    display_name = f"[EXTERNAL] {item.name}"
                    datasets.append((display_name, item))
        
        return datasets
    
    @staticmethod
    def get_all_data_sources() -> List[Tuple[str, Optional[Path]]]:
        """
        Get all available data sources (synthetic + external).
        
        Returns:
            List of tuples: [(display_name, dataset_path), ...]
            - First entry is always synthetic (path=None)
            - Then external datasets in alphabetical order
        """
        sources = []
        
        # Always offer synthetic as first option
        sources.append((DataSourceManager.SYNTHETIC_NAME, None))
        
        # Add external datasets
        external = DataSourceManager.discover_external_datasets()
        sources.extend(external)
        
        return sources
    
    @staticmethod
    def get_dataset_display_names() -> List[str]:
        """Get just the display names for UI selectbox."""
        return [name for name, _ in DataSourceManager.get_all_data_sources()]
    
    @staticmethod
    def get_dataset_path_by_name(display_name: str) -> Optional[Path]:
        """
        Get dataset path from display name.
        
        Args:
            display_name: The display name from get_all_data_sources()
        
        Returns:
            Path to dataset directory, or None for synthetic
        """
        sources = DataSourceManager.get_all_data_sources()
        for name, path in sources:
            if name == display_name:
                return path
        return None
    
    @staticmethod
    def get_synthetic_dataset_path() -> Path:
        """Get path to synthetic dataset (weld_defects)."""
        return DATASET_DIR
    
    @staticmethod
    def validate_dataset(dataset_path: Path) -> Tuple[bool, str]:
        """
        Validate that a dataset has required hierarchical structure.
        
        Args:
            dataset_path: Path to dataset
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not dataset_path.exists():
            return False, "Dataset path does not exist"
        
        # Check for required splits
        valid_split = (dataset_path / "valid").exists() or (dataset_path / "val").exists()
        
        if not (dataset_path / "train").exists():
            return False, "Missing 'train' directory"
        
        if not valid_split:
            return False, "Missing 'valid' or 'val' directory"
        
        if not (dataset_path / "test").exists():
            return False, "Missing 'test' directory"
        
        # Check if dataset is flat (INCOMPATIBLE)
        if DataSourceManager.is_flat_dataset(dataset_path):
            return False, (
                "FLAT STRUCTURE DETECTED: All images are in train/valid/test folders without class subdirectories.\n"
                "REQUIRED: Hierarchical structure with train/normal/, train/defect/, valid/normal/, valid/defect/, test/normal/, test/defect/\n"
                "TO FIX: Use the restructure_flat_dataset.py script in data/ folder"
            )
        
        # Check if hierarchical structure exists
        if not DataSourceManager.is_hierarchical_dataset(dataset_path):
            return False, (
                "MISSING CLASS STRUCTURE: No normal/defect subdirectories found in train/ folder.\n"
                "REQUIRED: Hierarchical structure with train/normal/, train/defect/, valid/normal/, valid/defect/, test/normal/, test/defect/\n"
                "TO FIX: Use the restructure_flat_dataset.py script in data/ folder"
            )
        
        return True, "OK"
    
    @staticmethod
    def get_dataset_info(dataset_path: Path) -> dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_path: Path to dataset
        
        Returns:
            Dictionary with dataset info
        """
        info = {
            'path': str(dataset_path),
            'name': dataset_path.name,
            'exists': dataset_path.exists(),
            'is_valid': False,
            'error': None,
            'has_train': False,
            'has_valid': False,
            'has_test': False
        }
        
        if not dataset_path.exists():
            info['error'] = "Path does not exist"
            return info
        
        info['has_train'] = (dataset_path / "train").exists()
        info['has_valid'] = (dataset_path / "valid").exists() or (dataset_path / "val").exists()
        info['has_test'] = (dataset_path / "test").exists()
        
        is_valid, error = DataSourceManager.validate_dataset(dataset_path)
        info['is_valid'] = is_valid
        info['error'] = error if not is_valid else None
        
        return info
