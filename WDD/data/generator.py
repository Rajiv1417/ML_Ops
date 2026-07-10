"""
Synthetic weld defect data generator.

Generates realistic synthetic weld inspection images with various defect types.
Uses procedural generation for reproducibility and portability.
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple
import random
from config import data_generation_config, DATASET_DIR


class WeldDefectGenerator:
    """Generates synthetic weld defect images."""
    
    def __init__(self, seed: int = 42, image_size: int = 224):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
            image_size: Size of generated images (image_size x image_size)
        """
        self.seed = seed
        self.image_size = image_size
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_weld_base(self) -> Image.Image:
        """
        Generate base weld image (normal weld seam).
        
        Returns:
            PIL Image of weld seam
        """
        # Create grayscale base (simulating metal surface)
        base = np.random.randint(100, 150, (self.image_size, self.image_size), dtype=np.uint8)
        
        # Add metal grain texture
        for _ in range(5):
            noise = np.random.normal(0, 10, (self.image_size, self.image_size))
            base = np.clip(base + noise, 0, 255).astype(np.uint8)
        
        # Create weld seam (horizontal line)
        center = self.image_size // 2
        width = 30
        for i in range(center - width // 2, center + width // 2):
            base[i, :] = np.clip(base[i, :] + np.random.randint(30, 60), 0, 255)
        
        img = Image.fromarray(base, mode='L')
        return img.convert('RGB')
    
    def add_crack_defect(self, img: Image.Image) -> Image.Image:
        """Add crack defect to weld image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Crack path
        start_x = random.randint(50, self.image_size - 50)
        start_y = random.randint(80, self.image_size - 80)
        
        points = [(start_x, start_y)]
        x, y = start_x, start_y
        for _ in range(15):
            x += random.randint(-5, 10)
            y += random.randint(-5, 5)
            x = max(10, min(self.image_size - 10, x))
            y = max(50, min(self.image_size - 50, y))
            points.append((x, y))
        
        # Draw crack (dark line with slight fuzz)
        draw.line(points, fill=(50, 50, 50, 200), width=2)
        
        # Add some noise around crack
        for _ in range(20):
            px = random.randint(start_x - 10, start_x + 50)
            py = random.randint(start_y - 5, start_y + 15)
            draw.point((px, py), fill=(30, 30, 30, 150))
        
        return img
    
    def add_porosity_defect(self, img: Image.Image) -> Image.Image:
        """Add porosity (small holes) defect to weld image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Center area
        cx = random.randint(50, self.image_size - 50)
        cy = random.randint(100, self.image_size - 100)
        
        # Generate multiple pores
        num_pores = random.randint(5, 15)
        for _ in range(num_pores):
            x = cx + random.randint(-40, 40)
            y = cy + random.randint(-20, 20)
            radius = random.randint(2, 6)
            
            # Draw pore (dark circle)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(20, 20, 20, 200)
            )
            # Add slight shadow
            draw.ellipse(
                [x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1],
                outline=(40, 40, 40, 100)
            )
        
        return img
    
    def add_undercut_defect(self, img: Image.Image) -> Image.Image:
        """Add undercut defect to weld image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Undercut: deep groove along edge
        start_x = random.randint(30, 100) if random.random() > 0.5 else random.randint(self.image_size - 100, self.image_size - 30)
        
        # Draw groove
        for i in range(self.image_size):
            depth = int(20 * np.sin(i / self.image_size * np.pi) ** 2) + 5
            for d in range(depth):
                intensity = int(50 - (d * 3))
                draw.point((start_x + d, i), fill=(intensity, intensity, intensity, 200))
        
        return img
    
    def add_spatter_defect(self, img: Image.Image) -> Image.Image:
        """Add spatter defect to weld image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Spatter: scattered metal bits
        num_splatters = random.randint(10, 30)
        for _ in range(num_splatters):
            x = random.randint(20, self.image_size - 20)
            y = random.randint(20, self.image_size - 20)
            size = random.randint(1, 4)
            
            # Draw spatter blob
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=(100 + random.randint(-20, 20), 100 + random.randint(-20, 20), 100 + random.randint(-20, 20), 180)
            )
        
        return img
    
    def add_incomplete_fusion_defect(self, img: Image.Image) -> Image.Image:
        """Add incomplete fusion defect to weld image."""
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Incomplete fusion: unfused area in weld
        cx = random.randint(50, self.image_size - 50)
        cy = random.randint(80, self.image_size - 80)
        
        # Draw unfused region
        width = random.randint(20, 40)
        height = random.randint(10, 20)
        
        for i in range(height):
            alpha = int(200 * (1 - i / height))
            draw.line(
                [(cx - width // 2, cy + i), (cx + width // 2, cy + i)],
                fill=(40, 40, 40, alpha),
                width=2
            )
        
        return img
    
    def generate_defect_image(self, defect_type: str) -> Image.Image:
        """
        Generate a defect image of specified type.
        
        Args:
            defect_type: One of 'crack', 'porosity', 'undercut', 'spatter', 'incomplete_fusion'
        
        Returns:
            PIL Image with defect
        """
        img = self.generate_weld_base()
        
        if defect_type == "crack":
            img = self.add_crack_defect(img)
        elif defect_type == "porosity":
            img = self.add_porosity_defect(img)
        elif defect_type == "undercut":
            img = self.add_undercut_defect(img)
        elif defect_type == "spatter":
            img = self.add_spatter_defect(img)
        elif defect_type == "incomplete_fusion":
            img = self.add_incomplete_fusion_defect(img)
        
        # Apply slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def generate_normal_image(self) -> Image.Image:
        """Generate a normal (no-defect) weld image."""
        return self.generate_weld_base()


def generate_dataset(num_defects: int = 400, num_normal: int = 400, force_regenerate: bool = False) -> Path:
    """
    Generate complete weld defect dataset.
    
    Args:
        num_defects: Number of defect images to generate
        num_normal: Number of normal images to generate
        force_regenerate: If True, regenerate even if dataset exists
    
    Returns:
        Path to dataset directory
    """
    # Check if dataset already exists
    dataset_dir = DATASET_DIR
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"
    
    if (train_dir.exists() and not force_regenerate):
        print(f"Dataset already exists at {dataset_dir}")
        return dataset_dir
    
    print("\n" + "="*80)
    print("SYNTHETIC DATASET GENERATION")
    print("="*80)
    print(f"Generating synthetic weld defect dataset...")
    print(f"Target images:")
    print(f"  - Defective welds: {num_defects}")
    print(f"  - Normal welds: {num_normal}")
    print(f"  - Total: {num_defects + num_normal} images")
    print("="*80 + "\n")
    
    # Create directory structure
    print("Step 1: Creating directory structure...")
    for split in [train_dir, val_dir, test_dir]:
        (split / "normal").mkdir(parents=True, exist_ok=True)
        (split / "defect").mkdir(parents=True, exist_ok=True)
    print("  Created train/val/test folders with normal/defect subfolders\n")
    
    generator = WeldDefectGenerator(seed=data_generation_config.seed, image_size=224)
    
    # Calculate split sizes
    total_defects = num_defects
    total_normal = num_normal
    
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    
    train_defects = int(total_defects * train_split)
    val_defects = int(total_defects * val_split)
    test_defects = total_defects - train_defects - val_defects
    
    train_normal = int(total_normal * train_split)
    val_normal = int(total_normal * val_split)
    test_normal = total_normal - train_normal - val_normal
    
    print("Step 2: Data split configuration")
    print(f"  Training (70%):   {train_defects} defects + {train_normal} normal = {train_defects + train_normal} total")
    print(f"  Validation (15%): {val_defects} defects + {val_normal} normal = {val_defects + val_normal} total")
    print(f"  Test (15%):       {test_defects} defects + {test_normal} normal = {test_defects + test_normal} total\n")
    
    # Generate defect images
    print("Step 3: Generating defective weld images...")
    defect_types = data_generation_config.defect_types
    print(f"  Using {len(defect_types)} defect types: {', '.join(defect_types)}\n")
    
    img_idx = 0
    split_info = [
        ("Training", train_dir, train_defects),
        ("Validation", val_dir, val_defects),
        ("Test", test_dir, test_defects)
    ]
    
    for split_name, split_dir, count in split_info:
        print(f"  Generating {count} defective images for {split_name} set...")
        for i in range(count):
            defect_type = defect_types[img_idx % len(defect_types)]
            img = generator.generate_defect_image(defect_type)
            img.save(split_dir / "defect" / f"defect_{img_idx:04d}.png")
            img_idx += 1
            
            if (i + 1) % max(1, count // 3) == 0 or (i + 1) == count:
                progress_pct = ((i + 1) / count) * 100
                print(f"    {split_name}: {i + 1}/{count} ({progress_pct:.0f}%)")
    
    print(f"  Total defective images created: {img_idx}\n")
    
    # Generate normal images
    print("Step 4: Generating normal weld images...")
    img_idx = 0
    
    for split_name, split_dir, count in split_info:
        print(f"  Generating {count} normal images for {split_name} set...")
        for i in range(count):
            img = generator.generate_normal_image()
            img.save(split_dir / "normal" / f"normal_{img_idx:04d}.png")
            img_idx += 1
            
            if (i + 1) % max(1, count // 3) == 0 or (i + 1) == count:
                progress_pct = ((i + 1) / count) * 100
                print(f"    {split_name}: {i + 1}/{count} ({progress_pct:.0f}%)")
    
    print(f"  Total normal images created: {img_idx}\n")
    
    print("="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Total images generated: {total_defects + total_normal}")
    print(f"Training set:   {train_defects + train_normal} images")
    print(f"Validation set: {val_defects + val_normal} images")
    print(f"Test set:       {test_defects + test_normal} images")
    print(f"Dataset location: {dataset_dir}")
    print("="*80 + "\n")
    
    return dataset_dir


if __name__ == "__main__":
    # Generate dataset when run directly
    generate_dataset()
