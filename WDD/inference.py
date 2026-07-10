"""
Phase 3: Inference Engine with Out-Of-Distribution Detection

Provides inference capabilities with:
- Single image classification (weld vs non-weld, defect vs normal)
- Out-of-distribution (OOD) detection to identify non-welding images
- Confidence scoring and uncertainty quantification
- Batch inference support
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import json

from config import training_config, CHECKPOINTS_DIR
from trainer.dataset import get_data_transforms


@dataclass
class InferencePrediction:
    """Result of inference on a single image."""
    
    # Main classification
    is_defect: bool
    predicted_class: str  # "normal" or "defect"
    confidence: float  # Probability [0, 1]
    
    # Out-of-distribution detection
    is_welding_image: bool  # True if image appears to be a weld
    ood_score: float  # OOD confidence [0, 1] (higher = more OOD)
    ood_reason: str  # Explanation if OOD detected
    
    # Raw model outputs
    normal_prob: float
    defect_prob: float
    
    # Metadata
    image_path: Optional[str] = None
    processing_time_ms: float = 0.0


class OODDetector:
    """
    Out-of-Distribution Detector using multiple heuristics.
    
    Detects if an image is likely NOT a welding image using:
    1. Feature distribution analysis
    2. Entropy-based uncertainty
    3. Edge detection (welds have distinct edges)
    4. Color distribution analysis
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize OOD detector.
        
        Args:
            threshold: OOD threshold [0, 1]. Score > threshold = OOD
        """
        self.threshold = threshold
    
    def compute_features(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Compute features for OOD detection from image tensor.
        
        Args:
            image: Normalized image tensor [3, 224, 224]
        
        Returns:
            Dictionary of feature scores
        """
        # Denormalize for analysis
        unnorm = image.clone()
        unnorm = unnorm * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnorm = unnorm + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        unnorm = torch.clamp(unnorm, 0, 1)
        
        features = {}
        
        # 1. Color saturation (welds are typically grayscale-ish, low saturation)
        rgb = unnorm * 255
        r, g, b = rgb[0], rgb[1], rgb[2]
        max_c = torch.max(torch.max(r, g), b)
        min_c = torch.min(torch.min(r, g), b)
        saturation = (max_c - min_c) / (max_c + 1e-6)
        features['avg_saturation'] = saturation.mean().item()
        
        # 2. Brightness (welds typically have varying brightness)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        features['brightness_std'] = brightness.std().item()
        features['brightness_mean'] = brightness.mean().item()
        
        # 3. Edge density (welds have strong edges)
        # Sobel edge detection on grayscale image
        gray = brightness / 255.0
        gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # gray shape: [224, 224] -> [1, 1, 224, 224]
        gray_expanded = gray.unsqueeze(0).unsqueeze(0)
        
        # Apply Sobel operators (1-channel input, 1-channel kernels)
        edge_x = torch.nn.functional.conv2d(gray_expanded, gx, padding=1)
        edge_y = torch.nn.functional.conv2d(gray_expanded, gy, padding=1)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        
        features['edge_density'] = (edge_mag > 0.1).float().mean().item()
        
        # 4. Texture uniformity (welds have textured surface)
        features['texture_variance'] = gray.var().item()
        
        return features
    
    def compute_ood_score(
        self,
        features: Dict[str, float],
        model_confidence: float,
        entropy: float
    ) -> Tuple[float, str]:
        """
        Compute OOD score from features.
        
        Args:
            features: Feature dictionary from compute_features()
            model_confidence: Model's confidence in prediction [0, 1]
            entropy: Entropy of model output
        
        Returns:
            Tuple of (ood_score, reason_string)
        """
        ood_score = 0.0
        reasons = []
        
        # High saturation is OK - real welded images can be colored (from lighting, heat effects, etc)
        # Only flag extremely high saturation (like a pure color image)
        if features['avg_saturation'] > 0.7:
            ood_score += 0.1
            reasons.append("Very high color saturation (unusual for welds)")
        
        # Very high brightness = likely not a weld
        if features['brightness_mean'] > 0.85:
            ood_score += 0.15
            reasons.append("Very bright image (welds typically darker)")
        
        # Low texture variance = likely not a weld
        if features['texture_variance'] < 0.01:
            ood_score += 0.2
            reasons.append("Very low texture (welds have rough surface texture)")
        
        # Low edge density = likely not a weld (this is the strongest indicator)
        if features['edge_density'] < 0.08:
            ood_score += 0.3
            reasons.append("Low edge density (welds have distinct edges and structure)")
        
        # Low model confidence = uncertain
        if model_confidence < 0.55:
            ood_score += 0.15
            reasons.append("Model uncertain about prediction")
        
        # High entropy = uncertain
        if entropy > 0.55:
            ood_score += 0.1
            reasons.append("High uncertainty in classification")
        
        # Normalize to [0, 1]
        ood_score = min(ood_score, 1.0)
        
        reason_text = "; ".join(reasons) if reasons else "Image appears to be a valid weld"
        
        return ood_score, reason_text


class InferenceEngine:
    """
    Inference engine for weld defect classification with OOD detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        ood_threshold: float = 0.5
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on ('cpu' or 'cuda')
            ood_threshold: OOD detection threshold [0, 1]
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.transforms = get_data_transforms()['val']
        self.ood_detector = OODDetector(threshold=ood_threshold)
        
        # Statistics for normalization (computed from training set)
        self.class_names = {0: 'normal', 1: 'defect'}
    
    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to Path if string
            if isinstance(checkpoint_path, str):
                checkpoint_path = Path(checkpoint_path)
            
            print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
            print(f"[INFO] File exists: {checkpoint_path.exists()}")
            
            if not checkpoint_path.exists():
                print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"[INFO] Checkpoint loaded, keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'tensor'}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                # Try different possible state dict keys
                if 'model_state_dict' in checkpoint:
                    print("[INFO] Loading model_state_dict")
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model_state' in checkpoint:
                    print("[INFO] Loading model_state")
                    self.model.load_state_dict(checkpoint['model_state'])
                else:
                    # Assume the entire dict is the state dict
                    print(f"[INFO] Loading entire dict as state dict")
                    self.model.load_state_dict(checkpoint)
            else:
                print("[INFO] Checkpoint is tensor, setting as model")
                self.model = checkpoint
            
            self.model.to(self.device)
            self.model.eval()
            print("[INFO] Model loaded and set to eval mode")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading checkpoint: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_single(
        self,
        image_path: str,
        return_raw_scores: bool = False
    ) -> InferencePrediction:
        """
        Predict on a single image.
        
        Args:
            image_path: Path to image file
            return_raw_scores: Whether to return raw logits
        
        Returns:
            InferencePrediction object with all details
        """
        import time
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions
            pred_class = torch.argmax(probabilities[0]).item()
            class_name = self.class_names[pred_class]
            confidence = probabilities[0, pred_class].item()
            
            normal_prob = probabilities[0, 0].item()
            defect_prob = probabilities[0, 1].item()
            
            # Compute entropy for uncertainty
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)[0].item()
            
            # OOD detection
            features = self.ood_detector.compute_features(image_tensor[0])
            ood_score, ood_reason = self.ood_detector.compute_ood_score(
                features,
                confidence,
                entropy
            )
            is_ood = ood_score > self.ood_detector.threshold
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return InferencePrediction(
                is_defect=(pred_class == 1),
                predicted_class=class_name,
                confidence=confidence,
                is_welding_image=not is_ood,
                ood_score=ood_score,
                ood_reason=ood_reason,
                normal_prob=normal_prob,
                defect_prob=defect_prob,
                image_path=str(image_path),
                processing_time_ms=processing_time
            )
        
        except Exception as e:
            # Return error prediction
            return InferencePrediction(
                is_defect=False,
                predicted_class="error",
                confidence=0.0,
                is_welding_image=False,
                ood_score=1.0,
                ood_reason=f"Error processing image: {str(e)}",
                normal_prob=0.0,
                defect_prob=0.0,
                image_path=str(image_path),
                processing_time_ms=0.0
            )
    
    def predict_batch(
        self,
        image_paths: List[str]
    ) -> List[InferencePrediction]:
        """
        Predict on multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of InferencePrediction objects
        """
        predictions = []
        for image_path in image_paths:
            pred = self.predict_single(image_path)
            predictions.append(pred)
        return predictions
    
    def get_prediction_summary(self, predictions: List[InferencePrediction]) -> Dict:
        """
        Get summary statistics from predictions.
        
        Args:
            predictions: List of predictions
        
        Returns:
            Dictionary with summary statistics
        """
        if not predictions:
            return {}
        
        defect_count = sum(1 for p in predictions if p.is_defect)
        normal_count = len(predictions) - defect_count
        
        ood_count = sum(1 for p in predictions if not p.is_welding_image)
        
        avg_confidence = np.mean([p.confidence for p in predictions])
        avg_ood_score = np.mean([p.ood_score for p in predictions])
        
        return {
            'total_images': len(predictions),
            'defect_count': defect_count,
            'normal_count': normal_count,
            'defect_ratio': defect_count / len(predictions),
            'ood_count': ood_count,
            'ood_ratio': ood_count / len(predictions),
            'avg_confidence': avg_confidence,
            'avg_ood_score': avg_ood_score,
        }
