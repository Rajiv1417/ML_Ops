"""
Deployment profiles for weld defect detection.

Defines deployment constraints and scenarios for different weld inspection environments.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DeploymentProfile:
    """Deployment constraints and scenarios for weld inspection."""
    
    id: str
    name: str
    description: str
    use_case: str
    
    # Hard constraints (must meet)
    max_model_size_mb: float
    max_latency_ms: float
    min_accuracy: float
    
    # Soft constraints (nice to have)
    preferred_throughput_fps: float
    preferred_accuracy: float
    
    # Context
    hardware_description: str
    deployment_location: str


class ProfileManager:
    """Manages deployment profiles for different weld inspection scenarios."""
    
    PROFILES = [
        DeploymentProfile(
            id="weld_vision",
            name="[WELD] Weld Vision",
            description="Real-time weld seam inspection on assembly line",
            use_case="Inline quality control during welding process",
            
            max_model_size_mb=20.0,
            max_latency_ms=100.0,
            min_accuracy=0.92,
            
            preferred_throughput_fps=10.0,
            preferred_accuracy=0.96,
            
            hardware_description="Edge device mounted on robotic arm (Jetson Nano-class)",
            deployment_location="Manufacturing floor (harsh environment)"
        ),
        
        DeploymentProfile(
            id="battery_inspection",
            name="[BATTERY] Battery Inspection",
            description="Battery cell defect detection in quality control",
            use_case="Detect manufacturing defects in battery cells",
            
            max_model_size_mb=50.0,
            max_latency_ms=200.0,
            min_accuracy=0.95,
            
            preferred_throughput_fps=5.0,
            preferred_accuracy=0.98,
            
            hardware_description="Quality control station with moderate compute (Raspberry Pi 4 or better)",
            deployment_location="Clean room / QC lab"
        ),
        
        DeploymentProfile(
            id="paint_inspection",
            name="[PAINT] Paint Inspection",
            description="Paint defect detection for automotive bodies",
            use_case="Detect paint surface defects before final assembly",
            
            max_model_size_mb=100.0,
            max_latency_ms=500.0,
            min_accuracy=0.90,
            
            preferred_throughput_fps=2.0,
            preferred_accuracy=0.95,
            
            hardware_description="Gateway device with moderate resources (Intel Atom / ARM64)",
            deployment_location="Paint shop (high humidity, temperature variations)"
        ),
        
        DeploymentProfile(
            id="edge_extreme",
            name="[EDGE] Edge Extreme",
            description="Extreme resource constraints (IoT/embedded device)",
            use_case="Deploy on ultra-low-power devices with minimal memory",
            
            max_model_size_mb=5.0,
            max_latency_ms=500.0,
            min_accuracy=0.85,
            
            preferred_throughput_fps=1.0,
            preferred_accuracy=0.90,
            
            hardware_description="Microcontroller or ultra-low power device (STM32, nRF, etc.)",
            deployment_location="Remote sensor nodes"
        ),
    ]
    
    @classmethod
    def get_profiles(cls) -> List[DeploymentProfile]:
        """Get all available profiles."""
        return cls.PROFILES
    
    @classmethod
    def get_profile_by_id(cls, profile_id: str) -> Optional[DeploymentProfile]:
        """Get profile by ID."""
        for profile in cls.PROFILES:
            if profile.id == profile_id:
                return profile
        return None
    
    @classmethod
    def get_profile_by_name(cls, name: str) -> Optional[DeploymentProfile]:
        """Get profile by name."""
        for profile in cls.PROFILES:
            if profile.name == name:
                return profile
        return None
    
    @classmethod
    def get_profile_names(cls) -> List[str]:
        """Get list of profile names."""
        return [p.name for p in cls.PROFILES]


def get_all_profiles() -> List[DeploymentProfile]:
    """Convenience function to get all profiles."""
    return ProfileManager.get_profiles()


def get_profile_by_id(profile_id: str) -> Optional[DeploymentProfile]:
    """Convenience function to get profile by ID."""
    return ProfileManager.get_profile_by_id(profile_id)


def get_profile_by_name(name: str) -> Optional[DeploymentProfile]:
    """Convenience function to get profile by name."""
    return ProfileManager.get_profile_by_name(name)
