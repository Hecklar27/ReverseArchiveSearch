"""
Configuration management for Reverse Archive Search application.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VisionConfig:
    """Configuration for computer vision features"""
    enable_map_art_detection: bool = True  # Enabled by default for better accuracy
    detection_method: str = "opencv"  # 'opencv', 'yolo', 'segment', 'hybrid'
    confidence_threshold: float = 0.5  # Minimum confidence for detections
    crop_padding: int = 10  # Extra padding around detected regions
    fallback_to_full_image: bool = True  # Use full image if no map art detected
    
    # NEW: Performance optimization settings
    use_fast_detection: bool = False  # Use faster, less precise detection during cache building
    batch_processing: bool = True  # Enable batch processing for better performance
    cache_detection_results: bool = True  # Cache detection results to avoid reprocessing

@dataclass
class ClipConfig:
    """Configuration for CLIP model"""
    model_name: str = "DINOv2-Base"  # Default to DINOv2 for exact matching
    device: str = "auto"  # auto, cuda, cpu
    batch_size: int = 16  # Will be adjusted based on model
    use_mixed_precision: bool = True  # Enable mixed precision for faster processing
    
    # Model performance metadata
    MODEL_OPTIONS = {
        "ViT-L/14": {
            "display_name": "ViT-L/14 (Accurate, Slow)",
            "description": "High accuracy, slower processing (~2-5 minutes for cache build)",
            "embedding_dim": 768,
            "recommended_batch_size": 8,
            "speed_rating": "Slow",
            "accuracy_rating": "High",
            "model_type": "clip"
        },
        "DINOv2-Base": {
            "display_name": "DINOv2-Base (Exact Matches, Fast)",
            "description": "Structural similarity, exact matches (~1-3min for cache build)",
            "embedding_dim": 768,
            "recommended_batch_size": 16,
            "speed_rating": "Fast",
            "accuracy_rating": "Exact",
            "model_type": "dinov2",
            "similarity_threshold": 0.90  # High threshold for exact matches
        }
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get performance information for a specific model"""
        return cls.MODEL_OPTIONS.get(model_name, {})
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names"""
        return list(cls.MODEL_OPTIONS.keys())
    
    @classmethod
    def get_display_options(cls) -> list:
        """Get list of model display names for GUI dropdown"""
        return [info["display_name"] for info in cls.MODEL_OPTIONS.values()]
    
    @classmethod
    def model_name_from_display(cls, display_name: str) -> str:
        """Convert display name back to model name"""
        for model_name, info in cls.MODEL_OPTIONS.items():
            if info["display_name"] == display_name:
                return model_name
        return "DINOv2-Base"  # Default fallback updated
    
    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """Get model type (clip or dinov2)"""
        model_info = cls.get_model_info(model_name)
        return model_info.get("model_type", "clip")
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for current model"""
        model_info = self.get_model_info(self.model_name)
        return model_info.get("recommended_batch_size", 16)
    
    def get_cache_subdirectory(self) -> str:
        """Get cache subdirectory name for current model"""
        return self.model_name.replace("/", "-").replace("@", "-")

@dataclass
class UIConfig:
    """Configuration for user interface"""
    window_title: str = "Reverse Archive Search"
    window_size: str = "900x800"
    max_results: int = 20

@dataclass
class DiscordConfig:
    """Configuration for Discord integration"""
    server_id: str = "349201680023289867"
    channel_id: str = "349277718954901514"
    
    def get_message_url(self, message_id: str) -> str:
        """Generate Discord message URL"""
        return f"https://discord.com/channels/{self.server_id}/{self.channel_id}/{message_id}"

@dataclass
class CacheConfig:
    """Configuration for embedding cache system (Phase 2)"""
    enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    embeddings_file: str = "embeddings.pkl"
    metadata_file: str = "metadata.pkl"
    index_file: str = "urls_index.pkl"
    max_age_days: int = 30  # Cache expiry
    batch_size: int = 32    # Default batch size, will be overridden by model-specific settings
    compression: bool = True  # Compress cache files
    
    def get_model_cache_dir(self, model_name: str) -> Path:
        """Get cache directory for specific model"""
        model_subdir = model_name.replace("/", "-").replace("@", "-")
        return self.cache_dir / model_subdir
    
    def get_embeddings_path(self, model_name: str = None) -> Path:
        """Get path to embeddings cache file for specific model"""
        if model_name:
            return self.get_model_cache_dir(model_name) / self.embeddings_file
        return self.cache_dir / self.embeddings_file  # Backward compatibility
    
    def get_metadata_path(self, model_name: str = None) -> Path:
        """Get path to metadata cache file for specific model"""
        if model_name:
            return self.get_model_cache_dir(model_name) / self.metadata_file
        return self.cache_dir / self.metadata_file  # Backward compatibility
    
    def get_index_path(self, model_name: str = None) -> Path:
        """Get path to URL index cache file for specific model"""
        if model_name:
            return self.get_model_cache_dir(model_name) / self.index_file
        return self.cache_dir / self.index_file  # Backward compatibility

@dataclass
class FeatureConfig:
    """Configuration for CLIP feature extraction engine"""
    pass  # All CLIP settings are in ClipConfig

@dataclass
class Config:
    """Main application configuration"""
    clip: ClipConfig = field(default_factory=ClipConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Data sources
    discord_archive_path: str = "mapart-archive.html"  # Changed from JSON to HTML
    
    @classmethod
    def load_default(cls) -> 'Config':
        """Load default configuration"""
        config = cls()
        
        # Ensure cache directory exists
        config.cache.cache_dir.mkdir(parents=True, exist_ok=True)
        
        return config
    
    def is_supported_image(self, file_path: Path) -> bool:
        """Check if file is a supported image format"""
        supported_formats = (".png", ".jpg", ".jpeg")
        return file_path.suffix.lower() in supported_formats

def setup_logging(level: str = "INFO") -> None:
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('reverse_archive_search.log')
        ]
    ) 

# Data sources
discord_archive_path: str = "mapart-archive.html"  # Changed from JSON to HTML
cache_dir: str = "cache" 