"""
Configuration management for Reverse Archive Search application.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class DiscordConfig:
    """Discord-specific configuration"""
    server_id: str = "349201680023289867"  # Map Artists of 2b2t
    channel_id: str = "349277718954901514"  # mapart-archive channel
    base_url: str = "https://discord.com/channels"
    
    def get_message_url(self, message_id: str) -> str:
        """Generate Discord deep link for a message"""
        return f"{self.base_url}/{self.server_id}/{self.channel_id}/{message_id}"

@dataclass
class CLIPConfig:
    """CLIP model configuration"""
    model_name: str = "ViT-B/32"  # Default CLIP model
    device: Optional[str] = None  # Auto-detect GPU/CPU (prefers CUDA if available)
    batch_size: int = 32  # For future batch processing
    prefer_cuda: bool = True  # Prefer CUDA over CPU when available

@dataclass
class CacheConfig:
    """Caching configuration (Phase 2)"""
    cache_dir: Path = Path("cache")
    embeddings_file: str = "embeddings.pkl"
    metadata_file: str = "metadata.db"
    max_cache_size_mb: int = 1000  # Maximum cache size

@dataclass
class UIConfig:
    """User interface configuration"""
    window_title: str = "Reverse Archive Search"
    window_size: str = "1200x800"
    max_results: int = 20  # Maximum search results to display
    supported_formats: tuple = (".png", ".jpg", ".jpeg")

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.discord = DiscordConfig()
        self.clip = CLIPConfig()
        self.cache = CacheConfig()
        self.ui = UIConfig()
        
        # Create cache directory if it doesn't exist
        self.cache.cache_dir.mkdir(exist_ok=True)
    
    def get_supported_image_extensions(self) -> tuple:
        """Get supported image file extensions"""
        return self.ui.supported_formats
    
    def is_supported_image(self, file_path: Path) -> bool:
        """Check if file is a supported image format"""
        return file_path.suffix.lower() in self.ui.supported_formats 