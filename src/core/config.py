"""
Configuration management for Reverse Archive Search application.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ClipConfig:
    """Configuration for CLIP model"""
    model_name: str = "ViT-B/32"
    device: str = "auto"  # auto, cuda, cpu
    batch_size: int = 32

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
    batch_size: int = 32    # Batch size for pre-processing
    compression: bool = True  # Compress cache files
    
    def get_embeddings_path(self) -> Path:
        """Get path to embeddings cache file"""
        return self.cache_dir / self.embeddings_file
    
    def get_metadata_path(self) -> Path:
        """Get path to metadata cache file"""
        return self.cache_dir / self.metadata_file
    
    def get_index_path(self) -> Path:
        """Get path to URL index cache file"""
        return self.cache_dir / self.index_file

@dataclass
class Config:
    """Main application configuration"""
    clip: ClipConfig = field(default_factory=ClipConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
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