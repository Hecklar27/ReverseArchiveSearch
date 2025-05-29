"""
Cache management system for pre-computed embeddings (Phase 2).
Handles embedding storage, metadata, and cache validation.
"""

import logging
import pickle
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta

from ..data.models import DiscordMessage, DiscordAttachment
from ..core.config import Config
from .clip_engine import CLIPEngine
from .image_downloader import ImageDownloader

logger = logging.getLogger(__name__)

class CacheMetadata:
    """Metadata for cached embeddings"""
    
    def __init__(self):
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.total_images = 0
        self.cache_version = "2.0"  # Updated for Phase 2 with message_type field
        self.clip_model = None
        self.url_to_index = {}  # URL -> index mapping
        self.index_to_metadata = {}  # index -> {message_id, attachment info, discord_url}
        
    def is_expired(self, max_age_days: int) -> bool:
        """Check if cache is expired"""
        expiry_date = self.created_at + timedelta(days=max_age_days)
        return datetime.now() > expiry_date
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'total_images': self.total_images,
            'cache_version': self.cache_version,
            'clip_model': self.clip_model,
            'age_days': (datetime.now() - self.created_at).days
        }

class EmbeddingCacheManager:
    """Manages pre-computed embeddings for Discord images"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_config = config.cache
        self.clip_engine = CLIPEngine(config.clip)
        self.image_downloader = ImageDownloader(max_workers=8)
        
        # Ensure cache directory exists
        self.cache_config.cache_dir.mkdir(exist_ok=True)
        
        self._embeddings = None
        self._metadata = None
        
    def has_valid_cache(self) -> bool:
        """Check if valid cache exists"""
        try:
            embeddings_path = self.cache_config.get_embeddings_path()
            metadata_path = self.cache_config.get_metadata_path()
            
            if not (embeddings_path.exists() and metadata_path.exists()):
                return False
            
            # Load and validate metadata
            metadata = self._load_metadata()
            if metadata is None:
                return False
            
            # Check if cache is expired
            if metadata.is_expired(self.cache_config.max_age_days):
                logger.info("Cache expired, needs regeneration")
                return False
            
            # Validate CLIP model compatibility
            current_model = self.clip_engine.model_name
            if metadata.clip_model != current_model:
                logger.info(f"CLIP model changed ({metadata.clip_model} -> {current_model}), cache invalid")
                return False
            
            # Validate cache version and required fields
            if not self._validate_cache_compatibility(metadata):
                logger.info("Cache format is incompatible, needs rebuilding")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating cache: {e}")
            return False
    
    def _load_metadata(self) -> Optional[CacheMetadata]:
        """Load cache metadata"""
        try:
            metadata_path = self.cache_config.get_metadata_path()
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None
    
    def _save_metadata(self, metadata: CacheMetadata) -> None:
        """Save cache metadata"""
        try:
            metadata_path = self.cache_config.get_metadata_path()
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            raise
    
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load cached embeddings"""
        try:
            embeddings_path = self.cache_config.get_embeddings_path()
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return None
    
    def _save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings to cache"""
        try:
            embeddings_path = self.cache_config.get_embeddings_path()
            with open(embeddings_path, 'wb') as f:
                if self.cache_config.compression:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    def build_cache(self, discord_messages: List[DiscordMessage], 
                   progress_callback: Optional[callable] = None) -> bool:
        """
        Build embedding cache from Discord messages.
        
        Args:
            discord_messages: List of Discord messages to process
            progress_callback: Optional progress callback (current, total, status)
            
        Returns:
            True if cache was built successfully
        """
        logger.info(f"Building embedding cache for {len(discord_messages)} messages")
        start_time = time.time()
        
        # Collect all image attachments
        image_info = []  # (message, attachment) pairs
        for message in discord_messages:
            if message.has_images():
                for attachment in message.get_image_attachments():
                    image_info.append((message, attachment))
        
        total_images = len(image_info)
        if total_images == 0:
            logger.warning("No images found in Discord messages")
            return False
        
        logger.info(f"Found {total_images} images to process")
        
        # Initialize metadata
        metadata = CacheMetadata()
        metadata.total_images = total_images
        metadata.clip_model = self.clip_engine.model_name
        
        # Process images in batches
        batch_size = self.cache_config.batch_size
        all_embeddings = []
        processed_count = 0
        
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_info = image_info[batch_start:batch_end]
            
            if progress_callback:
                progress_callback(processed_count, total_images, 
                                f"Processing batch {batch_start//batch_size + 1}...")
            
            # Download images for this batch
            batch_urls = [attachment.url for _, attachment in batch_info]
            download_results = self.image_downloader.download_images_batch(batch_urls)
            
            # Process successful downloads
            batch_images = []
            valid_batch_info = []
            
            for i, (url, image) in enumerate(download_results):
                if image is not None:
                    batch_images.append(image)
                    valid_batch_info.append(batch_info[i])
            
            if batch_images:
                # Encode images with CLIP
                try:
                    batch_embeddings = self.clip_engine.encode_images_batch(batch_images)
                    
                    # Store embeddings and metadata
                    for i, ((message, attachment), embedding) in enumerate(zip(valid_batch_info, batch_embeddings)):
                        # Generate Discord URL
                        discord_url = self.config.discord.get_message_url(message.id)
                        
                        # Store in metadata
                        index = len(all_embeddings)
                        metadata.url_to_index[attachment.url] = index
                        metadata.index_to_metadata[index] = {
                            'message_id': message.id,
                            'message_type': message.type,
                            'filename': attachment.filename,
                            'url': attachment.url,
                            'discord_url': discord_url,
                            'message_content': message.content,
                            'author': message.author,
                            'timestamp': message.timestamp,
                            'attachment_id': attachment.id,
                            'file_size_bytes': attachment.file_size_bytes
                        }
                        
                        all_embeddings.append(embedding)
                        processed_count += 1
                        
                        if progress_callback:
                            progress_callback(processed_count, total_images, 
                                            f"Cached {attachment.filename}")
                
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    continue
        
        if not all_embeddings:
            logger.error("No embeddings were generated")
            return False
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        if progress_callback:
            progress_callback(total_images, total_images, "Saving cache...")
        
        # Save to cache
        try:
            self._save_embeddings(embeddings_array)
            self._save_metadata(metadata)
            
            # Cache in memory for immediate use
            self._embeddings = embeddings_array
            self._metadata = metadata
            
            build_time = time.time() - start_time
            logger.info(f"Cache built successfully in {build_time:.2f}s")
            logger.info(f"Cached {len(all_embeddings)} embeddings from {total_images} images")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def load_cache(self) -> bool:
        """Load cache into memory"""
        try:
            if not self.has_valid_cache():
                return False
            
            self._embeddings = self._load_embeddings()
            self._metadata = self._load_metadata()
            
            if self._embeddings is None or self._metadata is None:
                return False
            
            logger.info(f"Loaded cache with {len(self._embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False
    
    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get cached embeddings"""
        if self._embeddings is None:
            self.load_cache()
        return self._embeddings
    
    def get_metadata(self) -> Optional[CacheMetadata]:
        """Get cache metadata"""
        if self._metadata is None:
            self.load_cache()
        return self._metadata
    
    def clear_cache(self) -> None:
        """Clear cache files and memory"""
        try:
            embeddings_path = self.cache_config.get_embeddings_path()
            metadata_path = self.cache_config.get_metadata_path()
            
            if embeddings_path.exists():
                embeddings_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            self._embeddings = None
            self._metadata = None
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.has_valid_cache():
            return {'status': 'no_cache'}
        
        metadata = self.get_metadata()
        if metadata:
            stats = metadata.get_stats()
            stats['status'] = 'valid'
            
            # Add file sizes
            try:
                embeddings_path = self.cache_config.get_embeddings_path()
                metadata_path = self.cache_config.get_metadata_path()
                
                if embeddings_path.exists():
                    stats['embeddings_size_mb'] = embeddings_path.stat().st_size / (1024 * 1024)
                if metadata_path.exists():
                    stats['metadata_size_mb'] = metadata_path.stat().st_size / (1024 * 1024)
                    
            except Exception as e:
                logger.warning(f"Failed to get cache file sizes: {e}")
            
            return stats
        
        return {'status': 'invalid'}

    def _validate_cache_compatibility(self, metadata: CacheMetadata) -> bool:
        """Validate cache compatibility and required fields"""
        try:
            # Check cache version
            if not hasattr(metadata, 'cache_version') or metadata.cache_version != "2.0":
                logger.info(f"Cache version mismatch: expected 2.0, got {getattr(metadata, 'cache_version', 'unknown')}")
                return False
            
            # Check if metadata has the required index_to_metadata structure
            if not hasattr(metadata, 'index_to_metadata') or not metadata.index_to_metadata:
                logger.info("Cache missing index_to_metadata")
                return False
            
            # Check if first entry has required fields (sample validation)
            if metadata.index_to_metadata:
                first_key = next(iter(metadata.index_to_metadata))
                first_entry = metadata.index_to_metadata[first_key]
                required_fields = ['message_id', 'message_type', 'filename', 'url', 'discord_url']
                
                for field in required_fields:
                    if field not in first_entry:
                        logger.info(f"Cache missing required field: {field}")
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating cache compatibility: {e}")
            return False
    