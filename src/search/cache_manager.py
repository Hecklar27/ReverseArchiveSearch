"""
Cache management system for pre-computed embeddings (Phase 2).
Handles embedding storage, metadata, and cache validation.
"""

import logging
import pickle
import time
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta

from data.models import DiscordMessage, DiscordAttachment, ProcessingStats
from core.config import Config
from .clip_engine import CLIPEngine
from .image_downloader import ImageDownloader

logger = logging.getLogger(__name__)

class ParsedMessagesMetadata:
    """Metadata for parsed Discord messages"""
    
    def __init__(self, html_file_path: str, html_file_mtime: float, messages_count: int):
        self.html_file_path = html_file_path
        self.html_file_mtime = html_file_mtime
        self.messages_count = messages_count
        self.parsed_at = datetime.now()
        self.version = "1.0"

class CacheMetadata:
    """Metadata for cached embeddings"""
    
    def __init__(self):
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.total_images = 0
        self.cache_version = "2.1"  # Updated for PyTorch version tracking
        self.clip_model = None
        self.pytorch_version = torch.__version__
        self.index_to_metadata: Dict[int, Dict[str, Any]] = {}
        
    def add_image_metadata(self, index: int, message: DiscordMessage, attachment: DiscordAttachment):
        """Add metadata for an image at the given index"""
        self.index_to_metadata[index] = {
            'message_id': message.id,
            'attachment_filename': attachment.filename,
            'attachment_url': attachment.url,
            'author_name': message.author.name,
            'timestamp': message.timestamp,
            'content': message.content[:100] if message.content else ""  # First 100 chars
        }
    
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
            'pytorch_version': getattr(self, 'pytorch_version', 'unknown'),
            'age_days': (datetime.now() - self.created_at).days
        }

class EmbeddingCacheManager:
    """Manages embedding cache for Discord messages"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.cache.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.embeddings_file = self.cache_dir / "embeddings.npy"
        self.metadata_file = self.cache_dir / "metadata.pkl"
        self.parsed_messages_file = self.cache_dir / "parsed_messages.pkl"
        self.parsed_metadata_file = self.cache_dir / "parsed_metadata.pkl"
        
        self.clip_engine = CLIPEngine(config.clip)
        self.image_downloader = ImageDownloader(timeout=10, max_retries=3, max_workers=8)
        
        self._embeddings = None
        self._metadata = None
        
    def has_valid_cache(self) -> bool:
        """Check if valid cache exists"""
        try:
            embeddings_path = self.embeddings_file
            metadata_path = self.metadata_file
            
            if not (embeddings_path.exists() and metadata_path.exists()):
                return False
            
            # Load and validate metadata
            metadata = self._load_metadata()
            if metadata is None:
                return False
            
            # Check if cache is expired
            if metadata.is_expired(self.config.cache.max_age_days):
                logger.info("Cache expired, needs regeneration")
                return False
            
            # Validate CLIP model compatibility
            current_model = self.clip_engine.model_name
            if metadata.clip_model != current_model:
                logger.info(f"CLIP model changed ({metadata.clip_model} -> {current_model}), cache invalid")
                return False
            
            # Validate PyTorch version compatibility
            current_pytorch_version = torch.__version__
            cached_pytorch_version = getattr(metadata, 'pytorch_version', 'unknown')
            if cached_pytorch_version != current_pytorch_version:
                logger.info(f"PyTorch version changed ({cached_pytorch_version} -> {current_pytorch_version}), cache invalid")
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
            metadata_path = self.metadata_file
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return None
    
    def _save_metadata(self, metadata: CacheMetadata) -> None:
        """Save cache metadata"""
        try:
            metadata_path = self.metadata_file
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            raise
    
    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load cached embeddings"""
        try:
            embeddings_path = self.embeddings_file
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return None
    
    def _save_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings to cache"""
        try:
            embeddings_path = self.embeddings_file
            with open(embeddings_path, 'wb') as f:
                if self.config.cache.compression:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    def build_cache(self, discord_messages: List[DiscordMessage], 
                   progress_callback: Optional[callable] = None) -> Tuple[bool, ProcessingStats]:
        """
        Build embedding cache from Discord messages.
        
        Args:
            discord_messages: List of Discord messages to process
            progress_callback: Optional progress callback (current, total, status)
            
        Returns:
            Tuple of (success: bool, stats: ProcessingStats)
        """
        logger.info(f"Building embedding cache for {len(discord_messages)} messages")
        start_time = time.time()
        
        # Reset image downloader stats before starting
        self.image_downloader.reset_stats()
        
        # Initialize statistics
        stats = ProcessingStats()
        stats.total_messages = len(discord_messages)
        
        # Collect all image attachments
        image_info = []  # (message, attachment) pairs
        for message in discord_messages:
            if message.has_images():
                stats.messages_with_images += 1
                for attachment in message.get_image_attachments():
                    image_info.append((message, attachment))
        
        total_images = len(image_info)
        stats.total_images = total_images
        
        if total_images == 0:
            logger.warning("No images found in Discord messages")
            stats.processing_time_seconds = time.time() - start_time
            return False, stats
        
        logger.info(f"Found {total_images} images to process")
        
        # Initialize storage
        all_embeddings = []
        metadata = CacheMetadata()
        processed_count = 0
        
        # Process in batches
        batch_size = 32
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            batch_info = image_info[batch_start:batch_end]
            
            if progress_callback:
                progress_callback(processed_count, total_images, 
                                f"Processing batch {batch_start//batch_size + 1}...")
            
            # Extract URLs for this batch
            batch_urls = [attachment.url for _, attachment in batch_info]
            
            # Download images for this batch
            logger.info(f"Downloading batch {batch_start//batch_size + 1}: {len(batch_urls)} images")
            download_results = self.image_downloader.download_images_batch(batch_urls)
            
            # Separate successful and failed downloads
            downloaded_images = []
            valid_batch_info = []
            
            for i, (url, image) in enumerate(download_results):
                if image is not None:
                    downloaded_images.append(image)
                    valid_batch_info.append(batch_info[i])
                else:
                    stats.failed_downloads += 1
            
            if downloaded_images:
                if progress_callback:
                    progress_callback(processed_count, total_images, 
                                    f"Encoding batch {batch_start//batch_size + 1} with CLIP...")
                
                # Encode images with CLIP
                try:
                    logger.info(f"Encoding {len(downloaded_images)} images with CLIP")
                    batch_embeddings = self.clip_engine.encode_images_batch(downloaded_images)
                    
                    # Store embeddings and metadata
                    for j, ((message, attachment), embedding) in enumerate(zip(valid_batch_info, batch_embeddings)):
                        all_embeddings.append(embedding)
                        
                        # Add to metadata with Discord URL
                        discord_url = self.config.discord.get_message_url(message.id)
                        metadata.add_image_metadata(len(all_embeddings) - 1, message, attachment)
                        metadata.index_to_metadata[len(all_embeddings) - 1]['discord_url'] = discord_url
                        
                        processed_count += 1
                        stats.processed_images += 1
                        
                        if progress_callback:
                            progress_callback(processed_count, total_images, 
                                            f"Cached {attachment.filename}")
                
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    continue
        
        if not all_embeddings:
            logger.error("No embeddings were generated")
            stats.processing_time_seconds = time.time() - start_time
            
            # Get download statistics and update ProcessingStats
            download_stats = self.image_downloader.get_stats()
            stats.expired_links = download_stats['expired_links']
            
            return False, stats
        
        # Get download statistics and update ProcessingStats
        download_stats = self.image_downloader.get_stats()
        stats.expired_links = download_stats['expired_links']
        
        # Log expired link information
        if stats.expired_links > 0:
            logger.warning(f"Found {stats.expired_links} expired links out of {stats.total_images} total images during cache building")
            expired_percentage = (stats.expired_links / stats.total_images) * 100
            logger.warning(f"Expired link percentage: {expired_percentage:.1f}%")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        
        if progress_callback:
            cache_status = "Saving cache..."
            if stats.expired_links > 0:
                cache_status += f" ({stats.expired_links} expired links detected)"
            progress_callback(stats.total_images, stats.total_images, cache_status)
        
        # Save to cache
        try:
            self._save_embeddings(embeddings_array)
            self._save_metadata(metadata)
            
            # Cache in memory for immediate use
            self._embeddings = embeddings_array
            self._metadata = metadata
            
            build_time = time.time() - start_time
            stats.processing_time_seconds = build_time
            
            logger.info(f"Cache built successfully in {build_time:.2f}s")
            logger.info(f"Cached {len(all_embeddings)} embeddings from {stats.total_images} images")
            
            if stats.expired_links > 0:
                logger.info(f"Cache build completed with {stats.expired_links} expired links - consider re-exporting HTML")
            
            return True, stats
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            stats.processing_time_seconds = time.time() - start_time
            return False, stats
    
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
            embeddings_path = self.embeddings_file
            metadata_path = self.metadata_file
            
            if embeddings_path.exists():
                embeddings_path.unlink()
                logger.info("Cleared embeddings cache")
            
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info("Cleared metadata cache")
            
            # Also clear parsed messages
            self.clear_parsed_messages()
            
            # Clear memory cache
            self._embeddings = None
            self._metadata = None
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
    
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
                embeddings_path = self.embeddings_file
                metadata_path = self.metadata_file
                
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
            if not hasattr(metadata, 'cache_version') or metadata.cache_version not in ["2.0", "2.1"]:
                logger.info(f"Cache version mismatch: expected 2.0/2.1, got {getattr(metadata, 'cache_version', 'unknown')}")
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
    
    def has_parsed_messages(self, html_file_path: str) -> bool:
        """Check if we have valid parsed messages for the given HTML file"""
        try:
            if not self.parsed_messages_file.exists() or not self.parsed_metadata_file.exists():
                return False
            
            # Load metadata
            with open(self.parsed_metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Check if HTML file path matches
            if metadata.html_file_path != html_file_path:
                logger.info(f"HTML file path changed: {metadata.html_file_path} -> {html_file_path}")
                return False
            
            # Check if HTML file still exists
            html_path = Path(html_file_path)
            if not html_path.exists():
                logger.info(f"HTML file no longer exists: {html_file_path}")
                return False
            
            # Check if HTML file has been modified
            current_mtime = html_path.stat().st_mtime
            if abs(current_mtime - metadata.html_file_mtime) > 1:  # 1 second tolerance
                logger.info(f"HTML file modified: {metadata.html_file_mtime} -> {current_mtime}")
                return False
            
            logger.info(f"Found valid parsed messages for {html_file_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking parsed messages: {e}")
            return False
    
    def load_parsed_messages(self) -> Optional[List[DiscordMessage]]:
        """Load previously parsed Discord messages"""
        try:
            if not self.parsed_messages_file.exists():
                return None
            
            with open(self.parsed_messages_file, 'rb') as f:
                messages = pickle.load(f)
            
            logger.info(f"Loaded {len(messages)} parsed Discord messages from cache")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to load parsed messages: {e}")
            return None
    
    def save_parsed_messages(self, messages: List[DiscordMessage], html_file_path: str):
        """Save parsed Discord messages to cache"""
        try:
            html_path = Path(html_file_path)
            html_mtime = html_path.stat().st_mtime
            
            # Save messages
            with open(self.parsed_messages_file, 'wb') as f:
                pickle.dump(messages, f)
            
            # Save metadata
            metadata = ParsedMessagesMetadata(html_file_path, html_mtime, len(messages))
            with open(self.parsed_metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved {len(messages)} parsed Discord messages to cache")
            
        except Exception as e:
            logger.error(f"Failed to save parsed messages: {e}")
    
    def clear_parsed_messages(self):
        """Clear saved parsed messages"""
        try:
            if self.parsed_messages_file.exists():
                self.parsed_messages_file.unlink()
                logger.info("Cleared parsed messages cache")
            
            if self.parsed_metadata_file.exists():
                self.parsed_metadata_file.unlink()
                logger.info("Cleared parsed messages metadata")
                
        except Exception as e:
            logger.error(f"Failed to clear parsed messages: {e}")
    