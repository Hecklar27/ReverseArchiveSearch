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
            'message_type': message.type,
            'filename': attachment.filename,
            'url': attachment.url,
            'author_name': message.author.name,
            'timestamp': message.timestamp,
            'content': message.content[:100] if message.content else "",
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CLIP engine
        self.clip_engine = CLIPEngine(config)
        self.image_downloader = ImageDownloader(timeout=10, max_retries=3, max_workers=8)
        
        # Cache will be model-specific
        self._embeddings = None
        self._metadata = None
        self._current_model = None
        
    def _get_model_cache_paths(self, model_name: str = None):
        """Get cache file paths for specific model"""
        if model_name is None:
            model_name = self.clip_engine.model_name
            
        model_cache_dir = self.config.cache.get_model_cache_dir(model_name)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'embeddings_file': model_cache_dir / "embeddings.npy",
            'metadata_file': model_cache_dir / "metadata.pkl",
            'parsed_messages_file': model_cache_dir / "parsed_messages.pkl",
            'parsed_metadata_file': model_cache_dir / "parsed_metadata.pkl"
        }
    
    def _load_cache_for_model(self, model_name: str):
        """Load cache for specific model"""
        paths = self._get_model_cache_paths(model_name)
        
        try:
            with open(paths['metadata_file'], 'rb') as f:
                metadata = pickle.load(f)
            
            with open(paths['embeddings_file'], 'rb') as f:
                embeddings = pickle.load(f)
                
            return embeddings, metadata
        except Exception as e:
            logger.warning(f"Failed to load cache for model {model_name}: {e}")
            return None, None
    
    def has_valid_cache(self, model_name: str = None) -> bool:
        """Check if valid cache exists for specific model"""
        try:
            if model_name is None:
                model_name = self.clip_engine.model_name
                
            paths = self._get_model_cache_paths(model_name)
            embeddings_path = paths['embeddings_file']
            metadata_path = paths['metadata_file']
            
            if not (embeddings_path.exists() and metadata_path.exists()):
                return False
            
            # Load and validate metadata
            metadata = self._load_metadata(model_name)
            if metadata is None:
                return False
            
            # Check if cache is expired
            if metadata.is_expired(self.config.cache.max_age_days):
                logger.info(f"Cache for {model_name} expired, needs regeneration")
                return False
            
            # Validate CLIP model compatibility
            if metadata.clip_model != model_name:
                logger.info(f"CLIP model changed ({metadata.clip_model} -> {model_name}), cache invalid")
                return False
            
            # Validate PyTorch version compatibility
            current_pytorch_version = torch.__version__
            cached_pytorch_version = getattr(metadata, 'pytorch_version', 'unknown')
            if cached_pytorch_version != current_pytorch_version:
                logger.info(f"PyTorch version changed ({cached_pytorch_version} -> {current_pytorch_version}), cache invalid for {model_name}")
                return False
            
            # Validate cache version and required fields
            if not self._validate_cache_compatibility(metadata):
                logger.info(f"Cache format is incompatible for {model_name}, needs rebuilding")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating cache for {model_name}: {e}")
            return False
    
    def _load_metadata(self, model_name: str) -> Optional[CacheMetadata]:
        """Load cache metadata for specific model"""
        try:
            paths = self._get_model_cache_paths(model_name)
            metadata_path = paths['metadata_file']
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata for {model_name}: {e}")
            return None
    
    def _save_metadata(self, metadata: CacheMetadata, model_name: str) -> None:
        """Save cache metadata for specific model"""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            paths = self._get_model_cache_paths(model_name)
            metadata_path = paths['metadata_file']
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save cache metadata for {model_name}: {e}")
            raise
    
    def _load_embeddings(self, model_name: str) -> Optional[np.ndarray]:
        """Load cached embeddings for specific model"""
        try:
            paths = self._get_model_cache_paths(model_name)
            embeddings_path = paths['embeddings_file']
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings for {model_name}: {e}")
            return None
    
    def _save_embeddings(self, embeddings: np.ndarray, model_name: str) -> None:
        """Save embeddings to cache for specific model"""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            paths = self._get_model_cache_paths(model_name)
            embeddings_path = paths['embeddings_file']
            with open(embeddings_path, 'wb') as f:
                if self.config.cache.compression:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to save embeddings for {model_name}: {e}")
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
        metadata.clip_model = self.clip_engine.model_name  # Set CLIP model for validation
        processed_count = 0
        
        # Use model-specific optimal batch size
        optimal_batch_size = self.config.clip.get_optimal_batch_size()
        batch_size = optimal_batch_size
        logger.info(f"Using optimal batch size {batch_size} for model {self.clip_engine.model_name}")
        
        # Aggressive batch size reduction for performance and memory management
        if self.clip_engine.model_name in ["ViT-L/14", "ViT-L/14@336px", "RN50x64"]:
            # Use much smaller batches for large models to prevent GPU memory issues
            batch_size = 8  # Reduced from 16 to 8 for better memory management
            logger.info(f"PERFORMANCE OPTIMIZATION: Using reduced batch size {batch_size} for large model {self.clip_engine.model_name}")
        elif self.clip_engine.model_name in ["ViT-B/32", "ViT-B/16"]:
            # Smaller models can handle larger batches but still be conservative
            batch_size = min(16, batch_size)  # Max 16 for smaller models
            logger.info(f"PERFORMANCE OPTIMIZATION: Using optimized batch size {batch_size} for model {self.clip_engine.model_name}")
        
        logger.info(f"FINAL BATCH SIZE: {batch_size} images per batch (down from previous 32)")
        
        # Additional memory optimization
        if torch.cuda.is_available():
            # Clear GPU cache before starting
            torch.cuda.empty_cache()
            logger.info("PERFORMANCE OPTIMIZATION: Cleared GPU cache before processing")
        
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
                batch_num = batch_start // batch_size + 1
                if progress_callback:
                    progress_callback(processed_count, total_images, 
                                    f"Encoding batch {batch_num} ({len(downloaded_images)} images) with CLIP...")
                
                # Encode images with CLIP
                try:
                    logger.info(f"Encoding {len(downloaded_images)} images with CLIP (batch {batch_num})")
                    start_clip_time = time.time()
                    batch_embeddings = self.clip_engine.encode_images_batch(downloaded_images)
                    clip_time = time.time() - start_clip_time
                    logger.info(f"CLIP encoding completed in {clip_time:.2f}s for batch {batch_num}")
                    
                    # Memory cleanup after CLIP processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Clear downloaded images from memory immediately after encoding
                    downloaded_images.clear()
                    
                    # Force garbage collection every few batches to prevent memory accumulation
                    if batch_num % 5 == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"Performed memory cleanup after batch {batch_num}")
                    
                    if progress_callback:
                        progress_callback(processed_count, total_images, 
                                        f"Storing batch {batch_num} embeddings...")
                    
                    # Store embeddings and metadata
                    for j, ((message, attachment), embedding) in enumerate(zip(valid_batch_info, batch_embeddings)):
                        all_embeddings.append(embedding)
                        
                        # Add to metadata with Discord URL
                        discord_url = self.config.discord.get_message_url(message.id)
                        metadata.add_image_metadata(len(all_embeddings) - 1, message, attachment)
                        metadata.index_to_metadata[len(all_embeddings) - 1]['discord_url'] = discord_url
                        
                        processed_count += 1
                        stats.processed_images += 1
                    
                    # Update progress once per batch instead of per image
                    if progress_callback:
                        progress_callback(processed_count, total_images, 
                                        f"Completed batch {batch_num} - cached {processed_count}/{total_images} images")
                
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_num}: {e}")
                    
                    # Clear memory on error
                    downloaded_images.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
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
        
        # Set total_images in metadata to actual count of processed embeddings
        metadata.total_images = len(all_embeddings)
        
        if progress_callback:
            cache_status = "Saving cache..."
            if stats.expired_links > 0:
                cache_status += f" ({stats.expired_links} expired links detected)"
            progress_callback(stats.total_images, stats.total_images, cache_status)
        
        # Save to cache
        try:
            self._save_embeddings(embeddings_array, self.clip_engine.model_name)
            self._save_metadata(metadata, self.clip_engine.model_name)
            
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
            # Check if cache is already loaded in memory
            if (self._embeddings is not None and self._metadata is not None):
                logger.info(f"Cache already loaded in memory with {len(self._embeddings)} embeddings")
                return True
            
            # Try to load from files
            if not self.has_valid_cache():
                logger.info("No valid cache files found")
                return False
            
            self._embeddings = self._load_embeddings(self.clip_engine.model_name)
            self._metadata = self._load_metadata(self.clip_engine.model_name)
            
            if self._embeddings is None or self._metadata is None:
                logger.info("Failed to load cache data from files")
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
            embeddings_path = self._get_model_cache_paths(self.clip_engine.model_name)['embeddings_file']
            metadata_path = self._get_model_cache_paths(self.clip_engine.model_name)['metadata_file']
            
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
            
            # Fallback: if total_images is 0 in metadata, use actual embedding count
            if stats.get('total_images', 0) == 0:
                embeddings = self.get_embeddings()
                if embeddings is not None:
                    stats['total_images'] = len(embeddings)
            
            # Add file sizes
            try:
                embeddings_path = self._get_model_cache_paths(self.clip_engine.model_name)['embeddings_file']
                metadata_path = self._get_model_cache_paths(self.clip_engine.model_name)['metadata_file']
                
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
            current_model = self.clip_engine.model_name
            current_paths = self._get_model_cache_paths(current_model)
            
            # First check current model
            if self._check_parsed_messages_for_model(html_file_path, current_model):
                return True
            
            # If current model doesn't have them, check other models
            other_models = ['ViT-B/32', 'ViT-L/14', 'RN50x64', 'ViT-B/16']
            for model in other_models:
                if model != current_model:
                    if self._check_parsed_messages_for_model(html_file_path, model):
                        logger.info(f"Found valid parsed messages for {html_file_path} in {model} cache")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking parsed messages: {e}")
            return False
    
    def _check_parsed_messages_for_model(self, html_file_path: str, model_name: str) -> bool:
        """Check if a specific model has valid parsed messages for the given HTML file"""
        try:
            paths = self._get_model_cache_paths(model_name)
            parsed_messages_file = paths['parsed_messages_file']
            parsed_metadata_file = paths['parsed_metadata_file']
            
            if not parsed_messages_file.exists() or not parsed_metadata_file.exists():
                return False
            
            # Load metadata
            with open(parsed_metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Check if HTML file path matches
            if metadata.html_file_path != html_file_path:
                return False
            
            # Check if HTML file still exists
            html_path = Path(html_file_path)
            if not html_path.exists():
                return False
            
            # Check if HTML file has been modified
            current_mtime = html_path.stat().st_mtime
            if abs(current_mtime - metadata.html_file_mtime) > 1:  # 1 second tolerance
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking parsed messages for {model_name}: {e}")
            return False
    
    def load_parsed_messages(self) -> Optional[List[DiscordMessage]]:
        """Load previously parsed Discord messages"""
        try:
            current_model = self.clip_engine.model_name
            current_paths = self._get_model_cache_paths(current_model)
            
            # First try current model
            if current_paths['parsed_messages_file'].exists():
                with open(current_paths['parsed_messages_file'], 'rb') as f:
                    messages = pickle.load(f)
                logger.info(f"Loaded {len(messages)} parsed Discord messages from cache for {current_model}")
                return messages
            
            # If current model doesn't have parsed messages, check other models
            logger.info(f"No parsed messages found for {current_model}, checking other models...")
            other_models = ['ViT-B/32', 'ViT-L/14', 'RN50x64', 'ViT-B/16']  # Common models
            
            for model in other_models:
                if model != current_model:
                    other_paths = self._get_model_cache_paths(model)
                    if other_paths['parsed_messages_file'].exists():
                        try:
                            with open(other_paths['parsed_messages_file'], 'rb') as f:
                                messages = pickle.load(f)
                            
                            # Copy to current model's cache for future use
                            logger.info(f"Found parsed messages in {model} cache, copying to {current_model}")
                            
                            # Load metadata from source model to preserve HTML file info
                            if other_paths['parsed_metadata_file'].exists():
                                with open(other_paths['parsed_metadata_file'], 'rb') as f:
                                    source_metadata = pickle.load(f)
                                
                                # Save to current model's cache
                                self.save_parsed_messages(messages, source_metadata.html_file_path)
                                
                                logger.info(f"Copied {len(messages)} parsed messages from {model} to {current_model}")
                                return messages
                                
                        except Exception as e:
                            logger.warning(f"Failed to load parsed messages from {model}: {e}")
                            continue
            
            logger.info("No parsed messages found in any model cache")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load parsed messages: {e}")
            return None
    
    def save_parsed_messages(self, messages: List[DiscordMessage], html_file_path: str):
        """Save parsed Discord messages to cache"""
        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            html_path = Path(html_file_path)
            html_mtime = html_path.stat().st_mtime
            
            # Save messages
            with open(self._get_model_cache_paths(self.clip_engine.model_name)['parsed_messages_file'], 'wb') as f:
                pickle.dump(messages, f)
            
            # Save metadata
            metadata = ParsedMessagesMetadata(html_file_path, html_mtime, len(messages))
            with open(self._get_model_cache_paths(self.clip_engine.model_name)['parsed_metadata_file'], 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved {len(messages)} parsed Discord messages to cache")
            
        except Exception as e:
            logger.error(f"Failed to save parsed messages: {e}")
    
    def clear_parsed_messages(self):
        """Clear saved parsed messages"""
        try:
            if self._get_model_cache_paths(self.clip_engine.model_name)['parsed_messages_file'].exists():
                self._get_model_cache_paths(self.clip_engine.model_name)['parsed_messages_file'].unlink()
                logger.info("Cleared parsed messages cache")
            
            if self._get_model_cache_paths(self.clip_engine.model_name)['parsed_metadata_file'].exists():
                self._get_model_cache_paths(self.clip_engine.model_name)['parsed_metadata_file'].unlink()
                logger.info("Cleared parsed messages metadata")
                
        except Exception as e:
            logger.error(f"Failed to clear parsed messages: {e}")
    
    def get_all_model_cache_status(self) -> dict:
        """Get cache status for all available models"""
        status = {}
        
        for model_name in self.config.clip.get_available_models():
            try:
                if self.has_valid_cache(model_name):
                    metadata = self._load_metadata(model_name)
                    if metadata:
                        model_stats = metadata.get_stats()
                        model_stats['status'] = 'valid'
                        
                        # Add file sizes
                        try:
                            paths = self._get_model_cache_paths(model_name)
                            embeddings_path = paths['embeddings_file']
                            metadata_path = paths['metadata_file']
                            
                            if embeddings_path.exists():
                                model_stats['embeddings_size_mb'] = embeddings_path.stat().st_size / (1024 * 1024)
                            if metadata_path.exists():
                                model_stats['metadata_size_mb'] = metadata_path.stat().st_size / (1024 * 1024)
                                
                        except Exception as e:
                            logger.warning(f"Failed to get cache file sizes for {model_name}: {e}")
                        
                        status[model_name] = model_stats
                    else:
                        status[model_name] = {'status': 'invalid'}
                else:
                    status[model_name] = {'status': 'no_cache'}
                    
            except Exception as e:
                logger.warning(f"Failed to get cache status for {model_name}: {e}")
                status[model_name] = {'status': 'error', 'error': str(e)}
        
        return status
    
    def switch_model(self, new_model_name: str):
        """Switch to a different CLIP model and update cache accordingly"""
        if new_model_name == self.clip_engine.model_name:
            logger.info(f"Already using model {new_model_name}")
            return
        
        logger.info(f"Switching from {self.clip_engine.model_name} to {new_model_name}")
        
        # Clear current cache from memory
        self._embeddings = None
        self._metadata = None
        self._current_model = None
        
        # Update CLIP engine with new model
        old_model = self.clip_engine.model_name
        self.clip_engine.model_name = new_model_name
        
        # Reinitialize CLIP engine with new model
        try:
            # Reload the CLIP model
            import clip
            self.clip_engine.model, self.clip_engine.preprocess = clip.load(new_model_name, device=self.clip_engine.device)
            self.clip_engine.model_name = new_model_name
            
            # Update embedding dimensions
            model_info = self.config.clip.get_model_info(new_model_name)
            if model_info:
                self.clip_engine.embedding_dim = model_info['embedding_dim']
            
            logger.info(f"Successfully switched to model {new_model_name}")
            
        except Exception as e:
            # Rollback on failure
            logger.error(f"Failed to switch to model {new_model_name}: {e}")
            self.clip_engine.model_name = old_model
            raise
    
    def clear_cache_for_model(self, model_name: str) -> None:
        """Clear cache files for specific model"""
        try:
            paths = self._get_model_cache_paths(model_name)
            embeddings_path = paths['embeddings_file']
            metadata_path = paths['metadata_file']
            parsed_messages_path = paths['parsed_messages_file']
            parsed_metadata_path = paths['parsed_metadata_file']
            
            files_cleared = []
            
            if embeddings_path.exists():
                embeddings_path.unlink()
                files_cleared.append("embeddings")
            
            if metadata_path.exists():
                metadata_path.unlink()
                files_cleared.append("metadata")
            
            if parsed_messages_path.exists():
                parsed_messages_path.unlink()
                files_cleared.append("parsed messages")
                
            if parsed_metadata_path.exists():
                parsed_metadata_path.unlink()
                files_cleared.append("parsed metadata")
            
            # Clear from memory if it's the current model
            if model_name == self.clip_engine.model_name:
                self._embeddings = None
                self._metadata = None
                self._current_model = None
            
            if files_cleared:
                logger.info(f"Cleared {model_name} cache: {', '.join(files_cleared)}")
            else:
                logger.info(f"No cache files found for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear cache for {model_name}: {e}")
            raise
    