"""
Search strategy implementations using the Strategy pattern.
"""

import logging
import requests
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable, Dict, Any
from pathlib import Path
from PIL import Image
from io import BytesIO

from data.models import DiscordMessage, SearchResult, ProcessingStats, DiscordAttachment, DiscordUser
from core.config import Config
from .engine_factory import UniversalEngine
from .image_downloader import ImageDownloader
from .cache_manager import EmbeddingCacheManager

logger = logging.getLogger(__name__)

class SearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    @abstractmethod
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage], 
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform image similarity search.
        
        Args:
            user_image_path: Path to user's query image
            discord_messages: List of Discord messages with image attachments
            progress_callback: Optional callback function (current, total, status)
            
        Returns:
            Tuple of (search results, processing statistics)
        """
        pass

class OptimizedRealTimeSearchStrategy(SearchStrategy):
    """Optimized real-time search strategy with batched downloads and processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.clip_engine = UniversalEngine(config)  # Use UniversalEngine instead of CLIPEngine
        self.image_downloader = ImageDownloader(max_workers=8)  # Concurrent downloads
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage],
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform optimized real-time search with batched downloads and universal engine processing.
        """
        logger.info(f"Starting optimized real-time search with {len(discord_messages)} messages using {self.clip_engine.get_model_type().upper()}")
        start_time = time.time()
        
        # Reset image downloader stats before starting
        self.image_downloader.reset_stats()
        
        # Initialize statistics
        stats = ProcessingStats()
        stats.total_messages = len(discord_messages)
        
        # Collect all image URLs first
        image_info = []  # (message, attachment) pairs
        for message in discord_messages:
            if message.has_images():
                stats.messages_with_images += 1
                for attachment in message.get_image_attachments():
                    image_info.append((message, attachment))
        
        stats.total_images = len(image_info)
        
        if not image_info:
            logger.warning("No images found in Discord messages")
            return [], stats
        
        # Report initial progress
        if progress_callback:
            progress_callback(0, stats.total_images, "Encoding user image...")
        
        # Encode user image
        try:
            logger.info(f"Encoding user image: {user_image_path}")
            user_embedding = self.clip_engine.encode_image(user_image_path)
        except Exception as e:
            logger.error(f"Failed to encode user image: {e}")
            raise
        
        # Process Discord images in optimized batches
        batch_size = self.config.cache.batch_size  # Default 32
        
        # Use model-specific optimal batch size
        optimal_batch_size = self.config.clip.get_optimal_batch_size()
        batch_size = optimal_batch_size
        logger.info(f"Using optimal batch size {batch_size} for model {self.clip_engine.model_name}")
        
        # Aggressive batch size reduction for performance and memory management
        if self.clip_engine.model_name in ["ViT-L/14", "ViT-L/14@336px", "RN50x64"]:
            # Use much smaller batches for large models to prevent GPU memory issues
            batch_size = 8  # Reduced from 16 to 8 for better memory management
            logger.info(f"Using reduced batch size {batch_size} for large model {self.clip_engine.model_name}")
        elif self.clip_engine.model_name in ["ViT-B/16"]:
            # Smaller models can handle larger batches but still be conservative
            batch_size = min(16, batch_size)  # Max 16 for smaller models
            logger.info(f"Using optimized batch size {batch_size} for model {self.clip_engine.model_name}")
        elif self.clip_engine.model_name.startswith("DINOv2"):
            # DINOv2 models are generally efficient
            batch_size = 16
            logger.info(f"Using DINOv2 optimized batch size {batch_size} for model {self.clip_engine.model_name}")
        
        # Additional memory optimization
        import torch
        if torch.cuda.is_available():
            # Clear GPU cache before starting
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache before real-time search")
        
        results = []
        processed_count = 0
        
        # Process in batches
        for batch_start in range(0, len(image_info), batch_size):
            batch_end = min(batch_start + batch_size, len(image_info))
            batch_info = image_info[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            # Update progress for batch start
            if progress_callback:
                progress_callback(processed_count, stats.total_images, 
                                f"Processing batch {batch_num} - downloading {len(batch_info)} images...")
            
            logger.info(f"Processing batch {batch_num}: {len(batch_info)} images")
            
            # Download batch
            try:
                batch_start_time = time.time()
                
                # Download images concurrently
                download_urls = [attachment.url for (message, attachment) in batch_info]
                download_results = self.image_downloader.download_images_batch(download_urls)
                
                download_time = time.time() - batch_start_time
                logger.info(f"Downloaded {len(download_results)}/{len(batch_info)} images in {download_time:.2f}s")
                
                # Extract valid downloads and images
                valid_batch_info = []
                valid_images = []
                for i, ((message, attachment), (url, image)) in enumerate(zip(batch_info, download_results)):
                    if image is not None:
                        valid_batch_info.append((message, attachment))
                        valid_images.append(image)
                
                if valid_images:
                    # Update progress for encoding - this is the slow part for large models
                    batch_num = batch_start // batch_size + 1
                    if progress_callback:
                        progress_callback(processed_count, stats.total_images, 
                                        f"Encoding batch {batch_num} ({len(valid_images)} images) with {self.clip_engine.get_model_type().upper()}...")
                    
                    # Encode images in batch with universal engine
                    try:
                        logger.info(f"Encoding {len(valid_images)} images with {self.clip_engine.get_model_type().upper()} (batch {batch_num})")
                        start_clip_time = time.time()
                        batch_embeddings = self.clip_engine.encode_images_batch(valid_images)
                        clip_time = time.time() - start_clip_time
                        logger.info(f"{self.clip_engine.get_model_type().upper()} encoding completed in {clip_time:.2f}s for batch {batch_num}")
                        
                        # Memory cleanup after encoding
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Clear downloaded images from memory immediately after encoding
                        valid_images.clear()
                        download_results.clear()
                        
                        # Update progress after encoding
                        if progress_callback:
                            progress_callback(processed_count, stats.total_images, 
                                            f"Calculating similarities for batch {batch_num}...")
                        
                        # Calculate similarities for this batch
                        similarities = self.clip_engine.calculate_similarities_batch(
                            user_embedding, batch_embeddings)
                        
                        # Log exact matches for DINOv2
                        if self.clip_engine.is_dinov2:
                            threshold = self.clip_engine.get_similarity_threshold()
                            exact_matches = np.sum(similarities >= threshold) if threshold else 0
                            logger.info(f"Batch {batch_num}: Found {exact_matches} exact matches above threshold")
                        
                        # Create search results (reduce progress update frequency)
                        for i, ((message, attachment), similarity) in enumerate(zip(valid_batch_info, similarities)):
                            # Generate Discord URL
                            discord_url = self.config.discord.get_message_url(message.id)
                            
                            # Create search result
                            result = SearchResult(
                                message=message,
                                attachment=attachment,
                                similarity_score=float(similarity),
                                discord_url=discord_url
                            )
                            results.append(result)
                            stats.processed_images += 1
                            
                            processed_count += 1
                            
                            logger.debug(f"Processed image {attachment.filename}: similarity={similarity:.3f}")
                        
                        # Force garbage collection every few batches to prevent memory accumulation
                        if batch_num % 5 == 0:
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.info(f"Performed memory cleanup after batch {batch_num}")
                        
                        # Update progress once per batch instead of per image
                        if progress_callback:
                            progress_callback(processed_count, stats.total_images, 
                                            f"Completed batch {batch_num} - processed {processed_count}/{stats.total_images} images")
                            
                    except Exception as e:
                        logger.error(f"Failed to process batch {batch_num}: {e}")
                        
                        # Clear memory on error
                        valid_images.clear()
                        download_results.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Skip this batch but continue with others
                        processed_count += len(valid_batch_info)
                        continue
                else:
                    # No valid images in this batch
                    processed_count += len(batch_info)
            
            except Exception as e:
                logger.error(f"Failed to download batch {batch_num}: {e}")
                processed_count += len(batch_info)
                continue
        
        # Final progress update
        if progress_callback:
            progress_callback(stats.total_images, stats.total_images, "Sorting results...")
        
        # Sort results by similarity (highest first)
        results.sort(reverse=True, key=lambda x: x.similarity_score)
        
        # Limit results
        max_results = self.config.ui.max_results
        if len(results) > max_results:
            results = results[:max_results]
        
        # Get download statistics and update ProcessingStats
        download_stats = self.image_downloader.get_stats()
        stats.expired_links = download_stats['expired_links']
        
        # Finalize statistics
        stats.processing_time_seconds = time.time() - start_time
        
        logger.info(f"Optimized real-time search completed in {stats.processing_time_seconds:.2f}s")
        logger.info(f"Processed {stats.processed_images}/{stats.total_images} images successfully")
        logger.info(f"Found {stats.expired_links} expired links")
        logger.info(f"Found {len(results)} results")
        
        # Log summary for DINOv2
        if self.clip_engine.is_dinov2 and results:
            threshold = self.clip_engine.get_similarity_threshold()
            exact_matches = sum(1 for r in results if r.similarity_score >= threshold) if threshold else 0
            logger.info(f"Total exact matches found: {exact_matches}/{len(results)}")
        
        return results, stats

class RealTimeSearchStrategy(SearchStrategy):
    """Real-time search strategy - downloads and processes images on demand"""
    
    def __init__(self, config: Config):
        self.config = config
        self.clip_engine = UniversalEngine(config)  # Use UniversalEngine instead of CLIPEngine
        self.image_downloader = ImageDownloader()
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage],
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform real-time search by downloading and processing Discord images on demand.
        """
        logger.info(f"Starting real-time search with {len(discord_messages)} messages using {self.clip_engine.get_model_type().upper()}")
        start_time = time.time()
        
        # Reset image downloader stats before starting
        self.image_downloader.reset_stats()
        
        # Initialize statistics
        stats = ProcessingStats()
        stats.total_messages = len(discord_messages)
        
        # Count total images first for progress tracking
        total_images = 0
        for message in discord_messages:
            if message.has_images():
                stats.messages_with_images += 1
                total_images += len(message.get_image_attachments())
        
        stats.total_images = total_images
        
        # Report initial progress
        if progress_callback:
            progress_callback(0, total_images, "Encoding user image...")
        
        # Encode user image
        try:
            logger.info(f"Encoding user image: {user_image_path}")
            user_embedding = self.clip_engine.encode_image(user_image_path)
        except Exception as e:
            logger.error(f"Failed to encode user image: {e}")
            raise
        
        # Process Discord messages
        results = []
        processed_count = 0
        
        for message in discord_messages:
            if not message.has_images():
                continue
            
            # Process each image attachment in the message
            for attachment in message.get_image_attachments():
                processed_count += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(processed_count, total_images, f"Processing {attachment.filename}...")
                
                try:
                    # Download image
                    discord_image = self.image_downloader.download_image(attachment.url)
                    if discord_image is None:
                        stats.failed_downloads += 1
                        continue
                    
                    # Encode Discord image
                    discord_embedding = self.clip_engine.encode_image(discord_image)
                    
                    # Calculate similarity
                    similarity = self.clip_engine.calculate_similarity(user_embedding, discord_embedding)
                    
                    # Generate Discord URL
                    discord_url = self.config.discord.get_message_url(message.id)
                    
                    # Create search result
                    result = SearchResult(
                        message=message,
                        attachment=attachment,
                        similarity_score=similarity,
                        discord_url=discord_url
                    )
                    results.append(result)
                    stats.processed_images += 1
                    
                    logger.debug(f"Processed image {attachment.filename}: similarity={similarity:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {attachment.url}: {e}")
                    stats.failed_downloads += 1
                    continue
        
        # Final progress update
        if progress_callback:
            progress_callback(total_images, total_images, "Sorting results...")
        
        # Sort results by similarity (highest first)
        results.sort(reverse=True, key=lambda x: x.similarity_score)
        
        # Limit results
        max_results = self.config.ui.max_results
        if len(results) > max_results:
            results = results[:max_results]
        
        # Get download statistics and update ProcessingStats
        download_stats = self.image_downloader.get_stats()
        stats.expired_links = download_stats['expired_links']
        
        # Finalize statistics
        stats.processing_time_seconds = time.time() - start_time
        
        logger.info(f"Real-time search completed in {stats.processing_time_seconds:.2f}s")
        logger.info(f"Processed {stats.processed_images}/{stats.total_images} images successfully")
        logger.info(f"Found {stats.expired_links} expired links")
        logger.info(f"Found {len(results)} results")
        
        # Log summary for DINOv2
        if self.clip_engine.is_dinov2 and results:
            threshold = self.clip_engine.get_similarity_threshold()
            exact_matches = sum(1 for r in results if r.similarity_score >= threshold) if threshold else 0
            logger.info(f"Total exact matches found: {exact_matches}/{len(results)}")
        
        return results, stats

class CachedSearchStrategy(SearchStrategy):
    """Cached search strategy - uses pre-computed embeddings (Phase 2)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.clip_engine = UniversalEngine(config)  # Use UniversalEngine instead of CLIPEngine
        self.cache_manager = EmbeddingCacheManager(config)
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage],
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform cached search using pre-computed embeddings.
        """
        logger.info(f"Starting cached search with pre-computed embeddings using {self.clip_engine.get_model_type().upper()}")
        start_time = time.time()
        
        # Initialize statistics
        stats = ProcessingStats()
        stats.total_messages = len(discord_messages)
        
        # Enhanced cache validation with in-memory fallback
        cache_available = False
        
        # First, try standard cache validation
        if self.cache_manager.has_valid_cache():
            cache_available = True
            logger.info("Cache validated via file-based checks")
        else:
            # Fallback: check if cache is available in memory (newly built)
            if (hasattr(self.cache_manager, '_embeddings') and self.cache_manager._embeddings is not None and
                hasattr(self.cache_manager, '_metadata') and self.cache_manager._metadata is not None):
                cache_available = True
                logger.info("Cache available via in-memory fallback - using newly built cache")
            else:
                logger.info("No cache available via any validation method")
        
        if not cache_available:
            logger.error("No valid cache available. Please run 'Pre-process Archive' first.")
            raise ValueError("No valid cache available. Please run 'Pre-process Archive' first.")
        
        # Load cache if not already loaded (this should now work with in-memory cache)
        if not self.cache_manager.load_cache():
            logger.error("Failed to load cache. Please rebuild cache.")
            raise ValueError("Failed to load cache. Please rebuild cache.")
        
        # Get cached data
        cached_embeddings = self.cache_manager.get_embeddings()
        metadata = self.cache_manager.get_metadata()
        
        if cached_embeddings is None or metadata is None:
            logger.error("Cache data is corrupted. Please rebuild cache.")
            raise ValueError("Cache data is corrupted. Please rebuild cache.")
        
        logger.info(f"Successfully loaded cache with {len(cached_embeddings)} embeddings")
        
        # Report progress
        if progress_callback:
            progress_callback(0, 1, "Encoding user image...")
        
        # Encode user image
        try:
            logger.info(f"Encoding user image: {user_image_path}")
            user_embedding = self.clip_engine.encode_image(user_image_path)
            logger.info("User image encoded successfully")
        except Exception as e:
            logger.error(f"Failed to encode user image: {e}")
            raise
        
        # Report progress
        if progress_callback:
            progress_callback(1, 2, "Calculating similarities...")
        
        # Calculate similarities with all cached embeddings
        try:
            logger.info("Calculating similarities with cached embeddings...")
            similarities = self.clip_engine.calculate_similarities_batch(
                user_embedding, cached_embeddings)
            logger.info(f"Calculated similarities for {len(similarities)} cached embeddings")
            
            # Log exact matches for DINOv2
            if self.clip_engine.is_dinov2:
                threshold = self.clip_engine.get_similarity_threshold()
                exact_matches = np.sum(similarities >= threshold) if threshold else 0
                logger.info(f"Found {exact_matches} exact matches above threshold {threshold}")
                
        except Exception as e:
            logger.error(f"Failed to calculate similarities: {e}")
            raise
        
        # Create search results from cache metadata
        results = []
        
        # Get indices sorted by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        for index in sorted_indices:
            try:
                cache_data = metadata.index_to_metadata[index]
                similarity_score = similarities[index]
                
                # Reconstruct DiscordMessage and DiscordAttachment from cache
                user = DiscordUser(
                    id="unknown",
                    name=cache_data.get('author_name', 'Unknown'),
                    discriminator="0000",
                    avatar_url=None
                )
                
                message = DiscordMessage(
                    id=cache_data['message_id'],
                    type=cache_data.get('message_type', 'default'),
                    timestamp=cache_data.get('timestamp', 'Unknown'),
                    content=cache_data.get('content', ''),
                    author=user,
                    attachments=[]
                )
                
                attachment = DiscordAttachment(
                    id="unknown",
                    url=cache_data['url'],
                    filename=cache_data['filename'],
                    file_size_bytes=0
                )
                
                # Get Discord URL
                discord_url = cache_data.get('discord_url', self.config.discord.get_message_url(message.id))
                
                # Create search result
                result = SearchResult(
                    message=message,
                    attachment=attachment,
                    similarity_score=float(similarity_score),
                    discord_url=discord_url
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to reconstruct objects from cache data at index {index}: {e}")
                logger.error(f"Cache data: {cache_data}")
                raise
        
        # Update statistics
        stats.total_images = len(cached_embeddings)
        stats.processed_images = len(results)
        stats.messages_with_images = len(set(r.message.id for r in results))
        
        # Report progress
        if progress_callback:
            progress_callback(2, 2, "Sorting results...")
        
        # Sort results by similarity (highest first) - already sorted but ensuring consistency
        results.sort(reverse=True, key=lambda x: x.similarity_score)
        
        # Limit results
        max_results = self.config.ui.max_results
        if len(results) > max_results:
            results = results[:max_results]
        
        # Finalize statistics
        stats.processing_time_seconds = time.time() - start_time
        
        logger.info(f"Cached search completed in {stats.processing_time_seconds:.2f}s")
        logger.info(f"Processed {stats.processed_images} cached embeddings")
        logger.info(f"Found {len(results)} results")
        
        # Log summary for DINOv2
        if self.clip_engine.is_dinov2 and results:
            threshold = self.clip_engine.get_similarity_threshold()
            exact_matches = sum(1 for r in results if r.similarity_score >= threshold) if threshold else 0
            logger.info(f"Total exact matches in results: {exact_matches}/{len(results)}")
        
        return results, stats

class SearchEngine:
    """Main search engine that orchestrates different search strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize strategies with UniversalEngine
        self.optimized_strategy = OptimizedRealTimeSearchStrategy(config)
        self.realtime_strategy = RealTimeSearchStrategy(config)
        self.cached_strategy = CachedSearchStrategy(config)
        self.cache_manager = EmbeddingCacheManager(config)
        
        # Get current model type for logging
        model_type = config.clip.get_model_type(config.clip.model_name)
        logger.info(f"SearchEngine initialized with {model_type.upper()} feature engine")
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage], 
               use_cache: bool = False, use_optimization: bool = True,
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform image search using the specified strategy.
        
        Args:
            user_image_path: Path to user's query image
            discord_messages: List of Discord messages with image attachments
            use_cache: Whether to use cached search (Phase 2+)
            use_optimization: Whether to use optimized real-time search (default: True)
            progress_callback: Optional callback function (current, total, status)
            
        Returns:
            Tuple of (search results, processing statistics)
        """
        
        model_type = self.config.clip.get_model_type(self.config.clip.model_name)
        
        if use_cache:
            logger.info(f"Using cached search strategy with {model_type.upper()} engine")
            try:
                return self.cached_strategy.search(user_image_path, discord_messages, progress_callback)
            except ValueError as e:
                logger.warning(f"Cached search failed (cache issue): {e}")
                logger.info("Falling back to real-time search")
                # Fall back to real-time search
                use_cache = False
            except Exception as e:
                logger.warning(f"Cached search failed (unexpected error): {e}")
                logger.info("Falling back to real-time search")
                # Fall back to real-time search
                use_cache = False
        
        if use_optimization:
            logger.info(f"Using optimized real-time search strategy with {model_type.upper()} engine")
            try:
                return self.optimized_strategy.search(user_image_path, discord_messages, progress_callback)
            except Exception as e:
                logger.warning(f"Optimized search failed, falling back to basic real-time: {e}")
                return self.realtime_strategy.search(user_image_path, discord_messages, progress_callback)
        else:
            logger.info(f"Using basic real-time search strategy with {model_type.upper()} engine")
            return self.realtime_strategy.search(user_image_path, discord_messages, progress_callback)
    
    def build_cache(self, discord_messages: List[DiscordMessage],
                   progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[bool, ProcessingStats]:
        """
        Build embedding cache for faster searches.
        
        Args:
            discord_messages: Discord messages to process
            progress_callback: Optional callback function (current, total, status)
            
        Returns:
            Tuple of (success, processing statistics)
        """
        model_type = self.config.clip.get_model_type(self.config.clip.model_name)
        logger.info(f"Building {model_type.upper()} cache for {len(discord_messages)} messages")
        return self.cache_manager.build_cache(discord_messages, progress_callback)
    
    def has_cache(self) -> bool:
        """Check if cache is available"""
        return self.cache_manager.has_valid_cache()
    
    def clear_cache(self) -> bool:
        """Clear cache"""
        try:
            self.cache_manager.clear_cache()
            model_type = self.config.clip.get_model_type(self.config.clip.model_name)
            logger.info(f"{model_type.upper()} cache cleared successfully")
            return True
        except Exception as e:
            model_type = self.config.clip.get_model_type(self.config.clip.model_name)
            logger.error(f"Failed to clear {model_type.upper()} cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        try:
            cache_stats = self.cache_manager.get_cache_stats()
            model_type = self.config.clip.get_model_type(self.config.clip.model_name)
            
            return {
                'current_engine': model_type,
                f'{model_type}_cache_available': cache_stats.get('status') == 'valid',
                f'{model_type}_cache_size_mb': cache_stats.get('embeddings_size_mb', 0),
                f'{model_type}_cache_images': cache_stats.get('total_images', 0),
                f'{model_type}_cache_created': cache_stats.get('created_at', 'Unknown'),
                f'{model_type}_cache_age_days': cache_stats.get('age_days', 0)
            }
            
        except Exception as e:
            model_type = self.config.clip.get_model_type(self.config.clip.model_name)
            logger.warning(f"Failed to get cache info: {e}")
            return {
                'current_engine': model_type,
                f'{model_type}_cache_available': False,
                'error': str(e)
            }
    
    def switch_model(self, new_model_name: str):
        """Switch model for all strategies"""
        logger.info(f"Switching SearchEngine to model: {new_model_name}")
        
        # Update config
        self.config.clip.model_name = new_model_name
        
        # Switch cache manager model
        self.cache_manager.switch_model(new_model_name)
        
        # Reinitialize strategies with new model
        self.optimized_strategy = OptimizedRealTimeSearchStrategy(self.config)
        self.realtime_strategy = RealTimeSearchStrategy(self.config) 
        self.cached_strategy = CachedSearchStrategy(self.config)
        
        logger.info(f"SearchEngine successfully switched to {new_model_name}")
    
    def get_all_cache_info(self) -> Dict[str, Any]:
        """Get cache information for all available models"""
        try:
            all_cache_status = self.cache_manager.get_all_model_cache_status()
            current_model = self.cache_manager.clip_engine.model_name
            current_engine = self.config.clip.get_model_type(current_model)
            
            return {
                'current_model': current_model,
                'all_models': all_cache_status,
                'current_engine': current_engine
            }
            
        except Exception as e:
            logger.warning(f"Failed to get all cache info: {e}")
            return {
                'current_model': self.cache_manager.clip_engine.model_name,
                'all_models': {},
                'current_engine': self.config.clip.get_model_type(self.config.clip.model_name),
                'error': str(e)
            }
    
    def clear_cache_for_model(self, model_name: str) -> bool:
        """Clear cache for specific model"""
        try:
            self.cache_manager.clear_cache_for_model(model_name)
            logger.info(f"Cache cleared for model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache for model {model_name}: {e}")
            return False

    def cleanup(self):
        """Properly cleanup SearchEngine and release GPU memory"""
        try:
            logger.info("Cleaning up SearchEngine resources")
            
            # Cleanup strategies
            strategies = [self.optimized_strategy, self.realtime_strategy, self.cached_strategy]
            for strategy in strategies:
                if hasattr(strategy, 'clip_engine'):
                    try:
                        strategy.clip_engine.cleanup()
                        logger.debug(f"Cleaned up engine for {strategy.__class__.__name__}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up strategy {strategy.__class__.__name__}: {e}")
            
            # Cleanup cache manager
            if hasattr(self, 'cache_manager') and self.cache_manager:
                try:
                    # Clear in-memory data
                    self.cache_manager._embeddings = None
                    self.cache_manager._metadata = None
                    
                    # Cleanup cache manager's engine
                    if hasattr(self.cache_manager, 'clip_engine'):
                        self.cache_manager.clip_engine.cleanup()
                        logger.debug("Cleaned up cache manager engine")
                        
                except Exception as e:
                    logger.warning(f"Error cleaning up cache manager: {e}")
            
            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.debug("GPU cache cleared")
            except Exception as e:
                logger.warning(f"Could not clear GPU cache: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("SearchEngine cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during SearchEngine cleanup: {e}")
            raise 