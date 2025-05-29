"""
Search strategy implementations using the Strategy pattern.
"""

import logging
import requests
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable
from pathlib import Path
from PIL import Image
from io import BytesIO

from ..data.models import DiscordMessage, SearchResult, ProcessingStats
from ..core.config import Config
from .clip_engine import CLIPEngine
from .image_downloader import ImageDownloader

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

class RealTimeSearchStrategy(SearchStrategy):
    """Real-time search strategy - downloads and processes images on demand"""
    
    def __init__(self, config: Config):
        self.config = config
        self.clip_engine = CLIPEngine(config.clip)
        self.image_downloader = ImageDownloader()
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage],
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform real-time search by downloading and processing Discord images on demand.
        """
        logger.info(f"Starting real-time search with {len(discord_messages)} messages")
        start_time = time.time()
        
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
        
        # Finalize statistics
        stats.processing_time_seconds = time.time() - start_time
        
        logger.info(f"Real-time search completed in {stats.processing_time_seconds:.2f}s")
        logger.info(f"Processed {stats.processed_images}/{stats.total_images} images successfully")
        logger.info(f"Found {len(results)} results")
        
        return results, stats

class CachedSearchStrategy(SearchStrategy):
    """Cached search strategy - uses pre-computed embeddings (Phase 2)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.clip_engine = CLIPEngine(config.clip)
        # TODO: Implement caching in Phase 2
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage],
               progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform cached search using pre-computed embeddings.
        
        Note: This is a placeholder for Phase 2 implementation.
        """
        logger.warning("Cached search not yet implemented - falling back to real-time search")
        
        # Fallback to real-time search for now
        realtime_strategy = RealTimeSearchStrategy(self.config)
        return realtime_strategy.search(user_image_path, discord_messages, progress_callback)

class SearchEngine:
    """Main search engine that orchestrates different search strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.realtime_strategy = RealTimeSearchStrategy(config)
        self.cached_strategy = CachedSearchStrategy(config)
        
    def search(self, user_image_path: Path, discord_messages: List[DiscordMessage], 
               use_cache: bool = False, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[List[SearchResult], ProcessingStats]:
        """
        Perform image search using the specified strategy.
        
        Args:
            user_image_path: Path to user's query image
            discord_messages: List of Discord messages to search
            use_cache: Whether to use cached search (Phase 2+)
            progress_callback: Optional callback function (current, total, status)
            
        Returns:
            Tuple of (search results, processing statistics)
        """
        
        if use_cache:
            logger.info("Using cached search strategy")
            return self.cached_strategy.search(user_image_path, discord_messages, progress_callback)
        else:
            logger.info("Using real-time search strategy")
            return self.realtime_strategy.search(user_image_path, discord_messages, progress_callback)
    
    def get_device_info(self) -> dict:
        """Get CLIP device information"""
        return self.realtime_strategy.clip_engine.get_device_info()
    
    def has_cache(self) -> bool:
        """Check if cached embeddings are available (Phase 2)"""
        # TODO: Implement cache detection in Phase 2
        return False 