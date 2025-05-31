#!/usr/bin/env python3
"""
Debug script to test cache building with map art detection disabled.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config import Config
from search.cache_manager import EmbeddingCacheManager
from search.clip_engine import CLIPEngine
from data.discord_parser import DiscordParser

def test_cache_build_without_map_art():
    """Test cache building with map art detection disabled"""
    logger.info("Testing cache building with map art detection disabled...")
    
    # Create config with map art detection disabled
    config = Config()
    config.vision.enable_map_art_detection = False  # Disable map art detection
    
    logger.info(f"Map art detection enabled: {config.vision.enable_map_art_detection}")
    
    # Create cache manager (which will create its own CLIP engine)
    cache_manager = EmbeddingCacheManager(config)
    logger.info(f"Cache manager initialized with model: {cache_manager.clip_engine.model_name}")
    
    # Look for Discord HTML file
    html_files = list(Path(".").glob("*.html"))
    if not html_files:
        logger.error("No HTML files found in current directory")
        return False
    
    html_file = html_files[0]
    logger.info(f"Using Discord HTML file: {html_file}")
    
    # Parse Discord messages
    parser = DiscordParser(str(html_file))
    messages = parser.parse_messages()
    
    if not messages:
        logger.error("No messages found in HTML file")
        return False
    
    logger.info(f"Parsed {len(messages)} messages")
    
    # Count messages with images
    messages_with_images = [msg for msg in messages if msg.has_images()]
    logger.info(f"Found {len(messages_with_images)} messages with images")
    
    if not messages_with_images:
        logger.error("No messages with images found")
        return False
    
    # Limit to first few messages for testing
    test_messages = messages_with_images[:10]
    logger.info(f"Testing with first {len(test_messages)} messages with images")
    
    def progress_callback(current, total, status):
        logger.info(f"Progress: {current}/{total} - {status}")
    
    # Build cache
    logger.info("Starting cache build...")
    success, stats = cache_manager.build_cache(test_messages, progress_callback)
    
    if success:
        logger.info("✅ Cache build completed successfully!")
        logger.info(f"Processing time: {stats.processing_time_seconds:.2f}s")
        logger.info(f"Processed images: {stats.processed_images}")
        logger.info(f"Failed downloads: {stats.failed_downloads}")
        logger.info(f"Expired links: {stats.expired_links}")
    else:
        logger.error("❌ Cache build failed")
        logger.error(f"Processing time: {stats.processing_time_seconds:.2f}s")
        logger.error(f"Processed images: {stats.processed_images}")
        logger.error(f"Failed downloads: {stats.failed_downloads}")
        logger.error(f"Expired links: {stats.expired_links}")
    
    return success

if __name__ == "__main__":
    logger.info("Starting cache build debug test...")
    try:
        success = test_cache_build_without_map_art()
        if success:
            logger.info("Test completed successfully!")
        else:
            logger.error("Test failed!")
    except Exception as e:
        logger.error(f"Test crashed: {e}")
        import traceback
        traceback.print_exc() 