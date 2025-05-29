#!/usr/bin/env python3
"""
Simple test script to verify CLIP integration and GPU detection.
"""

import logging
from src.core.logger import setup_logging
from src.core.config import Config
from src.search.clip_engine import CLIPEngine

def test_clip_engine():
    """Test CLIP engine initialization and basic functionality"""
    
    # Setup logging
    setup_logging(log_level="INFO", log_to_file=False)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CLIP Engine Test")
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize CLIP engine
        logger.info("Initializing CLIP engine...")
        clip_engine = CLIPEngine(config.clip)
        
        # Show device information
        device_info = clip_engine.get_device_info()
        logger.info(f"Device Information:")
        for key, value in device_info.items():
            logger.info(f"  {key}: {value}")
        
        # Test basic functionality (if we have test images)
        logger.info(f"CLIP embedding dimension: {clip_engine.get_embedding_dimension()}")
        
        logger.info("CLIP Engine test completed successfully!")
        
    except Exception as e:
        logger.error(f"CLIP Engine test failed: {e}")
        raise

if __name__ == "__main__":
    test_clip_engine() 