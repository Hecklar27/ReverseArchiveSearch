#!/usr/bin/env python3
"""
Test script to debug map art batch processing issues.
"""

import logging
import time
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Setup logging to see debug output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vision.map_art_detector import MapArtDetector

def create_test_image(width=800, height=600):
    """Create a simple test image"""
    # Create a random image to simulate a screenshot
    image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

def test_single_image():
    """Test single image processing"""
    logger.info("Testing single image processing...")
    
    detector = MapArtDetector(method="opencv", use_fast_detection=True)
    test_image = create_test_image()
    
    start_time = time.time()
    result = detector.process_image(test_image)
    end_time = time.time()
    
    logger.info(f"Single image processing took {end_time - start_time:.2f}s")
    logger.info(f"Result: {len(result)} cropped images")

def test_batch_processing():
    """Test batch processing"""
    logger.info("Testing batch processing...")
    
    detector = MapArtDetector(method="opencv", use_fast_detection=True)
    
    # Create a small batch of test images
    test_images = [create_test_image() for _ in range(3)]
    logger.info(f"Created {len(test_images)} test images")
    
    start_time = time.time()
    
    # This is where it might get stuck
    logger.info("Starting batch processing...")
    results = detector.process_images_batch(test_images)
    
    end_time = time.time()
    
    logger.info(f"Batch processing took {end_time - start_time:.2f}s")
    logger.info(f"Result: {len(results)} result lists")
    
    for i, result in enumerate(results):
        logger.info(f"Image {i}: {len(result)} cropped images")

def test_boundary_detection():
    """Test the boundary detection specifically"""
    logger.info("Testing boundary detection...")
    
    detector = MapArtDetector(method="opencv", use_fast_detection=True)
    test_image = create_test_image()
    
    # Convert to numpy and grayscale
    image_np = np.array(test_image.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    logger.info(f"Test image dimensions: {gray.shape}")
    
    # Test the fast detection directly
    start_time = time.time()
    regions = detector._detect_opencv_fast(gray)
    end_time = time.time()
    
    logger.info(f"Fast detection took {end_time - start_time:.2f}s")
    logger.info(f"Detected regions: {regions}")

if __name__ == "__main__":
    # Import cv2 here to avoid early import issues
    import cv2
    
    logger.info("Starting map art batch processing tests...")
    
    try:
        test_single_image()
        logger.info("Single image test completed successfully")
        
        test_boundary_detection()
        logger.info("Boundary detection test completed successfully")
        
        test_batch_processing()
        logger.info("Batch processing test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("All tests completed") 