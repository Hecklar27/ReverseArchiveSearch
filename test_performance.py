#!/usr/bin/env python3
"""
Performance test script for CLIP engine optimizations
"""

import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_clip_performance():
    """Test CLIP engine performance with different configurations"""
    print("🚀 Testing CLIP Engine Performance Optimizations...")
    
    from core.config import Config
    from search.clip_engine import CLIPEngine
    
    # Test with ViT-B/32 (should be fast)
    print("\n📊 Testing ViT-B/32 Performance:")
    config = Config.load_default()
    config.clip.model_name = "ViT-B/32"
    
    # Verify map art detection is disabled
    print(f"   Map art detection: {'Enabled' if config.vision.enable_map_art_detection else 'Disabled'}")
    
    clip_engine = CLIPEngine(config)
    print(f"   Mixed precision: {'Enabled' if clip_engine.use_mixed_precision else 'Disabled'}")
    print(f"   Device: {clip_engine.device}")
    
    # Create test images
    test_images = []
    for i in range(16):  # Test with 16 images
        # Create random test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(img_array))
    
    # Time the batch encoding
    print(f"   Encoding {len(test_images)} test images...")
    start_time = time.time()
    
    embeddings = clip_engine.encode_images_batch(test_images)
    
    encode_time = time.time() - start_time
    print(f"   ✅ Batch encoding time: {encode_time:.2f} seconds")
    print(f"   ✅ Per image: {encode_time/len(test_images):.3f} seconds")
    print(f"   ✅ Generated {len(embeddings)} embeddings")
    
    # Test with ViT-L/14 for comparison
    print("\n📊 Testing ViT-L/14 Performance:")
    config.clip.model_name = "ViT-L/14"
    clip_engine_l14 = CLIPEngine(config)
    
    print(f"   Mixed precision: {'Enabled' if clip_engine_l14.use_mixed_precision else 'Disabled'}")
    print(f"   Device: {clip_engine_l14.device}")
    
    print(f"   Encoding {len(test_images)} test images...")
    start_time = time.time()
    
    embeddings_l14 = clip_engine_l14.encode_images_batch(test_images)
    
    encode_time_l14 = time.time() - start_time
    print(f"   ✅ Batch encoding time: {encode_time_l14:.2f} seconds")
    print(f"   ✅ Per image: {encode_time_l14/len(test_images):.3f} seconds")
    print(f"   ✅ Generated {len(embeddings_l14)} embeddings")
    
    # Performance comparison
    print(f"\n🏃 Performance Comparison:")
    speedup = encode_time_l14 / encode_time if encode_time > 0 else 1
    print(f"   ViT-B/32 is {speedup:.1f}x faster than ViT-L/14")
    
    if speedup >= 2.0:
        print("   🎯 Excellent! ViT-B/32 shows significant speed improvement")
    elif speedup >= 1.5:
        print("   ✅ Good! ViT-B/32 is noticeably faster")
    else:
        print("   ⚠️  ViT-B/32 should be significantly faster - check optimizations")
    
    return encode_time, encode_time_l14

if __name__ == "__main__":
    try:
        test_clip_performance()
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc() 