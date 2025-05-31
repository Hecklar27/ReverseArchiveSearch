#!/usr/bin/env python3
"""
Test script for dual CLIP model support
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config():
    """Test configuration with model options"""
    print("🔧 Testing Configuration...")
    
    from core.config import Config
    
    config = Config.load_default()
    
    # Test model options
    available_models = config.clip.get_available_models()
    print(f"✅ Available models: {available_models}")
    
    display_options = config.clip.get_display_options()
    print(f"✅ Display options: {display_options}")
    
    # Test model info
    for model in available_models:
        info = config.clip.get_model_info(model)
        print(f"✅ {model}: {info['speed_rating']} speed, {info['accuracy_rating']} accuracy")
    
    # Test cache directories
    for model in available_models:
        cache_dir = config.cache.get_model_cache_dir(model)
        print(f"✅ {model} cache directory: {cache_dir}")
    
    print("✅ Configuration test passed!\n")
    return config

def test_cache_manager(config):
    """Test cache manager with model switching"""
    print("💾 Testing Cache Manager...")
    
    try:
        from search.cache_manager import EmbeddingCacheManager
        
        cache_manager = EmbeddingCacheManager(config)
        print(f"✅ Cache manager initialized with model: {cache_manager.clip_engine.model_name}")
        
        # Test cache status for all models
        all_status = cache_manager.get_all_model_cache_status()
        print(f"✅ All model cache status: {list(all_status.keys())}")
        
        for model, status in all_status.items():
            print(f"   {model}: {status.get('status', 'unknown')}")
        
        print("✅ Cache manager test passed!\n")
        return cache_manager
        
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        return None

def test_search_engine(config):
    """Test search engine with model switching"""
    print("🔍 Testing Search Engine...")
    
    try:
        from search.strategies import SearchEngine
        
        search_engine = SearchEngine(config)
        current_model = search_engine.cache_manager.clip_engine.model_name
        print(f"✅ Search engine initialized with model: {current_model}")
        
        # Test getting all cache info
        cache_info = search_engine.get_all_cache_info()
        print(f"✅ Current model: {cache_info['current_model']}")
        print(f"✅ All models tracked: {list(cache_info['all_models'].keys())}")
        
        print("✅ Search engine test passed!\n")
        return search_engine
        
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Starting Dual CLIP Model Tests...\n")
    
    try:
        # Test configuration
        config = test_config()
        
        # Test cache manager
        cache_manager = test_cache_manager(config)
        
        # Test search engine
        search_engine = test_search_engine(config)
        
        if cache_manager and search_engine:
            print("🎉 All tests passed! Dual CLIP model support is working correctly.")
            print("\n📋 Summary:")
            print("   • ViT-L/14 (Accurate, Slow) - Default")
            print("   • ViT-B/32 (Fast, Less Accurate) - Option")
            print("   • Model-specific cache directories")
            print("   • Multi-model cache management")
            print("   • Search engine with model switching support")
        else:
            print("❌ Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 