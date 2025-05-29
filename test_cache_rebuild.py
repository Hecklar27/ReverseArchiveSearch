#!/usr/bin/env python3
"""
Test script to verify cache functionality and troubleshoot issues.
"""

import sys
from pathlib import Path
from src.core.config import Config
from src.search.strategies import SearchEngine

def main():
    print("=== Cache Rebuild Test ===")
    
    # Load config
    try:
        config = Config()
        print(f"✓ Config loaded successfully")
        print(f"  Cache directory: {config.cache.cache_dir}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return 1
    
    # Initialize search engine
    try:
        search_engine = SearchEngine(config)
        print(f"✓ Search engine initialized")
    except Exception as e:
        print(f"✗ Failed to initialize search engine: {e}")
        return 1
    
    # Check current cache status
    print("\n=== Current Cache Status ===")
    try:
        has_cache = search_engine.has_cache()
        print(f"Has valid cache: {has_cache}")
        
        cache_stats = search_engine.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
        
        if cache_stats.get('status') == 'valid':
            print("✓ Cache is valid and ready to use")
        elif cache_stats.get('status') == 'no_cache':
            print("ℹ No cache exists - needs to be built")
        else:
            print("⚠ Cache exists but is invalid/incompatible - needs rebuilding")
            
    except Exception as e:
        print(f"✗ Error checking cache status: {e}")
        return 1
    
    # Check cache compatibility in detail
    print("\n=== Cache Compatibility Check ===")
    try:
        cache_manager = search_engine.cache_manager
        
        # Check if cache files exist
        embeddings_path = cache_manager.cache_config.get_embeddings_path()
        metadata_path = cache_manager.cache_config.get_metadata_path()
        
        print(f"Embeddings file exists: {embeddings_path.exists()}")
        print(f"Metadata file exists: {metadata_path.exists()}")
        
        if metadata_path.exists():
            # Try to load metadata
            metadata = cache_manager._load_metadata()
            if metadata:
                print(f"Metadata loaded successfully")
                print(f"  Cache version: {getattr(metadata, 'cache_version', 'unknown')}")
                print(f"  Total images: {getattr(metadata, 'total_images', 'unknown')}")
                print(f"  CLIP model: {getattr(metadata, 'clip_model', 'unknown')}")
                print(f"  Created: {getattr(metadata, 'created_at', 'unknown')}")
                
                # Check compatibility
                is_compatible = cache_manager._validate_cache_compatibility(metadata)
                print(f"  Compatible: {is_compatible}")
                
                if not is_compatible:
                    print("  ⚠ Cache needs rebuilding due to incompatibility")
            else:
                print("✗ Failed to load metadata")
                
    except Exception as e:
        print(f"✗ Error during compatibility check: {e}")
    
    print("\n=== Recommendations ===")
    
    if not search_engine.has_cache():
        print("1. Run 'Pre-process Archive' in the GUI to build cache")
        print("2. Or clear cache and rebuild if you see compatibility issues")
    else:
        cache_stats = search_engine.get_cache_stats()
        if cache_stats.get('status') != 'valid':
            print("1. Click 'Clear Cache' in the GUI")
            print("2. Then click 'Pre-process Archive' to rebuild")
            print("3. This will fix compatibility issues")
        else:
            print("✓ Cache is working correctly!")
    
    print("\nTest completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 