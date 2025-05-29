# Reverse Archive Search

## Quick Start
1. Install dependencies: `python setup.py`
2. Run: `python main.py`
3. Select your query image
4. Select Discord JSON export file
5. **Important**: Click "Pre-process Archive" to build cache (one-time setup)
6. Click "Cached Search" for lightning-fast results!

## Performance Comparison
- **Real-time Search**: 2+ minutes (downloads images on-demand)
- **Cached Search**: 1-5 seconds (uses pre-computed embeddings)

## Cache System (Phase 2)
The cache system provides 10-60x performance improvement by pre-computing CLIP embeddings.

### First Time Setup
1. Load your Discord JSON file (This take all day to export, I have provided one which goes up to 5/29/25)
2. Click "Pre-process Archive" - this will:
   - Download all images from Discord
   - Generate CLIP embeddings for each image
   - Save to local cache (~300MB for 3,500 images)
   - Takes 30-120 minutes (one-time only)

### Subsequent Searches
- Click "Cached Search" for instant results
- No re-downloading or re-processing needed
- Results in 1-5 seconds

### Cache Troubleshooting
If cached search fails and falls back to real-time search:

1. **Cache Missing**: Run "Pre-process Archive" first
2. **Cache Corrupted**: Clear cache and rebuild
3. **Model Mismatch**: Cache was built with different CLIP model
4. **Cache Expired**: Automatic expiry after configured days

**Note**: If you see `WARNING - Cached search failed, falling back to real-time`, your cache needs to be rebuilt.

## Discord Integration
- Results show clickable Discord links
- Direct navigation to original messages
- Full message context preserved 
