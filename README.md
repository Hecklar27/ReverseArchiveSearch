# Reverse Archive Search

## Quick Start
1. Install NVIDIA Cuda Toolkit (If you have an AMD card skip this step, will have to use CPU based processing)
2. Install dependencies: `python setup.py` (This takes a while so just leave it be if it seems stuck)
3. Run: `python main.py`
4. Select your query image
5. Select Discord mapart-archive channel HTML export file (NEEDS TO BE EXPORTED DAILY USING [DISCORDCHATEXPORTER](https://github.com/Tyrrrz/DiscordChatExporter))
6. **Important**: Click "Pre-process Archive" to build cache (one-time setup, needs to be cleared and updated when an updated HTML is selected if you want the latest results)
7. Click "Cached Search" for lightning-fast results!

## Performance Comparison
- **Real-time Search**: 2+ minutes (downloads images on-demand)
- **Cached Search**: 1-5 seconds (uses pre-computed embeddings)

### First Time Setup
1. Load your Discord HTML file (Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to export an HTML of the mapart-archive channel daily as links expire in 24 hours, takes a few minutes)
2. Click "Pre-process Archive" - this will:
   - Download all images from Discord
   - Generate CLIP embeddings for each image
   - Save to local cache (~300MB for 3,500 images)
   - Takes 30-120 seconds (one-time only per HTML)

### Subsequent Searches
- Click "Cached Search" for instant results
- Download HTML and reset cache every 24 hours 
- Results in seconds

### Cache Troubleshooting
If cached search fails and falls back to real-time search:

1. **Cache Missing**: Run "Pre-process Archive" first
2. **Cache Corrupted**: Clear cache and rebuild
3. **Cache Expired**: Clear cache and rebuild after selecting new mapart-archive HTML

**Note**: If you see `WARNING - Cached search failed, falling back to real-time`, your cache needs to be rebuilt.

## Discord Integration
- Results show clickable Discord links
- Direct navigation to original messages
- Full message context preserved 
