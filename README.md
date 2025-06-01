# Reverse Archive Search

## Quick Start
1. Install NVIDIA Cuda Toolkit (If you have an AMD card skip this step, will have to use CPU based processing)
   - the GPU detection is being funny, so if you installed CUDA and have an NVIDIA GPU, select PyTorch with CUDA (Option 2)
2. **Activate virtual environment**: `python -m venv .venv` then `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac) to enter the virtual enviroment (you will have (.venv) infront of your messages)
3. Install dependencies: `python setup.py` (This takes a while so just leave it be if it seems stuck)
4. Run: `python main.py`
5. Select your query image
6. Select Discord mapart-archive channel HTML export file (Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to export an HTML of the mapart-archive channel daily as links expire in 24 hours, takes a few minutes)
7. **Important**: Click "Pre-process Archive" to build cache (one-time setup, needs to be cleared and updated when an updated HTML is selected if you want the latest results)
8. Click "Cached Search" for lightning-fast results!

## Performance Comparison
- **Real-time Search**: 2+ minutes (downloads images on-demand)
- **Cached Search**: 1-5 seconds (uses pre-computed embeddings)

### Subsequent Searches after Caching
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


## Development & Testing

### Complete Environment Reset
For testing the setup process or troubleshooting installation issues, use the comprehensive cleanup script:

```bash
# Interactive cleanup (asks for confirmation)
python cleanup.py

# Force cleanup without prompts
python cleanup.py --force

# Selective cleanup options
python cleanup.py --keep-venv     # Keep virtual environment
python cleanup.py --keep-logs     # Keep log files  
python cleanup.py --keep-cache    # Keep application cache
python setup.py cleanup    # Remove core dependencies only
```

**What the cleanup script removes:**
- All Python dependencies (PyTorch, CLIP, etc.)
- Virtual environment (`.venv/`)
- Application cache and logs
- Build artifacts and temporary files
- Python cache files (`__pycache__`)
- IDE/editor configuration files

**After cleanup, reinstall from scratch:**
```bash
# 1. Create fresh virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# or
source .venv/bin/activate  # Linux/Mac

# 2. Run setup script
python setup.py
```
