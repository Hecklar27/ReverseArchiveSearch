# Technical Context: Reverse Archive Search

## Technology Stack

### Core Technologies
- **Python 3.8+**: Main application language
- **Tkinter**: GUI framework (built-in, cross-platform)
- **PyTorch**: Deep learning framework for CLIP
- **OpenAI CLIP**: Semantic image similarity model
- **Pillow (PIL)**: Image processing and format handling
- **Requests**: HTTP client for downloading Discord images
- **JSON**: Built-in library for Discord data parsing

### Performance Technologies (Phase 2-3)
- **FAISS**: Ultra-fast similarity search and clustering
- **SQLite**: Metadata and cache management
- **Pickle/Joblib**: Embedding serialization
- **NumPy**: Optimized array operations

### Image Processing Pipeline
```
# Phase 1: Real-time Pipeline
User Image → CLIP Encoding → Similarity Comparison → Ranked Results
Discord Images → Download → CLIP Encoding → Vector Database (in-memory)

# Phase 2-3: Optimized Pipeline  
Pre-processing: Discord Images → Batch Download → CLIP Encoding → FAISS Index
Search: User Image → CLIP Encoding → FAISS Search → Instant Results (<1s)
```

### Supported Formats
- **Input**: PNG, JPEG only
- **Processing**: Convert all to standard format for CLIP
- **Output**: Display original Discord image URLs

## 3-Phase Technical Strategy

### Phase 1: MVP Foundation
**Goal**: Working real-time search with clean architecture

#### Technical Approach
- **Search Strategy Pattern**: Pluggable search backends
- **Real-time Processing**: Download and process during search
- **GPU Acceleration**: Automatic CUDA detection
- **Modular Design**: Easy integration of future optimizations

#### Architecture
```python
class SearchEngine:
    def __init__(self):
        self.strategy = RealTimeSearchStrategy()
    
    def search(self, image):
        return self.strategy.search(image)
```

### Phase 2: Performance Layer  
**Goal**: 5-10x faster searches with caching

#### Technical Approach
- **Pre-computed Embeddings**: One-time processing, persistent storage
- **Dual-Mode Operation**: Real-time fallback + cached search
- **Batch Processing**: Optimized initial setup
- **Incremental Updates**: Handle new Discord messages

#### Architecture Enhancement
```python
class SearchEngine:
    def __init__(self):
        self.realtime_strategy = RealTimeSearchStrategy()
        self.cached_strategy = CachedSearchStrategy()
    
    def search(self, image, use_cache=True):
        if use_cache and self.has_cache():
            return self.cached_strategy.search(image)
        return self.realtime_strategy.search(image)
```

### Phase 3: FAISS Integration
**Goal**: Sub-second searches (<1 second)

#### Technical Approach
- **FAISS Library**: Ultra-optimized similarity search
- **Memory Mapping**: Handle large datasets efficiently  
- **Index Optimization**: Tuned for our specific use case
- **Fallback Chain**: FAISS → Basic similarity → Real-time

#### FAISS Configuration
```python
import faiss

# Optimal configuration for CLIP embeddings
dimension = 512  # CLIP ViT-B/32 embedding size
index = faiss.IndexFlatIP(dimension)  # Inner product similarity
# Alternative: IndexIVFFlat for larger datasets

# GPU acceleration if available
if faiss.get_num_gpus() > 0:
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
```

## Technical Decisions

### Image Matching Strategy
- **Primary**: OpenAI CLIP for semantic similarity
- **GPU Acceleration**: Automatic detection with CPU fallback
- **Phase 1**: Basic cosine similarity
- **Phase 2-3**: FAISS-optimized similarity search
- **Future Fallback**: OpenCV for traditional computer vision (planned)

### Performance Architecture
- **Phase 1**: Real-time processing (30s-2min)
- **Phase 2**: Cached embeddings (1-5 seconds)
- **Phase 3**: FAISS optimization (<1 second)
- **Fallback Strategy**: Graceful degradation across all phases

### Data Handling
- **Discord JSON**: Efficient parsing of large files (99MB+)
- **Image Downloads**: On-demand fetching with error handling
- **Embedding Storage**: Persistent cache with metadata
- **Memory Management**: FAISS memory mapping for large datasets

### Architecture Approach
- **Modular Design**: SearchStrategy pattern for easy enhancement
- **Progressive Enhancement**: Each phase adds capability
- **Always Functional**: Real-time fallback always available
- **User Choice**: Mode selection based on performance needs

## Dependencies by Phase

### Phase 1 (MVP)
```
torch>=1.9.0
torchvision>=0.10.0
clip-by-openai
Pillow>=8.0.0
requests>=2.25.0
numpy>=1.20.0
```

### Phase 2 (Caching)
```
# Add to Phase 1:
sqlite3  # Built-in Python
pickle   # Built-in Python
joblib>=1.0.0  # Alternative serialization
```

### Phase 3 (FAISS)
```
# Add to Phase 2:
faiss-cpu>=1.7.0    # CPU version
# OR
faiss-gpu>=1.7.0    # GPU version (if CUDA available)
```

## Hardware Requirements

### Minimum (Phase 1 - CPU only)
- **RAM**: 4GB+ available
- **Processing**: Modern multi-core CPU
- **Time**: 3-10 minutes for full archive search

### Recommended (All Phases - GPU)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (GTX 1060+ or equivalent)
- **RAM**: 8GB+ available
- **CUDA**: Compatible PyTorch installation
- **Time**: 
  - Phase 1: 30 seconds - 2 minutes
  - Phase 2: 1-5 seconds  
  - Phase 3: <1 second

### Optimal (Phase 3 - High-end)
- **GPU**: RTX 3070+ or equivalent (8GB+ VRAM)
- **RAM**: 16GB+ available
- **Storage**: SSD for faster embedding I/O
- **Time**: Consistent <1 second searches

## Performance Optimization Techniques

### FAISS Optimization
- **Index Selection**: IndexFlatIP for exact search, IndexIVFFlat for approximate
- **GPU Acceleration**: Automatic GPU usage when available
- **Memory Mapping**: Handle datasets larger than RAM
- **Batch Search**: Process multiple queries simultaneously

### Memory Management
- **Embedding Compression**: Float16 instead of Float32 where appropriate
- **Lazy Loading**: Load embeddings on-demand
- **Cache Partitioning**: Split large caches into manageable chunks

### Network Optimization
- **Connection Pooling**: Reuse HTTP connections for Discord downloads
- **Parallel Downloads**: Multiple concurrent image downloads
- **Retry Logic**: Exponential backoff for failed requests

## Development Constraints
- **Cross-platform**: Windows primary, but maintain Linux/Mac compatibility
- **Graceful Degradation**: Always maintain working functionality
- **User Choice**: Clear mode selection and performance expectations
- **Self-contained**: Minimal external dependencies 