# Active Context: MVP + Performance Optimization Strategy

## Current Focus
Building MVP with real-time search as foundation, then adding pre-computed embeddings with FAISS for maximum performance.

## Strategic Decision: Option A - Quick Win Approach
✅ **Chosen Strategy**: MVP first, then add caching with FAISS integration

### Phase 1: MVP Foundation (Week 1-2)
- **Real-time search** as core functionality and fallback
- **Functional GUI** with all basic features
- **Solid architecture** that supports performance upgrades

### Phase 2: Performance Layer (Week 2-3)
- **"Pre-process Archive" feature** - one-time setup
- **Embedding cache system** with persistent storage
- **Dual search modes**: Real-time vs Cached

### Phase 3: Maximum Performance (Week 3-4)
- **FAISS integration** for ultra-fast similarity search
- **Sub-second search times** (<1 second)
- **Optimized batch processing** and memory management

## Implementation Benefits
- **Risk Mitigation**: Working real-time fallback if caching fails
- **User Choice**: Quick search vs Fast search modes
- **Iterative Development**: Each phase delivers value
- **Performance Scaling**: From 2 minutes → 5 seconds → <1 second

## Immediate Goals
1. **Core Functionality**: Get basic real-time matching working with CLIP
2. **Clean Architecture**: Design for easy caching integration
3. **Dual-Mode GUI**: Interface that supports both search modes
4. **Error Handling**: Graceful failures with proper logging

## Development Approach
- **Foundation First**: Real-time search with solid patterns
- **Cache Integration**: Add pre-processing without breaking existing functionality
- **FAISS Optimization**: Drop-in replacement for similarity calculations
- **User Experience**: Progressive enhancement from working to blazing fast

## Key Technical Decisions

### MVP Phase
- **Real-time Processing**: Download and process during search
- **CLIP Model**: GPU-accelerated semantic similarity
- **Clean Separation**: Modular design for easy enhancement
- **Fallback Strategy**: Always maintain working real-time option

### Performance Phase
- **FAISS Integration**: Ultra-fast vector similarity search
- **Embedding Cache**: Persistent storage with metadata
- **Batch Processing**: Optimize initial pre-processing time
- **Incremental Updates**: Handle new Discord messages efficiently

## Architecture Strategy

### Search Strategy Pattern
```python
class SearchEngine:
    def __init__(self):
        self.realtime_search = RealTimeSearch()
        self.cached_search = CachedSearch()  # With FAISS
    
    def search(self, image, use_cache=True):
        if use_cache and self.has_cache():
            return self.cached_search.search(image)  # <1 second
        else:
            return self.realtime_search.search(image)  # 30s-2min
```

### Performance Targets
- **MVP (Real-time)**: 30 seconds - 2 minutes
- **Cached**: 1-5 seconds  
- **FAISS Optimized**: <1 second
- **Setup Time**: 30-120 minutes (one-time)

## Current Constraints
- **Foundation Priority**: Get real-time working first
- **UI Design**: Support both search modes elegantly
- **Storage**: Plan for ~300-500MB cache storage
- **Memory**: Handle large embedding arrays efficiently

## Success Criteria by Phase

### Phase 1 (MVP)
- ✅ Real-time search working end-to-end
- ✅ Handles 3,300+ Discord messages
- ✅ GPU acceleration functional
- ✅ Clean, extensible architecture

### Phase 2 (Caching)
- ✅ Pre-processing pipeline working
- ✅ Persistent embedding storage
- ✅ 5-10x performance improvement
- ✅ Dual-mode interface

### Phase 3 (FAISS)
- ✅ Sub-second search times
- ✅ Optimized memory usage
- ✅ Scalable to larger archives
- ✅ Production-ready performance

## Known Risks & Mitigations
- **Cache Corruption**: Real-time fallback always available
- **FAISS Dependencies**: Graceful fallback to basic similarity
- **Memory Issues**: FAISS memory mapping for large datasets
- **Performance Expectations**: Clear mode indicators in UI

## Next Immediate Steps
1. **Build MVP**: Focus on real-time search with clean architecture
2. **Design for Caching**: Repository pattern with pluggable backends
3. **FAISS Planning**: Research optimal FAISS configuration
4. **UI Design**: Interface supporting mode selection
5. **Testing Strategy**: Validate with real Discord data 