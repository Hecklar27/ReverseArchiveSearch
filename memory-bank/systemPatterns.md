# System Patterns: Reverse Archive Search

## Application Architecture

### Core Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │  Processing     │    │   Data Layer    │
│   (Tkinter)     │◄──►│   Engine        │◄──►│   (JSON/HTTP)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Image Input     │    │ CLIP Model      │    │ Discord JSON    │
│ File Browser    │    │ Similarity      │    │ Image URLs      │
│ Results Display │    │ Ranking         │    │ Message Data    │
│ Discord Links   │    │ Discord URLs    │    │ Deep Links      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Performance-Optimized Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │  Processing     │    │  Cache Layer    │
│   (Tkinter)     │◄──►│   Engine        │◄──►│  (Embeddings)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Search Modes    │    │ CLIP Model      │    │ Pre-computed    │
│ Real-time/Cached│    │ Similarity      │    │ Embeddings      │
│ Progress Bars   │    │ FAISS Search    │    │ Local Images    │
│ Discord Links   │    │ URL Generation  │    │ Message Metadata│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Design Patterns

### 1. Model-View-Controller (MVC)
- **Model**: Discord data, CLIP embeddings, search results, message metadata
- **View**: Tkinter GUI components, clickable Discord links
- **Controller**: Event handlers, search orchestration, link generation

### 2. Pipeline Pattern
```python
# Real-time Pipeline (MVP)
user_image → validate_format() → load_image() → encode_clip() → search_matches() → generate_discord_links()
discord_images → download_image() → validate_format() → encode_clip() → store_embedding()

# Optimized Pipeline (Phase 2)
# Pre-processing Phase:
discord_images → batch_download() → batch_encode() → save_embeddings() → cache_metadata()
# Search Phase:
user_image → encode_clip() → load_embeddings() → fast_similarity() → rank_results() → generate_discord_links()
```

### 3. Repository Pattern
- **DiscordRepository**: Handle JSON parsing and message extraction
- **ImageRepository**: Manage image downloads and caching
- **SearchRepository**: CLIP model interface and similarity search
- **EmbeddingRepository**: Cache management for pre-computed embeddings
- **LinkRepository**: Discord URL generation and validation

### 4. Strategy Pattern (Search Modes)
```python
class SearchStrategy:
    def search(self, user_image): pass

class RealTimeSearch(SearchStrategy):
    def search(self, user_image):
        # Download and process images on-demand
        results = self.process_images(user_image)
        return self.add_discord_links(results)
        
class CachedSearch(SearchStrategy):
    def search(self, user_image):
        # Use pre-computed embeddings
        results = self.fast_search(user_image)
        return self.add_discord_links(results)
```

## Discord Integration Patterns

### 1. URL Generation Pattern
```python
class DiscordLinkGenerator:
    BASE_URL = "https://discord.com/channels/349201680023289867/349277718954901514"
    
    def generate_message_link(self, message_id: str) -> str:
        return f"{self.BASE_URL}/{message_id}"
    
    def generate_result_with_link(self, search_result):
        search_result['discord_url'] = self.generate_message_link(search_result['message_id'])
        return search_result
```

### 2. Result Enhancement Pattern
```python
# Search Result Structure with Discord Integration
search_result = {
    "message_id": "1377373386183016660",
    "similarity_score": 0.89,
    "image_url": "https://cdn.discordapp.com/attachments/.../image.png",
    "discord_url": "https://discord.com/channels/349201680023289867/349277718954901514/1377373386183016660",
    "author": {
        "name": "hecklar27",
        "nickname": "Hecklar"
    },
    "content": "RWB Meetup\nmapped by Hecklar\n6x7 flat carpet",
    "timestamp": "2025-05-28T12:50:05.626-07:00"
}
```

### 3. UI Integration Pattern
```python
# Tkinter clickable link implementation
def create_clickable_link(parent, text, url):
    link_label = tk.Label(parent, text=text, fg="blue", cursor="hand2")
    link_label.bind("<Button-1>", lambda e: webbrowser.open(url))
    return link_label
```

## Performance Optimization Patterns

### 1. Embedding Cache Pattern
```python
# Cache Structure with Discord metadata
embeddings_cache = {
    "message_id": {
        "embedding": numpy_array,
        "metadata": {
            "message_id": "1377373386183016660",
            "author": message_data["author"],
            "content": message_data["content"],
            "timestamp": message_data["timestamp"],
            "discord_url": generated_url
        },
        "image_url": original_url,
        "cache_timestamp": cache_time
    }
}
```

### 2. Lazy Loading Pattern
- Load embeddings only when needed
- Generate Discord URLs on-demand
- Progressive result display
- Background pre-processing

### 3. Batch Processing Pattern
```python
# Process multiple images simultaneously
batch_size = 32  # GPU memory dependent
for batch in batches(discord_images, batch_size):
    embeddings = model.encode_batch(batch)
    enhanced_results = add_discord_links_batch(embeddings)
    cache.store_batch(enhanced_results)
```

## Data Flow

### Initialization Flow
1. Load Discord JSON file
2. Extract messages with attachments
3. Initialize CLIP model
4. Create logging infrastructure
5. **Check for existing embeddings cache**
6. **Validate Discord channel configuration**

### Real-time Search Flow (MVP)
1. User selects image file
2. Validate image format (PNG/JPEG)
3. Load and preprocess user image
4. Encode user image with CLIP
5. For each Discord message with attachments:
   - Download image (skip if expired)
   - Encode with CLIP
   - Calculate similarity score
   - **Generate Discord deep link**
6. Rank results by similarity
7. **Display top matches with clickable Discord links in GUI**

### Cached Search Flow (Optimized)
1. **Pre-processing Phase** (one-time):
   - Download all Discord images
   - Batch encode with CLIP
   - **Generate and cache Discord URLs**
   - Save embeddings to disk
2. **Search Phase** (near-instant):
   - Load user image and encode
   - Load cached embeddings
   - Calculate similarities (vectorized)
   - **Retrieve cached Discord URLs**
   - Rank and display results

## Advanced Performance Patterns

### 1. FAISS Integration (Phase 3)
```python
import faiss
# Ultra-fast similarity search with Discord metadata
index = faiss.IndexFlatIP(embedding_dim)
index.add(discord_embeddings)
distances, indices = index.search(user_embedding, k=10)

# Retrieve Discord URLs for top matches
results = []
for i, idx in enumerate(indices[0]):
    message_data = cached_metadata[idx]
    results.append({
        'similarity': distances[0][i],
        'discord_url': message_data['discord_url'],
        'metadata': message_data
    })
```

### 2. Incremental Updates
- Track new messages since last cache update
- Process only new images
- **Update Discord URL mappings**
- Maintain cache freshness

### 3. Memory-Mapped Files
- Large embedding arrays stored on disk
- **Discord metadata separately indexed**
- Access without loading entire dataset
- Reduced RAM usage

## Error Handling Strategy

### Graceful Degradation
- **Expired URLs**: Skip and log, continue processing
- **Invalid Formats**: Log error, continue with valid images
- **Network Issues**: Retry with timeout, then skip
- **CLIP Failures**: Log error, exclude from results
- **Cache Corruption**: Fallback to real-time processing
- **Discord URL Generation**: Fallback to displaying message ID only**

### Cache Management
- **Cache Validation**: Verify embedding integrity and Discord URL validity
- **Cache Rebuilding**: Re-process corrupted entries
- **Partial Cache**: Use available embeddings, process missing ones
- **URL Validation**: Verify Discord link format consistency

### Logging Levels
- **ERROR**: Failed downloads, invalid formats, Discord URL generation failures
- **WARN**: Expired URLs, skipped content, malformed message data
- **INFO**: Processing progress, results found, cache operations, Discord links generated
- **DEBUG**: Detailed CLIP operations, URL generation details (future)

## Extensibility Points

### Future Enhancements
- **Database**: Replace file-based cache with database
- **Threading**: Async image downloads and processing
- **Web Interface**: Replace Tkinter with web frontend
- **Batch Processing**: Multiple image search support
- **Cloud Storage**: Remote embedding cache
- **API Interface**: REST API for programmatic access
- **Advanced Discord Integration**: Rich embed previews, reaction counts
- **Multi-server Support**: Extend to other Discord servers 