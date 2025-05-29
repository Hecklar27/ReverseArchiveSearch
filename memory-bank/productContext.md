# Product Context: Reverse Archive Search

## Problem Statement
The 2b2t map art Discord channel contains thousands of messages with image attachments. Users frequently encounter map art images (taken at angles, zoomed out, or as photos of screens) and need to find the original Discord message that contained that artwork.

## User Journey
1. **Input**: User has an image (photo of map art, screenshot, etc.)
2. **Process**: App searches through ~3,300 Discord messages with attachments
3. **Analysis**: Uses semantic similarity (CLIP) to find matches despite angle/zoom differences
4. **Results**: Shows multiple potential matches with confidence scores
5. **Discovery**: User finds original message with artist, context, and community reactions
6. **Navigation**: **Click direct link to view original Discord message in context**

## Core Value Proposition
- **Flexible Matching**: Handles photos at angles, different zoom levels, lighting conditions
- **Community Discovery**: Connects users to original artists and discussions
- **Archive Navigation**: Makes massive Discord archives searchable by visual content
- **Direct Access**: **Seamless navigation to original Discord conversations**

## User Experience Goals
- **Simple Interface**: Single window, intuitive workflow
- **Visual Results**: Show matched images alongside message metadata
- **Transparency**: Clear confidence scores and multiple options
- **Reliability**: Graceful handling of broken links and unsupported formats
- **Direct Navigation**: **Clickable links to original Discord messages**

## Discord Integration Features

### Deep Link Navigation
- **Link Format**: `https://discord.com/channels/349201680023289867/349277718954901514/{MESSAGE_ID}`
- **Channel ID**: `349277718954901514` (2b2t Map Art Archive)
- **Server ID**: `349201680023289867` (Map Artists of 2b2t)
- **Dynamic URLs**: Replace `{MESSAGE_ID}` with actual message ID from JSON data

### Example Implementation
```python
# From messageExample.txt:
message_id = "1377373386183016660"
discord_url = f"https://discord.com/channels/349201680023289867/349277718954901514/{message_id}"
# Result: https://discord.com/channels/349201680023289867/349277718954901514/1377373386183016660
```

### User Benefits
- **Full Context**: See original message with all reactions, replies, and community discussion
- **Artist Attribution**: Direct access to creator and their other works
- **Community Engagement**: Join ongoing conversations about the artwork
- **Verification**: Confirm match accuracy by viewing original source

## Success Metrics
- Successfully matches photos of map art to original Discord messages
- Handles common image variations (angle, zoom, lighting)
- Processes 3,300+ message archive efficiently
- Provides meaningful results for legitimate queries
- **Enables seamless navigation to original Discord conversations** 