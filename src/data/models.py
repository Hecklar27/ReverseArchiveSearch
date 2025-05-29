"""
Data models for Discord messages and search results.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class DiscordUser:
    """Represents a Discord user"""
    id: str
    name: str
    discriminator: str
    nickname: Optional[str] = None
    color: Optional[str] = None
    is_bot: bool = False
    avatar_url: Optional[str] = None

@dataclass
class DiscordAttachment:
    """Represents a Discord message attachment"""
    id: str
    url: str
    filename: str
    file_size_bytes: int

@dataclass
class DiscordMessage:
    """Represents a Discord message with metadata"""
    id: str
    type: str
    timestamp: str
    content: str
    author: DiscordUser
    attachments: List[DiscordAttachment]
    is_pinned: bool = False
    timestamp_edited: Optional[str] = None
    
    def get_image_attachments(self) -> List[DiscordAttachment]:
        """Get only image attachments from this message"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        return [
            att for att in self.attachments 
            if any(att.filename.lower().endswith(ext) for ext in image_extensions)
        ]
    
    def has_images(self) -> bool:
        """Check if message has any image attachments"""
        return len(self.get_image_attachments()) > 0

@dataclass
class SearchResult:
    """Represents a search result with similarity score"""
    message: DiscordMessage
    attachment: DiscordAttachment
    similarity_score: float
    discord_url: str
    
    def __lt__(self, other):
        """Enable sorting by similarity score (highest first)"""
        return self.similarity_score > other.similarity_score

@dataclass
class ProcessingStats:
    """Statistics about the processing operation"""
    total_messages: int = 0
    messages_with_images: int = 0
    total_images: int = 0
    processed_images: int = 0
    failed_downloads: int = 0
    expired_links: int = 0
    processing_time_seconds: float = 0.0
    
    def get_success_rate(self) -> float:
        """Calculate download success rate"""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images / self.total_images) * 100.0 