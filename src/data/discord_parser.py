"""
Discord JSON parser for handling large archive files.
"""

import json
import logging
from pathlib import Path
from typing import List, Iterator, Dict, Any, Optional

from .models import DiscordMessage, DiscordUser, DiscordAttachment

logger = logging.getLogger(__name__)

class DiscordParser:
    """Parser for Discord channel export JSON files"""
    
    def __init__(self):
        self.stats = {
            'total_messages': 0,
            'messages_with_attachments': 0,
            'total_attachments': 0,
            'parse_errors': 0
        }
    
    def parse_file(self, json_file: Path) -> List[DiscordMessage]:
        """
        Parse a Discord JSON export file and return messages with image attachments.
        
        Args:
            json_file: Path to the Discord JSON export file
            
        Returns:
            List of DiscordMessage objects with image attachments
        """
        logger.info(f"Starting to parse Discord file: {json_file}")
        
        if not json_file.exists():
            raise FileNotFoundError(f"Discord JSON file not found: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return self._parse_messages(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file {json_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing Discord file {json_file}: {e}")
            raise
    
    def _parse_messages(self, data: Dict[str, Any]) -> List[DiscordMessage]:
        """Parse messages from loaded JSON data"""
        messages = []
        
        # Handle different possible JSON structures
        message_list = data.get('messages', [])
        if not message_list and isinstance(data, list):
            # If the root is directly a list of messages
            message_list = data
            
        logger.info(f"Found {len(message_list)} total messages in archive")
        
        for msg_data in message_list:
            try:
                message = self._parse_single_message(msg_data)
                if message and message.has_images():
                    messages.append(message)
                    self.stats['messages_with_attachments'] += 1
                    self.stats['total_attachments'] += len(message.get_image_attachments())
                
                self.stats['total_messages'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to parse message {msg_data.get('id', 'unknown')}: {e}")
                self.stats['parse_errors'] += 1
                continue
        
        logger.info(f"Parsed {len(messages)} messages with image attachments")
        logger.info(f"Statistics: {self.stats}")
        
        return messages
    
    def _parse_single_message(self, msg_data: Dict[str, Any]) -> Optional[DiscordMessage]:
        """Parse a single message from JSON data"""
        try:
            # Parse author
            author_data = msg_data.get('author', {})
            author = DiscordUser(
                id=author_data.get('id', ''),
                name=author_data.get('name', ''),
                discriminator=author_data.get('discriminator', '0000'),
                nickname=author_data.get('nickname'),
                color=author_data.get('color'),
                is_bot=author_data.get('isBot', False),
                avatar_url=author_data.get('avatarUrl')
            )
            
            # Parse attachments
            attachments = []
            for att_data in msg_data.get('attachments', []):
                attachment = DiscordAttachment(
                    id=att_data.get('id', ''),
                    url=att_data.get('url', ''),
                    filename=att_data.get('fileName', ''),
                    file_size_bytes=att_data.get('fileSizeBytes', 0)
                )
                attachments.append(attachment)
            
            # Create message
            message = DiscordMessage(
                id=msg_data.get('id', ''),
                type=msg_data.get('type', 'Default'),
                timestamp=msg_data.get('timestamp', ''),
                content=msg_data.get('content', ''),
                author=author,
                attachments=attachments,
                is_pinned=msg_data.get('isPinned', False),
                timestamp_edited=msg_data.get('timestampEdited')
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error parsing message data: {e}")
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get parsing statistics"""
        return self.stats.copy()
    
    def filter_by_date_range(self, messages: List[DiscordMessage], 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> List[DiscordMessage]:
        """
        Filter messages by date range (future enhancement).
        
        Args:
            messages: List of messages to filter
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            Filtered list of messages
        """
        # TODO: Implement date filtering in future versions
        logger.debug("Date filtering not yet implemented")
        return messages 