#!/usr/bin/env python3
"""
HTML Parser for Discord Archive Files

Parses HTML export files from Discord to extract messages with fresh attachment URLs.
This solves the issue where JSON export URLs expire after 24 hours.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable
from bs4 import BeautifulSoup

from .models import DiscordMessage, DiscordAttachment, DiscordUser

logger = logging.getLogger(__name__)

class HTMLParser:
    """Parse Discord HTML archive files"""
    
    def __init__(self, html_file: Path):
        self.html_file = html_file
        
    def parse(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[DiscordMessage]:
        """Parse HTML file and return list of Discord messages"""
        try:
            logger.info(f"Parsing HTML file: {self.html_file}")
            
            if progress_callback:
                progress_callback(0, 100, "Loading HTML file...")
            
            with open(self.html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if progress_callback:
                progress_callback(10, 100, "Parsing HTML structure...")
            
            soup = BeautifulSoup(content, 'html.parser')
            
            if progress_callback:
                progress_callback(20, 100, "Finding message containers...")
            
            # Find all message containers
            message_containers = soup.find_all('div', {'class': 'chatlog__message-container'})
            logger.info(f"Found {len(message_containers)} message containers")
            
            total_containers = len(message_containers)
            messages = []
            
            for i, container in enumerate(message_containers):
                if progress_callback and i % 50 == 0:  # Update every 50 messages
                    progress = 20 + int((i / total_containers) * 75)  # 20-95% range
                    progress_callback(progress, 100, f"Parsing message {i+1}/{total_containers}")
                
                message = self._parse_message_container(container)
                if message:
                    messages.append(message)
            
            if progress_callback:
                progress_callback(100, 100, f"Completed! Found {len(messages)} messages with images")
            
            logger.info(f"Successfully parsed {len(messages)} messages with attachments")
            return messages
            
        except Exception as e:
            logger.error(f"Error parsing HTML file: {e}")
            raise
    
    def _parse_message_container(self, container) -> Optional[DiscordMessage]:
        """Parse a single message container"""
        try:
            # Get message ID
            message_id = container.get('data-message-id')
            if not message_id:
                return None
            
            # Find the main message div
            message_div = container.find('div', {'class': 'chatlog__message'})
            if not message_div:
                return None
            
            # Get author info
            author_element = message_div.find('span', {'class': 'chatlog__author'})
            if author_element:
                author_name = author_element.get('title', 'Unknown')
                author_id = author_element.get('data-user-id', 'unknown')
            else:
                author_name = 'Unknown'
                author_id = 'unknown'
            
            # Create author object
            author = DiscordUser(
                id=author_id,
                name=author_name,
                discriminator='0000'  # Default since not available in HTML
            )
            
            # Get timestamp
            timestamp_element = message_div.find('span', {'class': 'chatlog__timestamp'})
            timestamp_str = datetime.now().isoformat()  # Default
            if timestamp_element:
                timestamp_title = timestamp_element.get('title')
                if timestamp_title:
                    # Parse timestamp like "Friday, November 17, 2017 4:17 PM"
                    parsed_dt = self._parse_timestamp(timestamp_title)
                    timestamp_str = parsed_dt.isoformat()
            
            # Get message content
            content_element = message_div.find('div', {'class': 'chatlog__content'})
            content = ''
            if content_element:
                content = content_element.get_text(strip=True)
            
            # Get attachments
            attachments = self._parse_attachments(message_div)
            
            # Only return messages that have image attachments
            if not attachments:
                return None
            
            return DiscordMessage(
                id=message_id,
                type='Default',  # Default message type
                timestamp=timestamp_str,
                content=content,
                author=author,
                attachments=attachments,
                is_pinned=False
            )
            
        except Exception as e:
            logger.warning(f"Error parsing message container: {e}")
            return None
    
    def _parse_attachments(self, message_div) -> List[DiscordAttachment]:
        """Parse attachments from a message div"""
        attachments = []
        
        # Find attachment divs
        attachment_divs = message_div.find_all('div', {'class': 'chatlog__attachment'})
        
        for attachment_div in attachment_divs:
            # Find the attachment link
            link = attachment_div.find('a')
            if not link:
                continue
                
            url = link.get('href')
            if not url:
                continue
            
            # Find the image element for metadata
            img = attachment_div.find('img', {'class': 'chatlog__attachment-media'})
            if not img:
                continue
            
            # Get filename and size from title attribute
            title = img.get('title', '')
            filename = 'unknown.png'
            file_size_bytes = 0
            
            if title:
                # Parse title like "Image: Screen_Shot_2017-11-17_at_4.16.23_PM.png (59.26 KB)"
                match = re.search(r'Image: ([^(]+) \(([^)]+)\)', title)
                if match:
                    filename = match.group(1).strip()
                    size_str = match.group(2).strip()
                    # Convert size to bytes (rough approximation)
                    if 'KB' in size_str:
                        file_size_bytes = int(float(size_str.replace('KB', '').strip()) * 1024)
                    elif 'MB' in size_str:
                        file_size_bytes = int(float(size_str.replace('MB', '').strip()) * 1024 * 1024)
            
            # Filter for image files
            if not self._is_image_file(filename):
                continue
            
            attachment = DiscordAttachment(
                id=str(len(attachments)),  # Simple incrementing ID as string
                filename=filename,
                file_size_bytes=file_size_bytes,
                url=url
            )
            
            attachments.append(attachment)
        
        return attachments
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string into datetime object"""
        try:
            # Handle formats like "Friday, November 17, 2017 4:17 PM"
            # Remove day of week and try parsing
            clean_str = re.sub(r'^[A-Za-z]+,\s*', '', timestamp_str)
            
            # Try different formats
            formats = [
                '%B %d, %Y %I:%M %p',  # November 17, 2017 4:17 PM
                '%m/%d/%Y %I:%M %p',   # 11/17/2017 4:17 PM  
                '%Y-%m-%d %H:%M:%S',   # 2017-11-17 16:17:00
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(clean_str, fmt)
                except ValueError:
                    continue
            
            # Fallback
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return datetime.now()
            
        except Exception as e:
            logger.warning(f"Error parsing timestamp '{timestamp_str}': {e}")
            return datetime.now()
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if filename is an image file"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        return any(filename.lower().endswith(ext) for ext in image_extensions) 