"""
Discord data parsing for Reverse Archive Search application.
Now uses HTML parser for fresh URLs instead of expired JSON URLs.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from .models import DiscordMessage, DiscordAttachment
from .html_parser import HTMLParser

logger = logging.getLogger(__name__)

class DiscordParser:
    """Parse Discord archive data from HTML files"""
    
    def __init__(self, html_file_path: str):
        self.html_file_path = Path(html_file_path)
        if not self.html_file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_file_path}")
    
    def parse_messages(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> List[DiscordMessage]:
        """Parse Discord messages with image attachments from HTML file"""
        logger.info(f"Parsing Discord HTML archive: {self.html_file_path}")
        
        html_parser = HTMLParser(self.html_file_path)
        messages = html_parser.parse(progress_callback)
        
        logger.info(f"Loaded {len(messages)} messages with image attachments")
        return messages
    
    def get_all_attachments(self) -> List[DiscordAttachment]:
        """Get all image attachments from all messages"""
        messages = self.parse_messages()
        attachments = []
        
        for message in messages:
            attachments.extend(message.attachments)
        
        logger.info(f"Found {len(attachments)} total image attachments")
        return attachments 