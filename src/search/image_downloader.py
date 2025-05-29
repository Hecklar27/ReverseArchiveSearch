"""
Image downloading functionality with error handling for Discord CDN URLs.
"""

import requests
import logging
from typing import Optional
from PIL import Image
from io import BytesIO
import time

logger = logging.getLogger(__name__)

class ImageDownloader:
    """Handles downloading images from URLs with proper error handling"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set a reasonable user agent
        self.session.headers.update({
            'User-Agent': 'ReverseArchiveSearch/1.0 (Educational/Research)'
        })
    
    def download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download an image from a URL and return as PIL Image.
        
        Args:
            url: Image URL to download
            
        Returns:
            PIL Image object or None if download failed
        """
        if not url:
            logger.warning("Empty URL provided")
            return None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Downloading image (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    logger.warning(f"URL does not point to an image: {url} (content-type: {content_type})")
                    return None
                
                # Load image data
                image_data = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    image_data.write(chunk)
                image_data.seek(0)
                
                # Create PIL Image
                image = Image.open(image_data)
                
                # Verify image can be loaded
                image.verify()
                
                # Reload image for actual use (verify() consumes the image)
                image_data.seek(0)
                image = Image.open(image_data)
                
                logger.debug(f"Successfully downloaded image: {url}")
                return image
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.info(f"Image not found (404) - likely expired: {url}")
                    return None
                elif e.response.status_code == 403:
                    logger.warning(f"Access forbidden (403): {url}")
                    return None
                else:
                    logger.warning(f"HTTP error {e.response.status_code} for {url}: {e}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout downloading image: {url}")
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error downloading image: {url}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error downloading image {url}: {e}")
                
            except Exception as e:
                logger.warning(f"Unexpected error downloading image {url}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.debug(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to download image after {self.max_retries} attempts: {url}")
        return None
    
    def is_valid_image_url(self, url: str) -> bool:
        """
        Check if URL appears to be a valid image URL without downloading.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL appears to be an image
        """
        if not url:
            return False
        
        # Check for common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        url_lower = url.lower()
        
        # Handle Discord CDN URLs with parameters
        if '?' in url_lower:
            url_lower = url_lower.split('?')[0]
        
        return any(url_lower.endswith(ext) for ext in image_extensions)
    
    def get_stats(self) -> dict:
        """Get download statistics (placeholder for future implementation)"""
        return {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'timeout_errors': 0,
            'http_errors': 0
        } 