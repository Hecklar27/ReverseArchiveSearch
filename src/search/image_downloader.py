"""
Image downloading functionality with error handling for Discord CDN URLs.
"""

import requests
import logging
from typing import Optional, List, Tuple
from PIL import Image
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class ImageDownloader:
    """Handles downloading images from URLs with proper error handling"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3, max_workers: int = 8):
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.session = requests.Session()
        
        # Set a reasonable user agent
        self.session.headers.update({
            'User-Agent': 'ReverseArchiveSearch/1.0 (Educational/Research)'
        })
        
        # Thread-local storage for per-thread sessions
        self._local = threading.local()
        
        # Track statistics with thread safety
        self.expired_links_count = 0
        self.failed_downloads_count = 0
        self._stats_lock = threading.Lock()  # Thread lock for statistics
    
    def _get_session(self) -> requests.Session:
        """Get or create a session for the current thread"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                'User-Agent': 'ReverseArchiveSearch/1.0 (Educational/Research)'
            })
        return self._local.session
    
    def _increment_expired_links(self):
        """Thread-safe increment of expired links counter"""
        with self._stats_lock:
            self.expired_links_count += 1
    
    def _increment_failed_downloads(self):
        """Thread-safe increment of failed downloads counter"""
        with self._stats_lock:
            self.failed_downloads_count += 1
    
    def download_image(self, url: str) -> Optional[Image.Image]:
        """
        Download an image from a URL and return as PIL Image.
        
        Args:
            url: Image URL to download
            
        Returns:
            PIL Image object or None if download failed
        """
        return self._download_single_image(url, self._get_session())
    
    def _download_single_image(self, url: str, session: requests.Session) -> Optional[Image.Image]:
        """Internal method to download a single image with a specific session"""
        if not url:
            logger.warning("Empty URL provided")
            return None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Downloading image (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                response = session.get(url, timeout=self.timeout, stream=True)
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
                    self._increment_expired_links()
                    return None
                elif e.response.status_code == 403:
                    logger.warning(f"Access forbidden (403): {url}")
                    self._increment_failed_downloads()
                    return None
                else:
                    logger.warning(f"HTTP error {e.response.status_code} for {url}: {e}")
                    self._increment_failed_downloads()
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout downloading image: {url}")
                self._increment_failed_downloads()
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error downloading image: {url}")
                self._increment_failed_downloads()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error downloading image {url}: {e}")
                self._increment_failed_downloads()
                
            except Exception as e:
                logger.warning(f"Unexpected error downloading image {url}: {e}")
                self._increment_failed_downloads()
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.debug(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to download image after {self.max_retries} attempts: {url}")
        self._increment_failed_downloads()
        return None
    
    def download_images_batch(self, urls: List[str]) -> List[Tuple[str, Optional[Image.Image]]]:
        """
        Download multiple images concurrently.
        
        Args:
            urls: List of image URLs to download
            
        Returns:
            List of (url, image) tuples where image may be None if download failed
        """
        if not urls:
            return []
        
        logger.info(f"Starting concurrent download of {len(urls)} images with {self.max_workers} workers")
        
        def download_with_url(url: str) -> Tuple[str, Optional[Image.Image]]:
            session = self._get_session()
            image = self._download_single_image(url, session)
            return (url, image)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(download_with_url, urls))
        
        successful = sum(1 for _, img in results if img is not None)
        logger.info(f"Batch download complete: {successful}/{len(urls)} successful")
        
        return results

    def reset_stats(self):
        """Reset download statistics"""
        with self._stats_lock:
            self.expired_links_count = 0
            self.failed_downloads_count = 0

    def get_stats(self) -> dict:
        """Get download statistics"""
        with self._stats_lock:
            return {
                'expired_links': self.expired_links_count,
                'failed_downloads': self.failed_downloads_count,
                'total_downloads': self.expired_links_count + self.failed_downloads_count
            }

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