"""
CLIP-based image similarity search engine with map art detection.
"""

import clip
import torch
import logging
import numpy as np
from PIL import Image
from typing import Optional, List, Union, Tuple
from pathlib import Path
import time

# Try to import version-compatible autocast
try:
    # New API (PyTorch 1.10+)
    from torch.amp import autocast
    USE_NEW_AUTOCAST = True
except ImportError:
    # Fallback to old API
    from torch.cuda.amp import autocast as cuda_autocast
    USE_NEW_AUTOCAST = False

from core.config import Config

logger = logging.getLogger(__name__)

class CLIPEngine:
    """CLIP-based image encoding and similarity search with map art detection"""
    
    def __init__(self, config: Config = None):
        """Initialize CLIP engine with given configuration"""
        if config is None:
            config = Config()
        
        self.clip_config = config.clip
        self.vision_config = config.vision
        
        # Set model name from config
        self.model_name = self.clip_config.model_name
        
        # Device detection based on config
        if self.clip_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.clip_config.device
            
        # Ensure device is valid
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        # Load CLIP model
        logger.info(f"Loading CLIP model {self.model_name} on device {self.device}")
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            logger.info(f"Successfully loaded CLIP model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model {self.model_name}: {e}")
            raise
        
        # Get embedding dimension based on model
        model_info = config.clip.get_model_info(self.model_name)
        self.embedding_dim = model_info['embedding_dim'] if model_info else 512
        
        # Mixed precision support
        self.use_mixed_precision = False
        if self.device == "cuda" and self.clip_config.use_mixed_precision:
            try:
                # Test if mixed precision works with this setup
                logger.debug("Testing mixed precision compatibility...")
                test_tensor = torch.randn(1, 3, 224, 224, device=self.device)
                
                with torch.no_grad():
                    with self._get_autocast_context():
                        _ = self.model.encode_image(test_tensor)
                
                self.use_mixed_precision = True
                logger.info("Mixed precision enabled for CLIP model")
                
            except Exception as e:
                logger.warning(f"Mixed precision not available: {e}")
                self.use_mixed_precision = False
            finally:
                # Always clean up test tensor
                try:
                    del test_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
        
        # Initialize map art detector if enabled
        self.map_art_detector = None
        if self.vision_config.enable_map_art_detection:
            try:
                from vision.map_art_detector import create_map_art_detector
                self.map_art_detector = create_map_art_detector(
                    method=self.vision_config.detection_method,
                    use_fast_detection=self.vision_config.use_fast_detection
                )
                logger.info(f"Map art detector initialized with method: {self.vision_config.detection_method}, "
                           f"fast mode: {self.vision_config.use_fast_detection}")
            except ImportError:
                logger.warning("Map art detection module not available, disabling map art detection")
                self.vision_config.enable_map_art_detection = False
            except Exception as e:
                logger.warning(f"Failed to initialize map art detector: {e}")
                logger.warning("Disabling map art detection")
                self.vision_config.enable_map_art_detection = False
        
        logger.info(f"CLIP engine initialized successfully on {self.device}")
    
    def _get_autocast_context(self):
        """Get the appropriate autocast context for the current PyTorch version"""
        if self.device == "cuda" and self.use_mixed_precision:
            if USE_NEW_AUTOCAST:
                return autocast('cuda')
            else:
                return cuda_autocast()
        else:
            # Return a dummy context manager that does nothing
            from contextlib import nullcontext
            return nullcontext()
    
    def _crop_map_art(self, image: Image.Image) -> List[Image.Image]:
        """
        Detect and crop map art from the image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of cropped map art images (or original image if no detection)
        """
        if self.map_art_detector is None:
            return [image]
        
        try:
            # Process image to detect and crop map art
            cropped_images = self.map_art_detector.process_image(image)
            
            if cropped_images:
                logger.debug(f"Detected {len(cropped_images)} map art regions")
                return cropped_images
            else:
                # No map art detected
                if self.vision_config.fallback_to_full_image:
                    logger.debug("No map art detected, using full image")
                    return [image]
                else:
                    logger.debug("No map art detected, skipping image")
                    return []
                    
        except Exception as e:
            logger.warning(f"Map art detection failed: {e}, using full image")
            return [image]
    
    def _crop_map_art_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Detect and crop map art from multiple images in batch (optimized for cache building).
        
        Args:
            images: List of input PIL Images
            
        Returns:
            List of processed images (cropped map art or original if no detection/fallback)
        """
        if self.map_art_detector is None:
            return images
        
        if not images:
            return []
        
        try:
            # Use batch processing for significant speedup
            logger.debug(f"Processing {len(images)} images with batch map art detection")
            start_time = time.time()
            
            # Add timeout protection for batch processing
            timeout_seconds = 30.0  # Maximum 30 seconds for entire batch
            
            # Process all images in a single batch call
            batch_results = self.map_art_detector.process_images_batch(images)
            
            detection_time = time.time() - start_time
            
            # Check if processing took too long (potential hang)
            if detection_time > timeout_seconds:
                logger.warning(f"Map art batch detection took {detection_time:.2f}s (>{timeout_seconds}s), disabling for this session")
                # Disable map art detection for this session to prevent further hangs
                self.vision_config.enable_map_art_detection = False
                self.map_art_detector = None
                return images
            
            logger.debug(f"Batch map art detection completed in {detection_time:.2f}s for {len(images)} images")
            
            processed_images = []
            
            for i, (original_image, cropped_images) in enumerate(zip(images, batch_results)):
                if cropped_images:
                    # Use the first (largest) detected region
                    processed_images.append(cropped_images[0])
                    logger.debug(f"Image {i}: detected {len(cropped_images)} map art regions, using largest")
                else:
                    # No map art detected
                    if self.vision_config.fallback_to_full_image:
                        processed_images.append(original_image)
                        logger.debug(f"Image {i}: no map art detected, using full image")
                    else:
                        logger.debug(f"Image {i}: no map art detected, skipping image")
                        # Skip this image by not adding to processed_images
                        continue
            
            logger.debug(f"Batch map art processing: {len(processed_images)}/{len(images)} images processed")
            return processed_images
                    
        except Exception as e:
            logger.warning(f"Batch map art detection failed: {e}, falling back to original images")
            # On any error, disable map art detection for this session and return original images
            logger.warning("Disabling map art detection for this session due to errors")
            self.vision_config.enable_map_art_detection = False
            self.map_art_detector = None
            return images
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode a single image into a feature vector with optional map art detection.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Normalized feature vector as numpy array
        """
        try:
            # Load image if it's a path
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')
            
            # Apply map art detection and cropping if enabled
            if self.vision_config.enable_map_art_detection:
                cropped_images = self._crop_map_art(pil_image)
                
                if not cropped_images:
                    # No map art detected and fallback disabled
                    logger.warning("No map art detected and fallback disabled")
                    return np.zeros(self.embedding_dim)
                
                # Use the first (largest) detected map art region
                pil_image = cropped_images[0]
                
                if len(cropped_images) > 1:
                    logger.debug(f"Multiple map art regions detected, using largest one")
            
            # Preprocess and encode
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.use_mixed_precision:
                    with self._get_autocast_context():
                        # Get image features
                        image_features = self.model.encode_image(image_tensor)
                        
                        # Normalize features
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                else:
                    # Get image features
                    image_features = self.model.encode_image(image_tensor)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                return image_features.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def encode_images_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """
        Encode multiple images in batch with optional map art detection.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of normalized feature vectors as numpy arrays
        """
        if not images:
            return []
        
        try:
            # Fast path: if map art detection is disabled, use efficient batch processing
            if not self.vision_config.enable_map_art_detection:
                # Use the original fast method - direct torch.stack
                image_tensors = torch.stack([
                    self.preprocess(img.convert('RGB')) for img in images
                ]).to(self.device)
                
                with torch.no_grad():
                    # Skip mixed precision for smaller models as it adds overhead
                    if self.use_mixed_precision and self.model_name in ["ViT-L/14", "ViT-L/14@336px"]:
                        with self._get_autocast_context():
                            image_features = self.model.encode_image(image_tensors)
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    else:
                        # Direct processing for better performance
                        image_features = self.model.encode_image(image_tensors)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Return as list of numpy arrays for compatibility
                    embeddings = []
                    for i in range(image_features.shape[0]):
                        embedding = image_features[i].cpu().numpy()
                        embeddings.append(embedding)
                    
                    # Clear memory efficiently
                    del image_tensors
                    del image_features
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    return embeddings
            
            # OPTIMIZED PATH: Map art detection enabled - use batch processing
            logger.debug(f"Starting batch map art detection for {len(images)} images")
            map_art_start_time = time.time()
            
            # Process all images with batch map art detection (MAJOR OPTIMIZATION)
            processed_images = self._crop_map_art_batch(images)
            
            map_art_time = time.time() - map_art_start_time
            logger.debug(f"Batch map art detection completed in {map_art_time:.2f}s - processed {len(processed_images)}/{len(images)} images")
            
            if not processed_images:
                logger.warning("No valid images after batch map art processing")
                return []
            
            # Now encode all processed images efficiently
            logger.debug(f"Encoding {len(processed_images)} processed images with CLIP")
            clip_start_time = time.time()
            
            image_tensors = torch.stack([
                self.preprocess(img.convert('RGB')) for img in processed_images
            ]).to(self.device)
            
            with torch.no_grad():
                # Use mixed precision only for large models
                if self.use_mixed_precision and self.model_name in ["ViT-L/14", "ViT-L/14@336px"]:
                    with self._get_autocast_context():
                        image_features = self.model.encode_image(image_tensors)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                else:
                    image_features = self.model.encode_image(image_tensors)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to list of numpy arrays
                embeddings = []
                for i in range(image_features.shape[0]):
                    embedding = image_features[i].cpu().numpy()
                    embeddings.append(embedding)
                
                # Clear memory efficiently
                del image_tensors
                del image_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                clip_time = time.time() - clip_start_time
                total_time = map_art_time + clip_time
                logger.debug(f"CLIP encoding completed in {clip_time:.2f}s - Total batch time: {total_time:.2f}s (map art: {map_art_time:.2f}s, CLIP: {clip_time:.2f}s)")
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to encode image batch: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First image embedding (should already be normalized)
            embedding2: Second image embedding (should already be normalized)
            
        Returns:
            Cosine similarity score between -1 and 1 (higher = more similar)
        """
        try:
            # Calculate cosine similarity directly (embeddings are already normalized from CLIP)
            similarity = np.dot(embedding1, embedding2)
            
            # Return raw cosine similarity (-1 to 1 range)
            # This provides better discrimination between similar/dissimilar images
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return -1.0  # Return minimum similarity on error
    
    def calculate_similarities_batch(self, query_embedding: np.ndarray, 
                                   embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between one query and multiple embeddings.
        
        Args:
            query_embedding: Query image embedding (1D array, already normalized)
            embeddings: Database embeddings (2D array, shape: n_images x embedding_dim, already normalized)
            
        Returns:
            Array of cosine similarity scores (-1 to 1 range)
        """
        try:
            # Calculate cosine similarities directly (embeddings are already normalized from CLIP)
            similarities = np.dot(embeddings, query_embedding)
            
            # Return raw cosine similarities for better discrimination
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarities: {e}")
            return np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings based on model"""
        return self.embedding_dim
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used"""
        return self.device == "cuda"
    
    def get_device_info(self) -> dict:
        """Get comprehensive information about the current device"""
        info = {
            'device': self.device,
            'model_name': self.clip_config.model_name,
            'gpu_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            try:
                info.update({
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory // 1024**3,
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated(0) // 1024**2,
                    'gpu_memory_reserved_mb': torch.cuda.memory_reserved(0) // 1024**2,
                    'cuda_version': torch.version.cuda,
                    'compute_capability': f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
                })
                
                # Check if we're actually using CUDA
                info['using_cuda'] = self.device == "cuda"
                
                if self.device != "cuda":
                    info['cuda_status'] = "Available but not used (check configuration or fallback occurred)"
                else:
                    info['cuda_status'] = "Active"
                    
            except Exception as e:
                info.update({
                    'cuda_error': str(e),
                    'cuda_status': "Error retrieving info"
                })
        else:
            info.update({
                'cuda_status': "Not available",
                'using_cuda': False
            })
        
        return info 