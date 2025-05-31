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

from core.config import Config

logger = logging.getLogger(__name__)

class CLIPEngine:
    """CLIP-based image encoding and similarity search with map art detection"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.clip_config = config.clip if config else Config().clip
        self.vision_config = config.vision if config else Config().vision
        
        # Device selection
        if self.clip_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.clip_config.device
        
        logger.info(f"Initializing CLIP on device: {self.device}")
        
        # Load CLIP model
        try:
            self.model, self.preprocess = clip.load(self.clip_config.model_name, device=self.device)
            self.model_name = self.clip_config.model_name
            logger.info(f"CLIP model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # Initialize map art detector if enabled
        self.map_art_detector = None
        if self.vision_config.enable_map_art_detection:
            try:
                from vision.map_art_detector import create_map_art_detector
                self.map_art_detector = create_map_art_detector(method=self.vision_config.detection_method)
                logger.info(f"Map art detector initialized with method: {self.vision_config.detection_method}")
            except ImportError:
                logger.warning("Map art detection module not available, falling back to full image processing")
            except Exception as e:
                logger.error(f"Failed to initialize map art detector: {e}")
        
        # Model dimensions (needed for validation)
        if self.model_name in ["ViT-L/14", "ViT-L/14@336px"]:
            self.embedding_dim = 768
        elif self.model_name in ["ViT-B/32", "ViT-B/16"]:
            self.embedding_dim = 512
        elif "RN" in self.model_name:  # ResNet variants
            self.embedding_dim = 1024 if "x64" in self.model_name else 2048
        else:
            self.embedding_dim = 512  # Default
    
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
            processed_images = []
            
            # Apply map art detection to each image if enabled
            for image in images:
                if self.vision_config.enable_map_art_detection:
                    cropped_images = self._crop_map_art(image)
                    
                    if cropped_images:
                        # Use the first (largest) detected region
                        processed_images.append(cropped_images[0])
                    elif self.vision_config.fallback_to_full_image:
                        # Fallback to full image if no map art detected
                        processed_images.append(image)
                    else:
                        # Skip image if no map art and fallback disabled
                        logger.warning("Skipping image: no map art detected and fallback disabled")
                        processed_images.append(None)
                else:
                    # No map art detection, use original image
                    processed_images.append(image)
            
            # Filter out None values (skipped images)
            valid_images = [img for img in processed_images if img is not None]
            
            if not valid_images:
                logger.warning("No valid images after map art processing")
                return []
            
            # Preprocess all valid images
            image_tensors = []
            for img in valid_images:
                tensor = self.preprocess(img.convert('RGB')).unsqueeze(0)
                image_tensors.append(tensor)
            
            # Batch encode
            batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                # Get features for all images
                image_features = self.model.encode_image(batch_tensor)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to list of numpy arrays
                embeddings = []
                for i in range(image_features.shape[0]):
                    embedding = image_features[i].cpu().numpy()
                    embeddings.append(embedding)
                
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