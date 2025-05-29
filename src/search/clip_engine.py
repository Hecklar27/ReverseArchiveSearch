"""
CLIP model integration for semantic image similarity.
"""

import clip
import torch
import logging
import numpy as np
from PIL import Image
from typing import Optional, List, Union
from pathlib import Path

from ..core.config import CLIPConfig

logger = logging.getLogger(__name__)

class CLIPEngine:
    """CLIP model wrapper for image encoding and similarity calculation"""
    
    def __init__(self, config: CLIPConfig):
        self.config = config
        self.model = None
        self.preprocess = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model with automatic device detection"""
        try:
            # Auto-detect device
            if self.config.device:
                self.device = self.config.device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading CLIP model '{self.config.model_name}' on device: {self.device}")
            
            # Load model and preprocessing
            self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
            
            # Log GPU information if available
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
                logger.info(f"Using GPU: {gpu_name} ({gpu_memory}GB VRAM)")
            else:
                logger.info("Using CPU for CLIP processing")
                
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise
    
    def encode_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode a single image using CLIP.
        
        Args:
            image: Image path, PIL Image, or image data
            
        Returns:
            Normalized image embedding as numpy array
        """
        try:
            # Load image if it's a path
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Preprocess and encode
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def encode_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Encode multiple images in a batch for better performance.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Array of normalized embeddings, shape (n_images, embedding_dim)
        """
        try:
            if not images:
                return np.array([])
            
            # Preprocess all images
            image_tensors = torch.stack([
                self.preprocess(img.convert('RGB')) for img in images
            ]).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Failed to encode image batch: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First image embedding
            embedding2: Second image embedding
            
        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            return float((similarity + 1) / 2)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def calculate_similarities_batch(self, query_embedding: np.ndarray, 
                                   embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between one query and multiple embeddings.
        
        Args:
            query_embedding: Query image embedding (1D array)
            embeddings: Database embeddings (2D array, shape: n_images x embedding_dim)
            
        Returns:
            Array of similarity scores
        """
        try:
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize database embeddings
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(embeddings_norm, query_embedding)
            
            # Convert to 0-1 range
            return (similarities + 1) / 2
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarities: {e}")
            return np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings"""
        return 512  # ViT-B/32 embedding dimension
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used"""
        return self.device == "cuda"
    
    def get_device_info(self) -> dict:
        """Get information about the current device"""
        info = {
            'device': self.device,
            'model_name': self.config.model_name,
            'gpu_available': torch.cuda.is_available()
        }
        
        if self.device == "cuda":
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory // 1024**3,
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated(0) // 1024**2
            })
        
        return info 