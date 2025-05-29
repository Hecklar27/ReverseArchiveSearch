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
        """Initialize CLIP model with automatic device detection and fallback"""
        try:
            # Determine the best available device
            if self.config.device:
                # Use explicitly configured device
                requested_device = self.config.device
                if requested_device == "cuda" and not torch.cuda.is_available():
                    logger.warning(f"CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
                else:
                    self.device = requested_device
            else:
                # Auto-detect best device
                if torch.cuda.is_available():
                    # Verify CUDA actually works
                    try:
                        torch.cuda.current_device()
                        test_tensor = torch.tensor([1.0]).cuda()
                        self.device = "cuda"
                        logger.info("CUDA verification successful")
                    except Exception as e:
                        logger.warning(f"CUDA available but verification failed: {e}")
                        self.device = "cpu"
                else:
                    self.device = "cpu"
            
            logger.info(f"Loading CLIP model '{self.config.model_name}' on device: {self.device}")
            
            # Load model and preprocessing
            self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
            
            # Log device information
            if self.device == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_props = torch.cuda.get_device_properties(0)
                    gpu_memory_gb = gpu_props.total_memory // 1024**3
                    gpu_compute = f"{gpu_props.major}.{gpu_props.minor}"
                    
                    logger.info(f"🚀 Using GPU: {gpu_name}")
                    logger.info(f"   - Total VRAM: {gpu_memory_gb}GB")
                    logger.info(f"   - Compute Capability: {gpu_compute}")
                    logger.info(f"   - PyTorch CUDA version: {torch.version.cuda}")
                    
                    # Log current memory usage
                    torch.cuda.empty_cache()  # Clear cache for accurate reading
                    allocated = torch.cuda.memory_allocated(0) // 1024**2
                    cached = torch.cuda.memory_reserved(0) // 1024**2
                    logger.info(f"   - Memory allocated: {allocated}MB, reserved: {cached}MB")
                    
                except Exception as e:
                    logger.warning(f"Could not get detailed GPU info: {e}")
                    logger.info("Using CUDA for CLIP processing")
            else:
                logger.info("💻 Using CPU for CLIP processing")
                
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            # Try fallback to CPU if CUDA failed
            if self.device == "cuda":
                logger.info("Attempting fallback to CPU...")
                try:
                    self.device = "cpu"
                    self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
                    logger.info("Successfully fell back to CPU")
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
                    raise
            else:
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
        """Get comprehensive information about the current device"""
        info = {
            'device': self.device,
            'model_name': self.config.model_name,
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