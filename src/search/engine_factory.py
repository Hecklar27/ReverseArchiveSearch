"""
Factory for creating the appropriate image similarity engine based on model type.
"""

import logging
from typing import Union
from core.config import Config
from .clip_engine import CLIPEngine
from .dinov2_engine import DINOv2Engine

logger = logging.getLogger(__name__)

def create_engine(config: Config = None) -> Union[CLIPEngine, DINOv2Engine]:
    """
    Create the appropriate engine based on model configuration.
    
    Args:
        config: Configuration object containing model settings
        
    Returns:
        Either CLIPEngine or DINOv2Engine instance
    """
    if config is None:
        config = Config()
    
    model_name = config.clip.model_name
    model_type = config.clip.get_model_type(model_name)
    
    logger.info(f"Creating {model_type} engine for model: {model_name}")
    
    if model_type == "dinov2":
        return DINOv2Engine(config)
    else:
        # Default to CLIP for backward compatibility
        return CLIPEngine(config)

class UniversalEngine:
    """
    Universal wrapper that provides a consistent interface for both CLIP and DINOv2 engines.
    This allows the rest of the codebase to use either engine transparently.
    """
    
    def __init__(self, config: Config = None):
        """Initialize with the appropriate engine based on configuration"""
        self.config = config or Config()
        self.engine = create_engine(config)
        self.model_name = self.engine.model_name
        self.device = self.engine.device
        self.embedding_dim = self.engine.embedding_dim
        
        # Determine if this is a DINOv2 engine for special handling
        self.is_dinov2 = isinstance(self.engine, DINOv2Engine)
        
        logger.info(f"UniversalEngine initialized with {type(self.engine).__name__}")
    
    def encode_image(self, image):
        """Encode a single image"""
        return self.engine.encode_image(image)
    
    def encode_images_batch(self, images):
        """Encode multiple images in batch"""
        return self.engine.encode_images_batch(images)
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate similarity between two embeddings"""
        return self.engine.calculate_similarity(embedding1, embedding2)
    
    def calculate_similarities_batch(self, query_embedding, embeddings):
        """Calculate similarities between query and multiple embeddings"""
        similarities = self.engine.calculate_similarities_batch(query_embedding, embeddings)
        
        # Apply exact matching filter for DINOv2
        if self.is_dinov2 and hasattr(self.engine, 'filter_exact_matches'):
            similarities = self.engine.filter_exact_matches(similarities)
            
        return similarities
    
    def get_embedding_dimension(self):
        """Get embedding dimension"""
        return self.engine.get_embedding_dimension()
    
    def is_gpu_available(self):
        """Check if GPU is available"""
        return self.engine.is_gpu_available()
    
    def get_device_info(self):
        """Get device information"""
        if hasattr(self.engine, 'get_device_info'):
            return self.engine.get_device_info()
        else:
            # Fallback for engines without get_device_info
            return {
                'device': f"{type(self.engine).__name__} {self.model_name}",
                'cuda_status': 'Available' if self.is_gpu_available() else 'Not available'
            }
    
    def cleanup(self):
        """Clean up engine resources"""
        if hasattr(self.engine, 'cleanup'):
            self.engine.cleanup()
        elif hasattr(self.engine, 'model'):
            # Fallback cleanup for engines without explicit cleanup method
            try:
                if hasattr(self.engine.model, 'cpu'):
                    self.engine.model = self.engine.model.cpu()
                if hasattr(self.engine, 'model'):
                    del self.engine.model
                    self.engine.model = None
                    
                import torch
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
                logger.info(f"Cleaned up {type(self.engine).__name__} resources")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
    
    def get_similarity_threshold(self):
        """Get similarity threshold for exact matching (DINOv2 only)"""
        if self.is_dinov2 and hasattr(self.engine, 'similarity_threshold'):
            return self.engine.similarity_threshold
        return None
    
    def get_model_type(self):
        """Get the type of model (clip or dinov2)"""
        return "dinov2" if self.is_dinov2 else "clip" 