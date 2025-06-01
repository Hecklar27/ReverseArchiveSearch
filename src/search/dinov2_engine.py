"""
DINOv2-based image similarity search engine for exact matching.
"""

import torch
import logging
import numpy as np
from PIL import Image
from typing import Optional, List, Union, Tuple
from pathlib import Path
import time
from transformers import AutoImageProcessor, AutoModel

from core.config import Config

logger = logging.getLogger(__name__)

class DINOv2Engine:
    """DINOv2-based image encoding and similarity search for exact matching"""
    
    def __init__(self, config: Config = None):
        """Initialize DINOv2 engine with given configuration"""
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
        
        # Map model names to HuggingFace model IDs
        self.model_mapping = {
            "DINOv2-Base": "facebook/dinov2-base",
            "DINOv2-Large": "facebook/dinov2-large",
            "DINOv2-Small": "facebook/dinov2-small"
        }
        
        # Get HuggingFace model ID
        hf_model_id = self.model_mapping.get(self.model_name, "facebook/dinov2-base")
        
        # Load DINOv2 model and processor
        logger.info(f"Loading DINOv2 model {hf_model_id} on device {self.device}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(hf_model_id)
            self.model = AutoModel.from_pretrained(hf_model_id)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded DINOv2 model {hf_model_id}")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model {hf_model_id}: {e}")
            raise
        
        # Get embedding dimension based on model
        model_info = config.clip.get_model_info(self.model_name)
        self.embedding_dim = model_info['embedding_dim'] if model_info else 768
        
        # Mixed precision support
        self.use_mixed_precision = False
        if self.device == "cuda" and self.clip_config.use_mixed_precision:
            try:
                # Test if mixed precision works with this setup
                logger.debug("Testing mixed precision compatibility...")
                test_tensor = torch.randn(1, 3, 224, 224, device=self.device)
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        inputs = {"pixel_values": test_tensor}
                        _ = self.model(**inputs)
                
                self.use_mixed_precision = True
                logger.info("Mixed precision enabled for DINOv2 model")
                
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
        
        # Get similarity threshold for exact matching
        model_info = config.clip.get_model_info(self.model_name)
        self.similarity_threshold = model_info.get("similarity_threshold", 0.90)
        
        logger.info(f"DINOv2 engine initialized successfully on {self.device}")
        logger.info(f"Exact matching threshold: {self.similarity_threshold}")
    
    def _apply_map_art_detection(self, pil_image: Image.Image) -> Image.Image:
        """Apply map art detection and cropping if enabled"""
        if not self.vision_config.enable_map_art_detection:
            return pil_image
            
        try:
            # Apply map art detection using process_image which returns PIL Images
            detected_images = self.map_art_detector.process_image(pil_image)
            
            if detected_images:
                # Use the first (largest) detected image
                return detected_images[0]
            else:
                # No map art detected, return original image
                logger.debug("No map art detected, using original image")
                return pil_image
                
        except Exception as e:
            logger.warning(f"Map art detection failed: {e}, using original image")
            return pil_image
    
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
            pil_image = self._apply_map_art_detection(pil_image)
            
            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                        # Use CLS token (first token) for global image representation
                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                        # Normalize embedding
                        cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)
                else:
                    outputs = self.model(**inputs)
                    # Use CLS token (first token) for global image representation
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    # Normalize embedding
                    cls_embedding = cls_embedding / cls_embedding.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                return cls_embedding.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Failed to encode image with DINOv2: {e}")
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
            # Apply map art detection if enabled
            processed_images = []
            if self.vision_config.enable_map_art_detection:
                logger.debug(f"Applying map art detection to {len(images)} images")
                for img in images:
                    processed_img = self._apply_map_art_detection(img.convert('RGB'))
                    processed_images.append(processed_img)
            else:
                processed_images = [img.convert('RGB') for img in images]
            
            # Process all images in batch
            inputs = self.processor(images=processed_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**inputs)
                        # Use CLS token for global representation
                        cls_embeddings = outputs.last_hidden_state[:, 0, :]
                        # Normalize embeddings
                        cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)
                else:
                    outputs = self.model(**inputs)
                    # Use CLS token for global representation
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                    # Normalize embeddings
                    cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)
                
                # Return as list of numpy arrays for compatibility
                embeddings = []
                for i in range(cls_embeddings.shape[0]):
                    embedding = cls_embeddings[i].cpu().numpy()
                    embeddings.append(embedding)
                
                # Clear memory efficiently
                del inputs
                del cls_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to encode image batch with DINOv2: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First image embedding (should already be normalized)
            embedding2: Second image embedding (should already be normalized)
            
        Returns:
            Cosine similarity score between 0 and 1 (higher = more similar)
        """
        try:
            # Calculate cosine similarity directly (embeddings are already normalized)
            similarity = np.dot(embedding1, embedding2)
            
            # Convert from [-1, 1] to [0, 1] range for consistency
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0  # Return minimum similarity on error
    
    def calculate_similarities_batch(self, query_embedding: np.ndarray, 
                                   embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between one query and multiple embeddings.
        
        Args:
            query_embedding: Query image embedding (1D array, already normalized)
            embeddings: Database embeddings (2D array, shape: n_images x embedding_dim, already normalized)
            
        Returns:
            Array of cosine similarity scores (0 to 1 range)
        """
        try:
            # Calculate cosine similarities directly (embeddings are already normalized)
            similarities = np.dot(embeddings, query_embedding)
            
            # Convert from [-1, 1] to [0, 1] range for consistency
            similarities = (similarities + 1.0) / 2.0
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarities: {e}")
            return np.array([])
    
    def filter_exact_matches(self, similarities: np.ndarray) -> np.ndarray:
        """
        Filter similarities to only include exact matches above threshold.
        
        Args:
            similarities: Array of similarity scores
            
        Returns:
            Array of similarity scores with low similarities set to 0
        """
        # Set similarities below threshold to 0 for exact matching
        filtered_similarities = np.where(similarities >= self.similarity_threshold, similarities, 0.0)
        
        # Count exact matches
        exact_matches = np.sum(filtered_similarities > 0)
        if exact_matches > 0:
            logger.info(f"Found {exact_matches} exact matches above threshold {self.similarity_threshold}")
        else:
            logger.info(f"No exact matches found above threshold {self.similarity_threshold}")
            # If no exact matches, still return top similarities but log it
            logger.info("Returning all similarities for ranking purposes")
            return similarities
            
        return filtered_similarities
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of DINOv2 embeddings based on model"""
        return self.embedding_dim
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used"""
        return self.device == "cuda"
    
    def get_device_info(self) -> dict:
        """Get device information for display"""
        try:
            info = {}
            
            if torch.cuda.is_available():
                # Get GPU info
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['cuda_status'] = f"{device_name}, {memory_gb:.0f}GB"
                info['device'] = f"DINOv2 {self.model_name} (CUDA)"
            else:
                info['cuda_status'] = "Not available"
                info['device'] = f"DINOv2 {self.model_name} (CPU)"
                
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
            return {
                'cuda_status': 'Error',
                'device': f"DINOv2 {self.model_name}",
                'cuda_error': str(e)
            } 