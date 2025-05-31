"""
AI-powered map art detection and cropping for Minecraft screenshots.

This module provides multiple approaches to detect and extract map artwork
from Minecraft screenshots before CLIP similarity search.
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageDraw
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class MapArtDetector:
    """Detects and crops map art from Minecraft screenshots"""
    
    def __init__(self, method: str = "opencv", confidence_threshold: float = 0.5):
        """
        Initialize map art detector.
        
        Args:
            method: Detection method ('opencv', 'yolo', 'segment', 'hybrid')
            confidence_threshold: Minimum confidence for detections
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Initialize the selected detection method
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the selected detection method"""
        if self.method == "opencv":
            logger.info("Using OpenCV-based map frame detection")
            # No model loading needed for OpenCV
            
        elif self.method == "yolo":
            logger.info("Initializing YOLO object detection for map frames")
            self._initialize_yolo()
            
        elif self.method == "segment":
            logger.info("Initializing semantic segmentation for map art")
            self._initialize_segmentation()
            
        elif self.method == "hybrid":
            logger.info("Initializing hybrid detection approach")
            self._initialize_hybrid()
            
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _initialize_yolo(self):
        """Initialize YOLO model for object detection"""
        try:
            # Try to use YOLOv8 via ultralytics
            from ultralytics import YOLO
            
            # Use a general object detection model and fine-tune for rectangular objects
            self.model = YOLO('yolov8n.pt')  # Nano model for speed
            logger.info("YOLO model loaded successfully")
            
        except ImportError:
            logger.warning("ultralytics not available, falling back to OpenCV")
            self.method = "opencv"
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.info("Falling back to OpenCV method")
            self.method = "opencv"
    
    def _initialize_segmentation(self):
        """Initialize semantic segmentation model"""
        try:
            # Use a pre-trained segmentation model
            import torchvision.transforms as transforms
            from torchvision.models import segmentation
            
            # Load a pre-trained DeepLabV3 model
            self.model = segmentation.deeplabv3_resnet50(pretrained=True)
            self.model.eval()
            
            # Define preprocessing transforms
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Segmentation model loaded successfully")
            
        except ImportError:
            logger.warning("torchvision not available, falling back to OpenCV")
            self.method = "opencv"
            
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            logger.info("Falling back to OpenCV method")
            self.method = "opencv"
    
    def _initialize_hybrid(self):
        """Initialize hybrid approach combining multiple methods"""
        # Start with OpenCV as base, then enhance with AI if available
        self.method = "opencv"
        logger.info("Hybrid approach initialized with OpenCV base")
    
    def detect_map_art(self, image: Union[Image.Image, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Detect map art regions in the image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            List of bounding boxes (x, y, width, height) for detected map art
        """
        if isinstance(image, Image.Image):
            # Convert PIL to numpy
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
        
        if self.method == "opencv":
            return self._detect_opencv(image_np)
        elif self.method == "yolo":
            return self._detect_yolo(image_np)
        elif self.method == "segment":
            return self._detect_segmentation(image_np)
        elif self.method == "hybrid":
            return self._detect_hybrid(image_np)
        else:
            return []
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect map content using outside-in boundary detection with texture analysis.
        
        This method starts from the center region and progressively finds the boundaries
        of detailed map artwork vs pixelated Minecraft blocks.
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            center_x, center_y = width // 2, height // 2
            
            logger.debug(f"Image dimensions: {width}x{height}, center: ({center_x}, {center_y})")
            
            # Step 1: Find the general center region where maps typically appear
            initial_region = self._find_center_region(gray)
            if initial_region is None:
                logger.debug("No suitable center region found")
                return []
            
            # Step 2: Use outside-in boundary detection to find precise map edges
            map_bbox = self._shrink_to_map_boundaries(gray, initial_region)
            if map_bbox is None:
                logger.debug("Could not determine precise map boundaries")
                return []
            
            # Step 3: Validate the final result
            x, y, w, h = map_bbox
            if w < 100 or h < 100:
                logger.debug(f"Final region too small: {w}x{h}")
                return []
            
            logger.debug(f"Final map region: {w}x{h} at ({x},{y})")
            return [map_bbox]
            
        except Exception as e:
            logger.error(f"OpenCV detection failed: {e}")
            return []
    
    def _find_center_region(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the general center region where a map is likely to be located.
        """
        height, width = gray.shape
        
        # Start with center 60% of image (maps are usually central)
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        
        # Initial center region
        x = margin_x
        y = margin_y
        w = width - 2 * margin_x
        h = height - 2 * margin_y
        
        logger.debug(f"Initial center region: {w}x{h} at ({x},{y})")
        
        # Verify this region has reasonable detail levels
        region = gray[y:y+h, x:x+w]
        detail_score = self._calculate_detail_density(region)
        
        logger.debug(f"Center region detail score: {detail_score:.3f}")
        
        if detail_score > 0.05:  # Much lower threshold (was 0.1)
            return (x, y, w, h)
        else:
            logger.debug(f"Center region rejected: detail_score={detail_score:.3f} < 0.05")
        
        return None
    
    def _shrink_to_map_boundaries(self, gray: np.ndarray, initial_region: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Shrink from outside edges inward until we find the precise map boundaries.
        """
        x, y, w, h = initial_region
        
        logger.debug(f"Starting boundary detection from region: {w}x{h} at ({x},{y})")
        
        # Shrink each edge independently
        left_boundary = self._find_left_boundary(gray, x, y, w, h)
        right_boundary = self._find_right_boundary(gray, x, y, w, h)
        top_boundary = self._find_top_boundary(gray, x, y, w, h)
        bottom_boundary = self._find_bottom_boundary(gray, x, y, w, h)
        
        # Calculate final bounding box
        final_x = left_boundary
        final_y = top_boundary
        final_w = right_boundary - left_boundary
        final_h = bottom_boundary - top_boundary
        
        logger.debug(f"Boundaries: left={left_boundary}, right={right_boundary}, "
                    f"top={top_boundary}, bottom={bottom_boundary}")
        logger.debug(f"Final bbox: {final_w}x{final_h} at ({final_x},{final_y})")
        
        # Validate boundaries make sense
        if final_w > 50 and final_h > 50 and final_x >= 0 and final_y >= 0:
            return (final_x, final_y, final_w, final_h)
        
        return None
    
    def _find_left_boundary(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
        """Find the leftmost edge of detailed map content."""
        height, width = gray.shape
        
        # Start from left edge and move right until we find detailed content
        for offset in range(0, w // 2):
            test_x = x + offset
            if test_x >= width:
                break
                
            # Extract vertical strip for analysis
            strip_width = min(20, w - offset)
            if test_x + strip_width >= width:
                strip_width = width - test_x - 1
                
            strip = gray[y:y+h, test_x:test_x+strip_width]
            
            # Check if this strip contains detailed map content
            detail_score = self._calculate_detail_density(strip)
            
            # If we find sufficient detail, this is likely the map boundary
            if detail_score > 0.2:
                logger.debug(f"Left boundary found at x={test_x}, detail={detail_score:.3f}")
                return max(0, test_x - 5)  # Small buffer
        
        # Default to starting position if no clear boundary found
        return x
    
    def _find_right_boundary(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
        """Find the rightmost edge of detailed map content."""
        height, width = gray.shape
        
        # Start from right edge and move left until we find detailed content
        for offset in range(0, w // 2):
            test_x = x + w - offset
            if test_x <= x:
                break
                
            # Extract vertical strip for analysis
            strip_width = min(20, offset)
            if test_x - strip_width < 0:
                strip_width = test_x
                
            strip = gray[y:y+h, test_x-strip_width:test_x]
            
            # Check if this strip contains detailed map content
            detail_score = self._calculate_detail_density(strip)
            
            if detail_score > 0.2:
                logger.debug(f"Right boundary found at x={test_x}, detail={detail_score:.3f}")
                return min(width, test_x + 5)  # Small buffer
        
        return x + w
    
    def _find_top_boundary(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
        """Find the topmost edge of detailed map content."""
        height, width = gray.shape
        
        # Start from top edge and move down until we find detailed content
        for offset in range(0, h // 2):
            test_y = y + offset
            if test_y >= height:
                break
                
            # Extract horizontal strip for analysis
            strip_height = min(20, h - offset)
            if test_y + strip_height >= height:
                strip_height = height - test_y - 1
                
            strip = gray[test_y:test_y+strip_height, x:x+w]
            
            # Check if this strip contains detailed map content
            detail_score = self._calculate_detail_density(strip)
            
            if detail_score > 0.2:
                logger.debug(f"Top boundary found at y={test_y}, detail={detail_score:.3f}")
                return max(0, test_y - 5)  # Small buffer
        
        return y
    
    def _find_bottom_boundary(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
        """Find the bottommost edge of detailed map content."""
        height, width = gray.shape
        
        # Start from bottom edge and move up until we find detailed content
        for offset in range(0, h // 2):
            test_y = y + h - offset
            if test_y <= y:
                break
                
            # Extract horizontal strip for analysis
            strip_height = min(20, offset)
            if test_y - strip_height < 0:
                strip_height = test_y
                
            strip = gray[test_y-strip_height:test_y, x:x+w]
            
            # Check if this strip contains detailed map content
            detail_score = self._calculate_detail_density(strip)
            
            if detail_score > 0.2:
                logger.debug(f"Bottom boundary found at y={test_y}, detail={detail_score:.3f}")
                return min(height, test_y + 5)  # Small buffer
        
        return y + h
    
    def _calculate_detail_density(self, region: np.ndarray) -> float:
        """
        Calculate detail density to distinguish map artwork from pixelated Minecraft blocks.
        Maps have much higher detail density than Minecraft blocks.
        """
        try:
            if region.size == 0:
                return 0.0
            
            # Use multiple texture analysis methods
            
            # 1. Gradient-based texture analysis
            sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 2. High frequency content (detailed textures have more high frequencies)
            # Use Laplacian to detect rapid intensity changes
            laplacian = cv2.Laplacian(region, cv2.CV_64F)
            high_freq_score = np.std(laplacian) / 50
            
            # 3. Local variance (detailed areas have higher local variance)
            kernel = np.ones((5,5), np.float32) / 25
            region_float = region.astype(np.float32)
            local_mean = cv2.filter2D(region_float, -1, kernel)
            local_variance = cv2.filter2D(region_float**2, -1, kernel) - local_mean**2
            variance_score = np.mean(local_variance) / 1000
            
            # 4. Edge density (maps have more complex edge patterns)
            edges = cv2.Canny(region, 30, 100)
            edge_density = np.sum(edges > 0) / region.size
            
            # 5. Intensity distribution complexity (fixed entropy calculation)
            hist = cv2.calcHist([region], [0], None, [256], [0, 256]).flatten()
            hist = hist / (np.sum(hist) + 1e-7)  # Normalize to probabilities
            hist = hist + 1e-7  # Add small epsilon to avoid log(0)
            entropy = -np.sum(hist * np.log(hist))
            entropy_score = entropy / 8.0  # Normalize (max entropy for 256 bins ≈ 8)
            
            # Combine all metrics with weights
            detail_score = (
                min(np.std(gradient_magnitude) / 100, 1.0) * 0.3 +  # Gradient complexity
                min(high_freq_score, 1.0) * 0.25 +                  # High frequency content
                min(variance_score, 1.0) * 0.2 +                    # Local variance
                min(edge_density * 10, 1.0) * 0.15 +                # Edge density
                min(entropy_score, 1.0) * 0.1                       # Intensity distribution
            )
            
            return min(max(detail_score, 0.0), 1.0)  # Ensure result is between 0 and 1
            
        except Exception as e:
            logger.debug(f"Detail calculation failed: {e}")
            return 0.0
    
    def _detect_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect map frames using YOLO object detection"""
        try:
            if self.model is None:
                return []
            
            # Convert numpy array to PIL for YOLO
            pil_image = Image.fromarray(image)
            
            # Run YOLO detection
            results = self.model(pil_image, verbose=False)
            
            map_regions = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence
                        conf = float(box.conf)
                        
                        if conf >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Convert to (x, y, w, h) format
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            
                            map_regions.append((x, y, w, h))
            
            return map_regions
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _detect_segmentation(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect map regions using semantic segmentation"""
        try:
            if self.model is None:
                return []
            
            # Convert to PIL and preprocess
            pil_image = Image.fromarray(image)
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output['out'][0].cpu().numpy()
            
            # Post-process segmentation mask to find rectangular regions
            # This is a simplified approach - would need refinement
            mask = np.argmax(prediction, axis=0)
            
            # Find connected components
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            map_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50:  # Minimum size filter
                    map_regions.append((x, y, w, h))
            
            return map_regions
            
        except Exception as e:
            logger.error(f"Segmentation detection failed: {e}")
            return []
    
    def _detect_hybrid(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Combine multiple detection methods for better results"""
        # Start with OpenCV
        opencv_regions = self._detect_opencv(image)
        
        # If we have other models available, combine results
        # For now, just return OpenCV results
        return opencv_regions
    
    def crop_map_art(self, image: Union[Image.Image, np.ndarray], 
                     bbox: Tuple[int, int, int, int], 
                     padding: int = 10) -> Optional[Image.Image]:
        """
        Crop map art from image using bounding box.
        
        Args:
            image: Source image
            bbox: Bounding box (x, y, width, height)
            padding: Extra padding around detected region
            
        Returns:
            Cropped PIL Image or None if crop failed
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            x, y, w, h = bbox
            
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.width, x + w + padding)
            y2 = min(image.height, y + h + padding)
            
            # Crop the image
            cropped = image.crop((x1, y1, x2, y2))
            
            return cropped
            
        except Exception as e:
            logger.error(f"Failed to crop image: {e}")
            return None
    
    def process_image(self, image: Union[Image.Image, np.ndarray]) -> List[Image.Image]:
        """
        Process an image to detect and crop all map art regions.
        
        Args:
            image: Input image
            
        Returns:
            List of cropped map art images
        """
        # Detect map art regions
        regions = self.detect_map_art(image)
        
        if not regions:
            logger.debug("No map art regions detected")
            return []
        
        # Crop each detected region
        cropped_images = []
        for bbox in regions:
            cropped = self.crop_map_art(image, bbox)
            if cropped is not None:
                cropped_images.append(cropped)
        
        return cropped_images
    
    def visualize_detections(self, image: Union[Image.Image, np.ndarray], 
                           save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize detected map art regions on the image.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Image with detection boxes drawn
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Make a copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Detect regions
        regions = self.detect_map_art(image)
        
        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(regions):
            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            
            # Add label
            draw.text((x, y - 20), f"Map {i+1}", fill="red")
        
        if save_path:
            vis_image.save(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        return vis_image


def create_map_art_detector(method: str = "opencv") -> MapArtDetector:
    """
    Factory function to create a map art detector.
    
    Args:
        method: Detection method to use
        
    Returns:
        Configured MapArtDetector instance
    """
    return MapArtDetector(method=method) 