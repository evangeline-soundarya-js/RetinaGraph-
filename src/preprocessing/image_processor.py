import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import io

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def process(self, image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a raw image byte stream.
        Returns the preprocessed image array and metadata.
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        
        metadata = {
            "original_size": img.size,
            "format": img.format,
            "mode": img.mode
        }
        
        # Convert to numpy and RGB if necessary
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img_resized = cv2.resize(img_np, self.target_size)
        
        # Extract Green channel (often best for retinal vessels)
        green_channel = img_resized[:, :, 1]
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_green = clahe.apply(green_channel)
        
        metadata["target_size"] = self.target_size
        metadata["preprocessing_steps"] = ["resize", "green_channel_extraction", "clahe_enhancement"]
        
        # Return the enhanced image (can be used for feature extraction)
        return enhanced_green, metadata
