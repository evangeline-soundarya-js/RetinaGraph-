import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import io

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def _crop_to_retinal_field(self, img_np: np.ndarray) -> np.ndarray:
        """
        Crops the image to the bounding box of the retinal field to remove excess background.
        """
        # Convert to grayscale to find the mask
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # Threshold to find non-black regions (retinal field)
        # Using a conservative threshold to avoid cropping dark retinal tissue
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find bounding box
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add a small margin (e.g., 5 pixels)
            x = max(0, x - 5)
            y = max(0, y - 5)
            w = min(img_np.shape[1] - x, w + 10)
            h = min(img_np.shape[0] - y, h + 10)
            
            return img_np[y:y+h, x:x+w]
        
        return img_np # Return original if no contour found (e.g., all black or no distinct edge)

    def _resize_with_aspect_ratio(self, img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes the image to target_size while preserving aspect ratio by padding with black.
        Returns the padded image and the valid retinal mask for the padded image.
        """
        h, w = img_np.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale to fit inside target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize using INTER_AREA which is good for downsampling
        resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a black canvas
        if len(img_np.shape) == 3:
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            
        # Create a mask tracking the valid retinal area
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
            
        # Calculate padding offsets to center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Paste the resized image into the center of the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 255
        
        return canvas, mask

    def process(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Processes a raw image byte stream through a deterministic retinal pipeline.
        Returns a dictionary with representations and metadata.
        """
        # 1. Load image (PIL loads as RGB natively)
        img = Image.open(io.BytesIO(image_bytes))
        
        metadata = {
            "original_size": img.size, # (width, height)
            "format": img.format,
            "mode": img.mode
        }
        
        # 2. Color Handling & Channel Ordering
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            metadata["channel_conversion"] = "GRAY2RGB"
        elif len(img_np.shape) == 3 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            metadata["channel_conversion"] = "RGBA2RGB"
        else:
            metadata["channel_conversion"] = "None (RGB)"
            
        # 3. Retinal Field Handling (Crop)
        cropped_img = self._crop_to_retinal_field(img_np)
        metadata["cropped_size"] = (cropped_img.shape[1], cropped_img.shape[0])
        
        # 4. Aspect-Ratio Preserving Resize
        padded_img, retinal_mask = self._resize_with_aspect_ratio(cropped_img)
        metadata["target_size"] = self.target_size
        metadata["resize_method"] = "aspect_ratio_preserve_with_padding"
        
        # 5. Green Channel Extraction (best contrast for vessels)
        green_channel = padded_img[:, :, 1]
        
        # 6. Contrast Enhancement (CLAHE)
        # Using standard parameters for retinal enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(green_channel)
        
        # 7. Normalization [0, 1] for the vessel representation
        normalized_green = enhanced_green.astype(np.float32) / 255.0
        
        metadata["preprocessing_steps"] = [
            "crop_to_retinal_field",
            "aspect_ratio_resize", 
            "green_channel_extraction", 
            "clahe_enhancement",
            "0_to_1_normalization"
        ]
        
        return {
            "processed_color": padded_img,         # RGB, uint8 [0, 255]
            "vessel_representation": normalized_green, # Grayscale, float32 [0.0, 1.0]
            "retinal_mask": retinal_mask,          # Binary mask, uint8 [0, 255]
            "metadata": metadata
        }
