import cv2
import numpy as np
from skimage.filters import frangi
from typing import Tuple, List

class FeatureExtractor:
    def __init__(self, max_keypoints: int = 100):
        self.max_keypoints = max_keypoints

    def extract_features(self, enhanced_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts prototype features from the preprocessed image.
        Returns:
            keypoints: (N, 2) array of [y, x] coordinates
            node_features: (N, D) array of local features
            vessel_mask: (H, W) binary mask of vessel-like structures
        """
        # 1. Vesselness filtering (baseline method)
        # Convert to float for Frangi filter
        img_float = enhanced_image.astype(np.float64) / 255.0
        vesselness = frangi(img_float)
        
        # Normalize vesselness to 0-255 uint8
        vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold to get a mask
        _, vessel_mask = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Keypoint extraction (finding junctions or significant points on vessels)
        # We use Shi-Tomasi corner detector on the vessel mask as a prototype
        corners = cv2.goodFeaturesToTrack(vessel_mask, maxCorners=self.max_keypoints, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            keypoints = corners.reshape(-1, 2) # [x, y]
            # Convert to [y, x] for standard image indexing if needed, but we keep [x, y] for geometry
        else:
            # Fallback if no corners found: random points on vessels
            y_coords, x_coords = np.where(vessel_mask > 0)
            if len(y_coords) > 0:
                indices = np.random.choice(len(y_coords), min(self.max_keypoints, len(y_coords)), replace=False)
                keypoints = np.column_stack((x_coords[indices], y_coords[indices]))
            else:
                # Absolute fallback
                keypoints = np.array([[enhanced_image.shape[1]//2, enhanced_image.shape[0]//2]])
                
        # 3. Node feature extraction (local intensity and vesselness)
        node_features = []
        for pt in keypoints:
            x, y = int(pt[0]), int(pt[1])
            # Ensure within bounds
            x = max(0, min(x, enhanced_image.shape[1]-1))
            y = max(0, min(y, enhanced_image.shape[0]-1))
            
            intensity = img_float[y, x]
            v_val = vesselness[y, x]
            # Coordinates normalized
            norm_x = x / enhanced_image.shape[1]
            norm_y = y / enhanced_image.shape[0]
            
            node_features.append([norm_x, norm_y, intensity, v_val])
            
        return keypoints, np.array(node_features, dtype=np.float32), vessel_mask
