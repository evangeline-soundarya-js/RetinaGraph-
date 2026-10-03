import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from typing import Tuple, List, Dict, Any, Optional

class FeatureExtractor:
    def __init__(self, max_keypoints: int = 150):
        self.max_keypoints = max_keypoints
        # 3x3 kernel to count neighbors in a binary skeleton
        self.neighbor_kernel = np.array([[1, 1, 1],
                                         [1, 0, 1],
                                         [1, 1, 1]], dtype=np.uint8)

    def extract_features(self, enhanced_image: np.ndarray) -> Dict[str, Any]:
        """
        Extracts structural features (junctions, endpoints) from the preprocessed image.
        Returns a dictionary containing keypoints, features, and debug masks.
        """
        # 1. Vesselness filtering
        # Input is expected to be normalized float [0.0, 1.0]
        img_float = enhanced_image.astype(np.float64)
        vesselness = frangi(img_float)
        
        # Normalize vesselness to 0-255 uint8 for thresholding
        vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Threshold to get a binary mask of vessels
        # We use Otsu's thresholding
        _, vessel_mask_uint8 = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel_mask = vessel_mask_uint8 > 0
        
        # 2. Skeletonization
        # skeletonize requires boolean array
        skeleton = skeletonize(vessel_mask)
        skeleton_uint8 = skeleton.astype(np.uint8)
        
        # 3. Detect Junctions and Endpoints via connectivity
        # Count neighbors for each skeleton pixel
        neighbor_count = convolve(skeleton_uint8, self.neighbor_kernel, mode='constant', cval=0)
        
        # Endpoints have exactly 1 neighbor in the skeleton
        endpoints_mask = (skeleton_uint8 == 1) & (neighbor_count == 1)
        
        # Junctions have 3 or more neighbors in the skeleton
        junctions_mask = (skeleton_uint8 == 1) & (neighbor_count >= 3)
        
        # 4. Group nearby junctions to avoid clusters
        # Dilate the junctions mask slightly to merge junction pixels that are very close (e.g. 1-2 pixels apart)
        # which commonly happens in skeletonization of thick vessels.
        kernel = np.ones((3,3), np.uint8)
        dilated_junctions = cv2.dilate(junctions_mask.astype(np.uint8), kernel, iterations=1)
        
        # Use connected components to group adjacent junction pixels
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_junctions, connectivity=8)
        
        candidate_points = []
        
        # Add endpoints to candidates
        ey, ex = np.where(endpoints_mask)
        for y, x in zip(ey, ex):
            candidate_points.append({"coord": (x, y), "type": "endpoint"})
            
        # Add grouped junctions to candidates (skip label 0 which is background)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            candidate_points.append({"coord": (int(cx), int(cy)), "type": "junction"})
            
        num_endpoints = len(ey)
        num_junctions = num_labels - 1
        
        # 5. Handle empty/insufficient structure safely
        if len(candidate_points) == 0:
            return {
                "status": "insufficient_retinal_structure",
                "keypoints": None,
                "node_features": None,
                "vessel_mask": vessel_mask_uint8,
                "skeleton": skeleton_uint8 * 255,
                "metadata": {
                    "num_endpoints": 0,
                    "num_junctions": 0,
                    "num_candidates": 0
                }
            }
            
        # Subsample if there are too many candidates to fit max_keypoints
        if len(candidate_points) > self.max_keypoints:
            # We shuffle deterministically by fixing seed temporarily or just pick evenly
            np.random.seed(42) # deterministic subsampling
            indices = np.random.choice(len(candidate_points), self.max_keypoints, replace=False)
            candidate_points = [candidate_points[i] for i in indices]
            
        # 6. Extract Node Features
        keypoints = []
        node_features = []
        h, w = enhanced_image.shape
        
        for pt in candidate_points:
            x, y = pt["coord"]
            # Ensure within bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            
            keypoints.append([x, y])
            
            # Features
            norm_x = x / w
            norm_y = y / h
            intensity = img_float[y, x]
            v_val = vesselness[y, x]
            
            node_features.append([norm_x, norm_y, intensity, v_val])
            
        return {
            "status": "success",
            "keypoints": np.array(keypoints, dtype=np.float32),
            "node_features": np.array(node_features, dtype=np.float32),
            "vessel_mask": vessel_mask_uint8,
            "skeleton": skeleton_uint8 * 255,
            "metadata": {
                "num_endpoints": num_endpoints,
                "num_junctions": num_junctions,
                "num_candidates": len(candidate_points)
            }
        }
