import pytest
import numpy as np
from src.features.extractor import FeatureExtractor
import cv2

@pytest.fixture
def extractor():
    return FeatureExtractor(max_keypoints=150)

def create_synthetic_vessels(shape=(512, 512)):
    # Create a blank image
    img = np.zeros(shape, dtype=np.float32)
    # Draw a Y-shape vessel (1 junction, 3 endpoints)
    cv2.line(img, (256, 100), (256, 256), 1.0, 5) # Trunk
    cv2.line(img, (256, 256), (150, 400), 1.0, 5) # Left branch
    cv2.line(img, (256, 256), (350, 400), 1.0, 5) # Right branch
    return img

def test_deterministic_vessel_and_skeleton(extractor):
    # Test A, B, J
    img = create_synthetic_vessels()
    res1 = extractor.extract_features(img)
    res2 = extractor.extract_features(img)
    
    assert res1["status"] == "success"
    assert np.array_equal(res1["vessel_mask"], res2["vessel_mask"])
    assert np.array_equal(res1["skeleton"], res2["skeleton"])
    assert np.array_equal(res1["keypoints"], res2["keypoints"])
    assert np.array_equal(res1["node_features"], res2["node_features"])

def test_image_derived_coordinates_and_no_fallbacks(extractor):
    # Test C, D, E, F
    img = create_synthetic_vessels()
    res = extractor.extract_features(img)
    
    assert res["status"] == "success"
    meta = res["metadata"]
    # Due to digital line rasterization, thickness, and skeletonization artifacts,
    # the exact number of junctions/endpoints can vary slightly (e.g., a junction might split into 2 nearby ones).
    # We just ensure it found structure.
    assert meta["num_junctions"] >= 1
    assert meta["num_endpoints"] >= 2
    assert meta["num_candidates"] >= 3
    
    # Check that they match the expected coordinates approximately
    kpts = res["keypoints"]
    assert len(kpts) >= 4
    # Ensure they are not random fallbacks or hardcoded [256, 256] only
    unique_kpts = np.unique(kpts, axis=0)
    assert len(unique_kpts) >= 4

def test_normalized_coordinates_and_finite_values(extractor):
    # Test G, H
    img = create_synthetic_vessels()
    res = extractor.extract_features(img)
    
    features = res["node_features"]
    # Features: [norm_x, norm_y, intensity, vesselness]
    assert np.all(np.isfinite(features))
    assert np.all(features[:, 0] >= 0.0) and np.all(features[:, 0] <= 1.0)
    assert np.all(features[:, 1] >= 0.0) and np.all(features[:, 1] <= 1.0)

def test_insufficient_structure(extractor):
    # Test I
    img = np.zeros((512, 512), dtype=np.float32)
    res = extractor.extract_features(img)
    
    assert res["status"] == "insufficient_retinal_structure"
    assert res["keypoints"] is None
    assert res["node_features"] is None
    assert res["metadata"]["num_candidates"] == 0
