import pytest
import numpy as np
import io
from PIL import Image
from src.preprocessing.image_processor import ImageProcessor

def create_image(size=(1024, 768), mode="RGB", color=(100, 50, 0)):
    """Helper to create dummy images for testing."""
    if mode == "RGB":
        img_np = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        # Add a mock retinal field (circle)
        import cv2
        cv2.circle(img_np, (size[0]//2, size[1]//2), min(size)//3, (200, 100, 50), -1)
    elif mode == "RGBA":
        img_np = np.full((size[1], size[0], 4), color + (255,), dtype=np.uint8)
        import cv2
        cv2.circle(img_np, (size[0]//2, size[1]//2), min(size)//3, (200, 100, 50, 255), -1)
    elif mode == "L":
        img_np = np.full((size[1], size[0]), color[0], dtype=np.uint8)
        import cv2
        cv2.circle(img_np, (size[0]//2, size[1]//2), min(size)//3, 150, -1)
        
    pil_img = Image.fromarray(img_np)
    byte_io = io.BytesIO()
    # Save as PNG
    pil_img.save(byte_io, 'PNG')
    return byte_io.getvalue()

@pytest.fixture
def processor():
    return ImageProcessor(target_size=(512, 512))

def test_normal_color_image(processor):
    # Test A & I (channel ordering via cvtColor RGB/BGR not messing up shapes)
    img_bytes = create_image(mode="RGB")
    result = processor.process(img_bytes)
    
    assert "processed_color" in result
    assert result["processed_color"].shape == (512, 512, 3)
    assert result["metadata"]["channel_conversion"] == "None (RGB)"

def test_grayscale_image(processor):
    # Test B
    img_bytes = create_image(mode="L")
    result = processor.process(img_bytes)
    
    assert result["processed_color"].shape == (512, 512, 3)
    assert result["metadata"]["channel_conversion"] == "GRAY2RGB"

def test_rgba_image(processor):
    # Test C
    img_bytes = create_image(mode="RGBA")
    result = processor.process(img_bytes)
    
    assert result["processed_color"].shape == (512, 512, 3)
    assert result["metadata"]["channel_conversion"] == "RGBA2RGB"

def test_different_dimensions_and_aspect_ratio(processor):
    # Test D and E
    img_bytes = create_image(size=(800, 300), mode="RGB") # Wide image
    result = processor.process(img_bytes)
    
    assert result["processed_color"].shape == (512, 512, 3)
    assert result["metadata"]["resize_method"] == "aspect_ratio_preserve_with_padding"
    # The mask should reflect the padding
    mask = result["retinal_mask"]
    # Since it's wide (800x300), after scaling it should be 512x192, so top/bottom are padded
    assert mask[0, 256] == 0 # Top padding
    assert mask[256, 256] == 255 # Center is valid

def test_normalization_range(processor):
    # Test F
    img_bytes = create_image(mode="RGB")
    result = processor.process(img_bytes)
    
    vessel_rep = result["vessel_representation"]
    assert vessel_rep.dtype == np.float32
    assert np.min(vessel_rep) >= 0.0
    assert np.max(vessel_rep) <= 1.0

def test_deterministic_output(processor):
    # Test G
    img_bytes = create_image(mode="RGB")
    result1 = processor.process(img_bytes)
    result2 = processor.process(img_bytes)
    
    assert np.array_equal(result1["processed_color"], result2["processed_color"])
    assert np.array_equal(result1["vessel_representation"], result2["vessel_representation"])

def test_invalid_input(processor):
    # Test H
    with pytest.raises(Exception):
        processor.process(b"not an image")
