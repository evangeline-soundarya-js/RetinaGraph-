import pytest
from fastapi.testclient import TestClient
from backend.app import app
import numpy as np
from PIL import Image
import io

client = TestClient(app)

def create_dummy_image():
    # Create a 512x512 RGB image with a red circle to simulate a basic fundus shape
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Background
    img[:] = [10, 10, 10]
    # Retinal disc (red/orange)
    import cv2
    cv2.circle(img, (256, 256), 200, (200, 100, 50), -1)
    
    # Add some random lines (vessels)
    for _ in range(10):
        x1, y1 = np.random.randint(156, 356, 2)
        x2, y2 = np.random.randint(156, 356, 2)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    pil_img = Image.fromarray(img)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, 'PNG')
    byte_io.seek(0)
    return byte_io

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_endpoint_valid():
    # Test A - Valid/plausible retinal image
    dummy_image = create_dummy_image()
    
    response = client.post(
        "/analyze", 
        files={"file": ("test_fundus.png", dummy_image, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "prediction" in data
    assert data["confidence"] is None
    assert "graph" in data
    assert "nodes" in data["graph"]
    assert "edges" in data["graph"]
    assert "model_status" in data
    assert data["model_status"] == "prototype/untrained"

def test_analyze_endpoint_portrait():
    # Test B - Human portrait (represented by e.g. a blue/white heavy image or lacking circular mask)
    # Let's create an image that fails the color/structure check
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [200, 200, 255] # Mostly blue/white, not red/orange, not circular
    
    pil_img = Image.fromarray(img)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post(
        "/analyze", 
        files={"file": ("portrait.png", byte_io, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["model_status"] == "not_run"
    assert data["prediction"] is None
    assert data["graph"] is None

def test_analyze_endpoint_blank():
    # Test C - Blank image (uniform color)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [128, 128, 128] # Uniform gray
    
    pil_img = Image.fromarray(img)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post(
        "/analyze", 
        files={"file": ("blank.png", byte_io, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["model_status"] == "not_run"

def test_analyze_endpoint_white():
    # Test D - Completely white image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [255, 255, 255]
    
    pil_img = Image.fromarray(img)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post(
        "/analyze", 
        files={"file": ("white.png", byte_io, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["model_status"] == "not_run"

def test_analyze_endpoint_dark():
    # Test E - Completely dark image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [2, 2, 2]
    
    pil_img = Image.fromarray(img)
    byte_io = io.BytesIO()
    pil_img.save(byte_io, 'PNG')
    byte_io.seek(0)
    
    response = client.post(
        "/analyze", 
        files={"file": ("dark.png", byte_io, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["model_status"] == "not_run"

def test_analyze_endpoint_corrupted():
    # Test F - Corrupted/unreadable image
    corrupted_data = io.BytesIO(b"This is not a valid image file")
    
    response = client.post(
        "/analyze", 
        files={"file": ("corrupted.png", corrupted_data, "image/png")}
    )
    
    # Should be handled gracefully without crashing the API
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert data["model_status"] == "not_run"
    assert data["checks"]["decodable"] is False
