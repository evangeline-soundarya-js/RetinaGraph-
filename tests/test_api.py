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

def test_analyze_endpoint():
    # Create a dummy image
    dummy_image = create_dummy_image()
    
    # Send request
    response = client.post(
        "/analyze", 
        files={"file": ("test_fundus.png", dummy_image, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert data["status"] == "success"
    assert "prediction" in data
    assert data["confidence"] is None
    assert "graph" in data
    assert "nodes" in data["graph"]
    assert "edges" in data["graph"]
    assert "model_status" in data
    assert data["model_status"] == "prototype/untrained"
