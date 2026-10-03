import os
import csv
import cv2
import numpy as np
from pathlib import Path

def create_synthetic_vessels(shape=(512, 512), noise_level=10):
    img = np.zeros(shape, dtype=np.uint8)
    
    # Random trunk
    cx = np.random.randint(200, 300)
    cy = np.random.randint(200, 300)
    
    # Draw Y-shape vessel (1 junction, 3 endpoints)
    cv2.line(img, (cx, 100), (cx, cy), 255, 5) # Trunk
    cv2.line(img, (cx, cy), (cx - 100, cy + 150), 255, 5) # Left branch
    cv2.line(img, (cx, cy), (cx + 100, cy + 150), 255, 5) # Right branch
    
    # Add random noise
    noise = np.random.randint(0, noise_level, shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Convert to 3 channel
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img_rgb

def generate_dataset(root_dir="data", num_samples=20):
    root = Path(root_dir)
    images_dir = root / "images"
    metadata_dir = root / "metadata"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = metadata_dir / "metadata.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "image_id", "patient_id"])
        
        for i in range(num_samples):
            # Class 0: healthy (less noise), Class 1: DR (more noise)
            label = np.random.randint(0, 2)
            noise_level = 10 if label == 0 else 50
            
            img = create_synthetic_vessels(noise_level=noise_level)
            img_filename = f"img_{i:03d}.png"
            img_path = images_dir / img_filename
            
            cv2.imwrite(str(img_path), img)
            
            # Write metadata with relative path
            writer.writerow([f"../images/{img_filename}", label, f"img_{i:03d}", f"pat_{i//5}"])
            
    print(f"Generated {num_samples} samples at {root_dir}")

if __name__ == "__main__":
    generate_dataset()
