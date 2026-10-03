import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.inference.pipeline import InferencePipeline
from tests.test_api import create_dummy_image

def visualize_graph():
    print("Initializing Pipeline...")
    pipeline = InferencePipeline()
    
    print("Generating Dummy Image...")
    img_io = create_dummy_image()
    img_bytes = img_io.getvalue()
    
    print("Running Preprocessing...")
    prep_result = pipeline.image_processor.process(img_bytes)
    color_img = prep_result["processed_color"]
    enhanced_image = prep_result["vessel_representation"]
    
    print("Running Feature Extraction...")
    feature_result = pipeline.feature_extractor.extract_features(enhanced_image)
    if feature_result["status"] == "insufficient_retinal_structure":
        print("Failed to detect structure.")
        return
        
    keypoints = feature_result["keypoints"]
    node_features = feature_result["node_features"]
    
    print(f"Detected {len(keypoints)} structural points.")
    
    print("Building Graph...")
    data = pipeline.graph_builder.build_graph(keypoints, node_features)
    
    print("Graph Statistics:")
    for k, v in data.stats.items():
        print(f"  {k}: {v}")
        
    # Visualization Overlay
    plt.figure(figsize=(10, 10))
    # We display the normalized color image
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    
    # Plot edges
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        src = edges[0, i]
        dst = edges[1, i]
        plt.plot([keypoints[src, 0], keypoints[dst, 0]], 
                 [keypoints[src, 1], keypoints[dst, 1]], 
                 'c-', alpha=0.5, linewidth=1)
                 
    # Plot nodes
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=20, zorder=5)
    
    plt.title(f"RetinaGraph Visualization (Nodes: {data.stats['num_nodes']}, Edges: {data.stats['num_edges']})")
    plt.axis('off')
    
    out_path = os.path.join(os.path.dirname(__file__), 'graph_visualization.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize_graph()
