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
        
    from src.inference.explainer import GATExplainer
    
    print("Extracting Evidence via GATExplainer...")
    explainer = GATExplainer(pipeline.model)
    explanation = explainer.explain(data, top_k=10)
    
    top_node_indices = [n["node_index"] for n in explanation["top_nodes"]]
    top_edge_indices = [e["edge_index"] for e in explanation["top_edges"]]
    
    # Visualization Overlay
    plt.figure(figsize=(10, 10))
    # We display the normalized color image
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    
    # Plot all edges
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        src = edges[0, i]
        dst = edges[1, i]
        
        # Color top edges differently
        if i in top_edge_indices:
            plt.plot([keypoints[src, 0], keypoints[dst, 0]], 
                     [keypoints[src, 1], keypoints[dst, 1]], 
                     'y-', alpha=0.9, linewidth=3, zorder=4)
        else:
            plt.plot([keypoints[src, 0], keypoints[dst, 0]], 
                     [keypoints[src, 1], keypoints[dst, 1]], 
                     'c-', alpha=0.3, linewidth=1, zorder=3)
                 
    # Plot all nodes
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='blue', s=20, zorder=5, alpha=0.5)
    
    # Plot top nodes
    top_kpts = keypoints[top_node_indices]
    plt.scatter(top_kpts[:, 0], top_kpts[:, 1], c='red', s=80, marker='*', zorder=6, label="Top 10 Important Nodes")
    
    plt.title(f"RetinaGraph Evidence (Nodes: {data.stats['num_nodes']}, Pred: {explanation['prediction']})")
    plt.legend()
    plt.axis('off')
    
    out_path = os.path.join(os.path.dirname(__file__), 'graph_evidence_visualization.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize_graph()
