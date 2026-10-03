# RetinaGraph AI - Architecture

## Overview
RetinaGraph AI is a research prototype that implements a graph-based analysis pipeline for retinal fundus images. 
It accepts an image, extracts feature structures (vessels, junctions), constructs a graph representation, and feeds it into a Graph Attention Network (GAT) to produce a classification with corresponding structural evidence.

## Pipeline Components

1. **Preprocessing**: Validates the input format (JPG, PNG), resizes the image to a standardized dimension, normalizes intensities, and performs basic quality checks.
2. **Feature Extraction**: Uses conventional computer vision (OpenCV/scikit-image) to extract prototype features. Specifically, it extracts vessel-like structures, junctions, and regional intensity statistics.
3. **Graph Construction**:
    - **Nodes**: Represent key structural points (e.g., vessel junctions or uniformly sampled keypoints on vessels).
    - **Edges**: Represent spatial proximity or structural connectivity between nodes.
    - **Node Features**: Local intensity and structural properties (e.g., vessel thickness, local contrast).
4. **Graph Neural Network (GAT)**:
    - Built with PyTorch Geometric.
    - Includes multi-head graph attention layers.
    - Outputs a graph-level classification and attention weights for explainability.
5. **Explainability Layer**: Converts attention weights and graph structures into visualizable evidence.
6. **Backend & Frontend**: FastAPI serves the model, and a React/Vite frontend provides the UI for interaction.

## Limitations
- This is a **research prototype**. 
- The feature extraction uses basic image processing baselines, not clinically validated segmentation models.
- The GNN model is a prototype and not trained on large clinical datasets. It provides a structural foundation for future trained weights.
- Any output predictions or confidence scores reflect the mathematical behavior of the prototype architecture, NOT medical diagnostic capability.
