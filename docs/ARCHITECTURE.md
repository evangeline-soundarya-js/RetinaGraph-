# RetinaGraph AI - Architecture

## Overview
RetinaGraph AI is a research prototype that implements a graph-based analysis pipeline for retinal fundus images. 
It accepts an image, extracts feature structures (vessels, junctions), constructs a graph representation, and feeds it into a Graph Attention Network (GAT) to produce a classification with corresponding structural evidence.

## Pipeline Components

1. **Preprocessing**: 
    - Loads images safely (handling grayscale, RGB, RGBA correctly to RGB).
    - Determines the retinal field bounding box to crop excess black background safely.
    - Resizes to a consistent `(512, 512)` maintaining aspect ratio by padding with black (letterboxing), avoiding retinal distortion.
    - Extracts the Green channel.
    - Enhances contrast using CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`).
    - Normalizes the output strictly to `float32 [0.0, 1.0]` yielding a deterministic `vessel_representation` alongside the unmodified padded RGB image.
2. **Feature Extraction**: 
    - Derives meaningful structural coordinates deterministically from the image without random guessing or hard-coding.
    - **Vesselness**: Applies a Frangi filter.
    - **Skeletonization**: Converts thresholded vessels to a 1-pixel wide skeleton.
    - **Topology Detection**: Scans the skeleton to detect exact Endpoints (1-neighbor) and Junctions (>=3 neighbors). Closely clustered junction pixels are merged via morphological dilation and connected components.
    - **Features**: For each candidate coordinate, generates a normalized tensor `[norm_x, norm_y, intensity, vesselness]`.
    - **Graceful Failure**: Safely returns an explicit failure state if no topology is found, preventing the downstream generation of fake graphs.
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
