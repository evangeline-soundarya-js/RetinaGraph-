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
3. **Graph Construction**:
    - Constructs a deterministic, directed spatial graph using Scipy's `cKDTree`.
    - **Nodes**: Exact mathematical coordinates derived from Phase 3 topology extraction.
    - **Node Features**: `[norm_x, norm_y, intensity, vesselness]`.
    - **Edges**: Connects each node to its $k$ closest topological neighbors (configurable, default $k=5$). 
    - **Edge Features**: `[euclidean_distance, relative_x, relative_y]`.
    - **Graph Statistics**: Automatically computes degree stats, component connectivity, and density for research auditing.
    - Outputs a standard `torch_geometric.data.Data` object.
4. **Graph Neural Network (GAT)**:
    - Built with PyTorch Geometric (`RetinaGAT`).
    - **Input Channel**: `in_channels=4` (Node topology features).
    - **Edge Dimension**: `edge_dim=3` (Passed explicitly via GATConv).
    - **Hidden Channel**: `hidden_channels=16`.
    - **Multi-Head**: 2 attention heads (Output of Layer 1 concatenated, Layer 2 averaged).
    - **Pooling**: `global_mean_pool` translates local structure to a singular representation per retina.
    - **Classifier**: Linear projection from `16` hidden features to `2` outcome logits.
    - **Status**: Currently explicitly marked `prototype/untrained`. It performs structurally deterministic matrix operations but does not output medical confidence.
5. **Explainability Layer**: Converts attention weights and graph structures into visualizable evidence.
6. **Backend & Frontend**: FastAPI serves the model natively. The API guarantees null clinical predictions while the model remains untrained.

## Limitations
- This is a **research prototype**. 
- The feature extraction uses basic image processing baselines, not clinically validated segmentation models.
- The GNN model is a prototype and not trained on large clinical datasets. It provides a structural foundation for future trained weights.
- Any output predictions or confidence scores reflect the mathematical behavior of the prototype architecture, NOT medical diagnostic capability.
