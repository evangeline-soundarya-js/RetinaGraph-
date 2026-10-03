# Research Status

## Implementation Status

### Implemented
- **Project Structure**: Clean modular separation (backend, frontend, src for ML).
- **Preprocessing**: Baseline implementation using grayscale conversion, resizing, Green channel extraction, and CLAHE.
- **Feature Extraction**: Prototype implementation utilizing Frangi vesselness filter and Shi-Tomasi corner detection to extract morphological keypoints and local statistics.
- **Graph Construction**: Coordinates converted to KNN (k-nearest neighbors) graph representation using PyTorch Geometric.
- **GAT Model**: Multi-head Graph Attention Network constructed with PyTorch Geometric (`GATConv`). Supports returning attention weights for explainability.
- **Explainability Layer**: Foundational evidence output, extracting node sizes, degrees, and attention methods (currently a placeholder wrapper around attention weights).
- **API**: FastAPI running with CORS middleware and `python-multipart` to accept file uploads.
- **Frontend UI**: React + Vite application with dynamic styling, error handling, and research disclaimers.

### Partially Implemented
- **Graph Attention to Heatmap**: The GAT model extracts attention weights, but the mapping back onto the original image pixels is simplified to text stats for the UI.

### Placeholders / Untrained
- **Model Weights**: The `RetinaGAT` is initialized with random weights. It produces predictions but these are structurally driven randomness, NOT trained clinical knowledge.

### Not Implemented
- Medical diagnostic capabilities.
- Large-scale dataset downloading (e.g. Kaggle EyePACS or Messidor-2).
- GPU support for inference (forced to run on CPU for local testing).

## External Libraries Used
- **OpenCV & scikit-image**: For baseline computer vision (vessel extraction, CLAHE).
- **PyTorch Geometric**: For the GNN components.
- **FastAPI**: Backend REST API.
- **React & Vite**: Frontend UI.
- **Lucide React**: UI Icons.
