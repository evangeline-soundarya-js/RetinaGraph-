# Research Status

## Implementation Status

### Implemented
- **Project Structure**: Clean modular separation (backend, frontend, src for ML).
- **Fundus Input Validator**: Heuristic plausibility gate to reject non-fundus (e.g., portrait) or corrupted/blank images before processing. This is a heuristic and NOT clinically validated.
- **Preprocessing**: Deterministic, non-distorting pipeline including retinal field cropping, aspect-ratio preserving resize (padding to 512x512), Green channel extraction, CLAHE contrast enhancement, and strict [0.0, 1.0] normalization. Output explicitly separates color representation from vessel representation.
- **Feature Extraction**: Deterministic topology-based structural point extraction. Extracts vesselness (Frangi), skeletonizes the mask, and explicitly calculates junction and endpoint coordinates via connectivity. Groups nearby junction artifacts and returns robust node feature vectors `[norm_x, norm_y, intensity, vessel_strength]`. Safely halts if no structure is found.
- **Graph Construction**: Coordinates converted to KNN (k-nearest neighbors) graph representation using Scipy `cKDTree` as a PyG `knn_graph` fallback.
- **GAT Model**: Multi-head Graph Attention Network constructed with PyTorch Geometric (`GATConv`). Supports returning attention weights for explainability.
- **Explainability Layer**: Foundational evidence output, extracting node sizes, degrees, and attention methods (currently a placeholder wrapper around attention weights).
- **API**: FastAPI running with CORS middleware and `python-multipart` to accept file uploads.
- **Frontend UI**: React + Vite application with dynamic styling, error handling, input validation states, and research disclaimers.

### Partially Implemented
- **Graph Attention to Heatmap**: The GAT model extracts attention weights, but the mapping back onto the original image pixels is simplified to text stats for the UI.

### Placeholders / Untrained
- **Model Weights**: The `RetinaGAT` is initialized with random weights. It produces predictions but these are structurally driven randomness, NOT trained clinical knowledge.

### Not Implemented / Future Work
- Medical diagnostic capabilities.
- Large-scale dataset downloading (e.g. Kaggle EyePACS or Messidor-2) for actual model training.
- GPU support for inference (forced to run on CPU for local testing).
- Advanced vessel segmentation and lesion detection.
- Doctor feedback and new dashboard capabilities.
- New authentication or cloud deployment.

## External Libraries Used
- **OpenCV & scikit-image**: For baseline computer vision (vessel extraction, CLAHE).
- **PyTorch Geometric**: For the GNN components.
- **FastAPI**: Backend REST API.
- **React & Vite**: Frontend UI.
- **Lucide React**: UI Icons.
