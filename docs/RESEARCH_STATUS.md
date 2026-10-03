# Research Status

## Implementation Status

### Implemented
- **Project Structure**: Clean modular separation (backend, frontend, src for ML).
- **Fundus Input Validator**: Heuristic plausibility gate to reject non-fundus (e.g., portrait) or corrupted/blank images before processing. This is a heuristic and NOT clinically validated.
- **Preprocessing**: Deterministic, non-distorting pipeline including retinal field cropping, aspect-ratio preserving resize (padding to 512x512), Green channel extraction, CLAHE contrast enhancement, and strict [0.0, 1.0] normalization. Output explicitly separates color representation from vessel representation.
- **Feature Extraction**: Deterministic topology-based structural point extraction. Extracts vesselness (Frangi), skeletonizes the mask, and explicitly calculates junction and endpoint coordinates via connectivity. Groups nearby junction artifacts and returns robust node feature vectors `[norm_x, norm_y, intensity, vessel_strength]`. Safely halts if no structure is found.
- **Graph Construction**: Deterministic directed KNN graph built via Scipy `cKDTree`. Nodes contain strictly derived topology features (`x`, `y`, `intensity`, `vesselness`). Edges encode Euclidean distance and relative offset. Graph includes rich metadata and statistics reporting. Compatible natively with PyTorch Geometric (`Data`). No fake nodes or fallback matrices.
- **GAT Model**: `RetinaGAT` successfully integrated. Accepts `4`-dim nodes and `3`-dim edge features through multi-head `GATConv`. Implements explicit config registry and dropout. Output explicitly nulled as `prototype/untrained` to strictly avoid falsely projecting untrained logits as medical inference.
- **Explainability Layer**: Foundational evidence output, extracting node sizes, degrees, and attention methods (currently a placeholder wrapper around attention weights).
- **API**: FastAPI pipeline dynamically propagates real topology through to GAT output natively. Returns structured nullified inference with valid underlying topology structures.
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
