# Experiment Log

## Experiment 1: End-to-End Pipeline Validation
**Date**: 2026-10-03
**Objective**: Validate that a fundus image can pass through the entire pipeline (Preprocessing -> Features -> Graph -> GAT -> FastAPI -> React) without throwing shape or dimension errors.
**Setup**: 
- Model: RetinaGAT (4 input features: normalized x, normalized y, intensity, vesselness)
- Graph: K=5 Nearest Neighbors based on spatial distance of extracted keypoints.
- Max Keypoints: 150
**Results**: Pipeline runs successfully with Dummy test images. Servers functional.

## Experiment 2: Validation Gate Implementation (Phase 1)
**Date**: 2026-10-03
**Objective**: Implement an input plausibility gate (`FundusValidator`) to ensure only fundus-like images enter the GAT inference pipeline, rejecting portraits, blanks, or corrupted files gracefully.
**Setup**: 
- Added heuristic image analysis for brightness, color profile, and circular field mask.
- Added pytest suite (Tests A-G) for various invalid conditions.
- Updated FastAPI endpoint to return early with rejection reason.
- Updated React UI to display rejection states cleanly without prediction components.
**Status**: Completed successfully.
**Results**: 
- 7/7 tests passed.
- Corrupted images are handled gracefully (`decodable=False`).
- Portraits, blank, all-white, and all-dark images are correctly rejected.
- Valid (dummy) fundus images pass through to GAT inference.

## Experiment 3: Deterministic Preprocessing Pipeline (Phase 2)
**Date**: 2026-10-03
**Objective**: Establish a robust, deterministic, non-distorting preprocessing pipeline that prepares images perfectly for downstream vessel extraction.
**Setup**: 
- Implement explicit channel ordering handling (RGBA/Grayscale -> RGB).
- Implement Retinal Field cropping via thresholded bounding box.
- Implement Aspect-Ratio Preserving resize (padded to 512x512).
- Apply CLAHE consistently on Green Channel.
- Normalize vessel representation strictly to float32 [0.0, 1.0].
- Added new `test_preprocessing.py` containing Tests A through I.
**Status**: Completed successfully.
**Results**:
- 14/14 Pytest suite passed (7 API + 7 Preprocessing tests).
- Edge cases (abnormal ratios, strange channel depths, etc.) process perfectly deterministically.

## Experiment 4: Deterministic Structural Extraction (Phase 3)
**Date**: 2026-10-03
**Objective**: Extract exact structural topology (junctions, endpoints) from the vessel map without relying on random fallbacks or hard-coded coordinates.
**Setup**: 
- Replaced arbitrary corner detection (Shi-Tomasi) with mathematical morphological `skeletonize`.
- Implemented neighbor-counting (3x3 convolution) on the skeleton to discover exactly where vessels terminate (1 neighbor) and intersect (>=3 neighbors).
- Merged closely packed junction pixels using dilation and connected components.
- Extracted deterministic coordinates and normalized features.
- Implemented explicit failure handler if `len(candidate_points) == 0`.
**Status**: Completed successfully.
**Results**:
- 18/18 Pytest suite passed (7 API, 7 Preprocessing, 4 Extractor).
- Hardcoded coordinates and random point generation routines were strictly removed.
- Valid coordinates dynamically track the synthetic test structures perfectly.

## Experiment 5: Robust Retinal Graph Construction (Phase 4)
**Date**: 2026-10-03
**Objective**: Construct a deterministic, mathematically reproducible PyTorch Geometric graph from Phase 3 topological features without modifying the downstream GAT yet.
**Setup**: 
- Rebuilt `GraphBuilder` to explicitly handle edge cases ($N=0, 1, \le k$) safely without crashing.
- Modified spatial KNN logic to prevent self-loop duplication when scanning zero-distance duplicate points.
- Extracted explicit Edge Features (`euclidean_distance, relative_x, relative_y`).
- Implemented rich `data.stats` generation (Degree distribution, Density, Connected Components).
- Added `scripts/visualize_graph.py` to trace raw topology -> graph connections visually.
**Status**: Completed successfully.
**Results**:
- 24/24 Pytest suite assertions passed (including new Graph structural bounds tests).
- Visual verification confirmed edges intelligently connect extracted topological nodes.
- PyG `Data` tensors correctly shaped: `x=[N, 4]`, `edge_index=[2, E]`, `edge_attr=[E, 3]`.

## Experiment 6: GAT Model Integration (Phase 5)
**Date**: 2026-10-03
**Objective**: Connect Phase 4 PyG graph to the downstream GAT accurately without allowing false medical predictions to surface.
**Setup**: 
- Re-configured `RetinaGAT` with `in_channels=4` and `edge_dim=3`.
- Validated Layer 1 & 2 dimensional stability with multi-head attention arrays.
- Enforced `global_mean_pool` for deterministic uniform graph summarization.
- Modified `InferencePipeline` to safely ingest `edge_attr` and explicitly nullify class predictions and confidence arrays.
- Appended `raw_logits` alongside structural metadata.
**Status**: Completed successfully.
**Results**:
- 28/28 Pytest suite assertions passed (API, Extractor, Graph, and 4 new Model constraints).
- Edge cases ($N \le 1$) process cleanly through GAT padding self-loops.
- End-to-end visual tests produce zero random tensors, strictly following: `Float32 Matrix → Topology Filter → PyG Structure → Torch Linear Classifier`.
