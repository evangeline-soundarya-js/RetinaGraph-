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
