# Experiment Log

## Experiment 1: End-to-End Pipeline Validation
**Date**: 2026-10-03
**Objective**: Validate that a fundus image can pass through the entire pipeline (Preprocessing -> Features -> Graph -> GAT -> FastAPI -> React) without throwing shape or dimension errors.
**Setup**: 
- Model: RetinaGAT (4 input features: normalized x, normalized y, intensity, vesselness)
- Graph: K=5 Nearest Neighbors based on spatial distance of extracted keypoints.
- Max Keypoints: 150
**Status**: Ready to run.
**Results**: Pending backend and frontend launch.
