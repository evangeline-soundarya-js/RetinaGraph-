# RetinaGraph AI - Research Prototype

This is a working end-to-end prototype for a graph-based analysis pipeline of retinal fundus images.

> **⚠️ RESEARCH PROTOTYPE ONLY**  
> This application is NOT clinically validated. It does not possess medical diagnostic capabilities. The model weights are currently initialized randomly to demonstrate the graph architecture and pipeline execution.

## Features
- **Input Plausibility Validation**: Heuristic gate to reject non-fundus, blank, or corrupted images early.
- **Image Preprocessing**: Adaptive resizing, green channel extraction, and CLAHE normalization.
- **Feature Extraction**: Structural keypoint and junction extraction using Frangi vesselness and morphological detection.
- **Graph Construction**: Dynamic K-Nearest Neighbors (KNN) graph representation of the retinal structure.
- **GAT Model**: Multi-head Graph Attention Network constructed with PyTorch Geometric (`GATConv`).
- **Explanation/Trust Layer**: Graph structural statistics and regional significance evidence.
- **Web Interface**: A premium React/Vite frontend for visualizing the pipeline results.

## Project Structure
- `backend/`: FastAPI application and API routes.
- `src/`: Core ML and processing pipeline (`preprocessing`, `features`, `graph`, `models`, `inference`).
- `frontend/`: React + Vite UI.
- `tests/`: Pytest suite for end-to-end validation.
- `docs/`: Architecture and research logs.

## Setup Instructions

### 1. Environment & Dependencies

Make sure you have Python 3.10+ and Node.js installed.

```powershell
# Clone or navigate to the directory
cd RetinaGraph

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Running Tests
You can verify the entire pipeline (including Image Processing -> Graph -> GAT Inference) locally:

```powershell
$env:PYTHONPATH="."
.\venv\Scripts\pytest
```

### 3. Starting the Application

You need two terminal windows to run the frontend and backend simultaneously.

**Terminal 1 (Backend - FastAPI)**
```powershell
# Ensure you are in the RetinaGraph root directory
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
The backend API will be available at `http://localhost:8000`.

**Terminal 2 (Frontend - React/Vite)**
```powershell
cd frontend
npm run dev
```
The frontend UI will be available at `http://localhost:5173`. Open this link in your browser to interact with the application.

## Dataset and Training Pipeline

This project now includes a reproducible labeled-dataset training path that reuses the existing preprocessing, vessel feature extraction, graph construction, and GAT model. The dataset path is configurable via either the `RETINAGRAPH_DATASET_PATH` environment variable or the CLI flag on the training script.

Example:

```powershell
$env:RETINAGRAPH_DATASET_PATH = "data"
python scripts/train.py --dataset-path data --metadata-file data/metadata/metadata.csv --epochs 2 --batch-size 2 --max-samples 20
```

The metadata CSV should contain at least:
- `image_path`
- `label`
- `dataset_source`
- `image_id`
- `patient_id` (optional, but strongly preferred for patient-level splitting)

The loader supports integer labels directly and maps common text labels like `No DR`, `Mild`, `Moderate`, and `Severe` to numeric values in a consistent way. If patient IDs are available, the split is performed at the patient level to avoid leakage; otherwise the loader falls back to image-level splitting.

## Limitations & Future Work
- Please review `docs/ARCHITECTURE.md` and `docs/RESEARCH_STATUS.md` for full implementation details.
- The KNN graph builder uses Scipy `cKDTree` as a fallback for PyG `knn_graph` to avoid local C++ compilation issues.
- The model predictions are currently stochastic due to untrained random weights.
- A full clinical dataset and evaluation pipeline are intentionally outside this training-only prototype phase.
