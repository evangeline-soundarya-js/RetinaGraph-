from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import RetinaGraphDataset, build_dataset_index, split_dataset
from src.models.gat_model import RetinaGAT


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def audit_split_integrity(split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    train = split.get("train", [])
    validation = split.get("validation", [])
    test = split.get("test", [])

    train_ids = {item["image_id"] for item in train}
    val_ids = {item["image_id"] for item in validation}
    test_ids = {item["image_id"] for item in test}

    train_paths = {item["image_path"] for item in train}
    val_paths = {item["image_path"] for item in validation}
    test_paths = {item["image_path"] for item in test}

    patient_train = {item["patient_id"] for item in train if item.get("patient_id") is not None}
    patient_val = {item["patient_id"] for item in validation if item.get("patient_id") is not None}
    patient_test = {item["patient_id"] for item in test if item.get("patient_id") is not None}

    return {
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "duplicate_image_ids": bool(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids),
        "duplicate_image_paths": bool(train_paths & val_paths or train_paths & test_paths or val_paths & test_paths),
        "patient_overlap_train_validation": bool(patient_train & patient_val),
        "patient_overlap_train_test": bool(patient_train & patient_test),
        "patient_overlap_validation_test": bool(patient_val & patient_test),
        "patient_level_separation": not (patient_train & patient_val) and not (patient_train & patient_test) and not (patient_val & patient_test),
        "has_patient_ids": bool(patient_train or patient_val or patient_test),
    }


def load_best_checkpoint(checkpoint_dir: str = "checkpoints", checkpoint_name: str = "best_model.pt") -> Path:
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    return checkpoint_path


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = RetinaGAT(
        in_channels=config.get("in_channels", 4),
        hidden_channels=config.get("hidden_channels", 16),
        num_classes=config.get("num_classes", 2),
        heads=config.get("heads", 2),
        edge_dim=config.get("edge_dim", 3),
        dropout=config.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config


def evaluate_model(model, dataset: Sequence[Any], device: str = "cpu") -> Dict[str, Any]:
    model.to(device)
    model.eval()
    predictions = []
    probs = []
    labels = []
    ids = []

    with torch.no_grad():
        for item in dataset:
            if isinstance(item, dict):
                graph = item["graph"]
                sample_id = item.get("image_id", item.get("image_path", "unknown"))
                true_label = int(item.get("label", 0))
            else:
                graph = item
                sample_id = str(getattr(graph, "image_id", getattr(graph, "image_path", "unknown")))
                true_label = int(getattr(graph, "y", 0).item() if hasattr(getattr(graph, "y", 0), "item") else graph.label)

            if graph.x is None or graph.x.numel() == 0:
                logits = torch.zeros((1, model.config["num_classes"]), device=device)
                out_probs = torch.softmax(logits, dim=1)
                pred_label = 0
            else:
                logits, _ = model(graph.x.to(device), graph.edge_index.to(device), edge_attr=graph.edge_attr.to(device) if hasattr(graph, "edge_attr") and graph.edge_attr.numel() > 0 else None, batch=None)
                out_probs = torch.softmax(logits, dim=1)
                pred_label = int(torch.argmax(out_probs, dim=1).item())

            predictions.append(pred_label)
            probs.append(out_probs.cpu().numpy()[0].tolist())
            labels.append(true_label)
            ids.append(sample_id)

    return {"image_id": ids, "true_label": labels, "predicted_label": predictions, "probabilities": probs}


class GraphFeatureDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = list(graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        x = graph.x
        if x.numel() == 0:
            feature = np.zeros((4,), dtype=np.float32)
        else:
            feature = np.concatenate([
                x.mean(dim=0).cpu().numpy().astype(np.float32),
                np.array([
                    getattr(graph, "stats", {}).get("num_nodes", 0),
                    getattr(graph, "stats", {}).get("num_edges", 0),
                    getattr(graph, "stats", {}).get("avg_degree", 0.0),
                    getattr(graph, "stats", {}).get("density", 0.0),
                ], dtype=np.float32),
            ])
        label = int(getattr(graph, "y", torch.tensor(0)).item()) if hasattr(getattr(graph, "y", torch.tensor(0)), "item") else int(graph.label)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class GraphFeatureBaseline(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def train_baseline(train_graphs, num_classes: int, epochs: int = 5, lr: float = 1e-3):
    dataset = GraphFeatureDataset(train_graphs)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    feature_dim = len(dataset[0][0])
    model = GraphFeatureBaseline(feature_dim, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    return model


def save_evaluation_outputs(results_dir: Path, predictions: Dict[str, Any], metrics: Dict[str, Any], confusion_matrix_data: Optional[Tuple[List[int], np.ndarray]] = None):
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "test_predictions.csv"
    metrics_path = results_dir / "metrics.json"

    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "true_label", "predicted_label", "probability"])
        for image_id, true_label, pred_label, prob in zip(predictions["image_id"], predictions["true_label"], predictions["predicted_label"], predictions["probabilities"]):
            writer.writerow([image_id, true_label, pred_label, json.dumps(prob)])

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    if confusion_matrix_data is not None and plt is not None:
        labels, matrix = confusion_matrix_data
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(results_dir / "confusion_matrix.png")
        plt.close(fig)


def evaluate_checkpoint(checkpoint_path: str | Path, dataset: Sequence[Any], device: str = "cpu"):
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    return evaluate_model(model, dataset, device=device), config
