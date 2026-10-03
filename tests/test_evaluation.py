import csv
import json

import numpy as np
import pytest
import torch

from src.data.dataset import build_dataset_index, split_dataset
from src.evaluation.evaluator import audit_split_integrity, load_best_checkpoint, train_baseline
from src.evaluation.metrics import accuracy_score, binary_classification_metrics, confusion_matrix, roc_auc_score


@pytest.fixture
def synthetic_split(tmp_path):
    metadata = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = []
    for index in range(8):
        image_path = images_dir / f"img_{index}.png"
        image = np.full((32, 32, 3), 50, dtype=np.uint8)
        image[8:24, 8:24] = 200
        from PIL import Image
        Image.fromarray(image).save(image_path)
        patient_id = f"p{index // 2}"
        rows.append({
            "image_id": f"img_{index}",
            "image_path": str(image_path),
            "label": 1 if index % 2 else 0,
            "dataset_source": "synthetic",
            "patient_id": patient_id,
        })

    with open(metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "label", "dataset_source", "patient_id"])
        writer.writeheader()
        writer.writerows(rows)

    samples = build_dataset_index(str(tmp_path), metadata_file=str(metadata))
    split = split_dataset(samples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, random_seed=13)
    return split, samples


def test_split_integrity(synthetic_split):
    split, _ = synthetic_split
    audit = audit_split_integrity(split)
    assert audit["train_count"] + audit["validation_count"] + audit["test_count"] == 8
    assert audit["duplicate_image_ids"] is False
    assert audit["duplicate_image_paths"] is False


def test_confusion_matrix_and_metrics():
    true = np.array([0, 1, 1, 0])
    pred = np.array([0, 1, 0, 0])
    labels, matrix = confusion_matrix(true, pred, labels=[0, 1])
    assert matrix.shape == (2, 2)
    assert accuracy_score(true, pred) == pytest.approx(0.75)

    metrics = binary_classification_metrics(true, pred, y_score=np.array([0.9, 0.6, 0.4, 0.8]), positive_label=1)
    assert "precision" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_roc_auc_and_binary_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.7, 0.9])
    auc = roc_auc_score(y_true, y_score, positive_label=1)
    assert auc > 0.8


def test_load_best_checkpoint(tmp_path):
    checkpoint = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}, "config": {"in_channels": 4, "hidden_channels": 8, "num_classes": 2, "heads": 2, "edge_dim": 3, "dropout": 0.0}}, checkpoint)
    assert load_best_checkpoint(str(tmp_path), "best_model.pt").exists()


def test_baseline_training_smoke(tmp_path):
    graphs = []
    for label in [0, 1, 0, 1]:
        graph = type("Graph", (), {})()
        graph.x = torch.rand((3, 4), dtype=torch.float32)
        graph.y = torch.tensor(label, dtype=torch.long)
        graph.stats = {"num_nodes": 3, "num_edges": 2, "avg_degree": 1.5, "density": 0.25}
        graphs.append(graph)
    model = train_baseline(graphs, num_classes=2, epochs=2)
    assert isinstance(model, torch.nn.Module)
    out = model(torch.rand((4,), dtype=torch.float32).unsqueeze(0))
    assert out.shape == (1, 2)


def test_result_serialization(tmp_path):
    data = {"accuracy": 0.8, "labels": [0, 1], "confusion_matrix": [[1, 0], [0, 1]]}
    output = tmp_path / "metrics.json"
    output.write_text(json.dumps(data), encoding="utf-8")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["accuracy"] == 0.8
