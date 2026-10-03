import csv
import io
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from src.data.dataset import (
    RetinaGraphDataset,
    build_dataset_index,
    compute_class_distribution,
    split_dataset,
)
from src.models.gat_model import RetinaGAT


@pytest.fixture
def sample_metadata_dir(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    metadata_dir = data_dir / "metadata"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    image_paths = []
    for idx in range(6):
        p = raw_dir / f"img_{idx}.png"
        img = np.full((256, 256, 3), 30, dtype=np.uint8)
        rr, cc = np.ogrid[:256, :256]
        center = (128, 128)
        mask = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= 100**2
        img[mask] = (20, 70, 30)
        from PIL import Image
        Image.fromarray(img).save(p)
        image_paths.append(str(p))

    rows = [
        {"image_id": "img_0", "image_path": str(image_paths[0]), "label": 0, "dataset_source": "synthetic", "patient_id": "p01"},
        {"image_id": "img_1", "image_path": str(image_paths[1]), "label": 1, "dataset_source": "synthetic", "patient_id": "p01"},
        {"image_id": "img_2", "image_path": str(image_paths[2]), "label": 0, "dataset_source": "synthetic", "patient_id": "p02"},
        {"image_id": "img_3", "image_path": str(image_paths[3]), "label": 1, "dataset_source": "synthetic", "patient_id": "p03"},
        {"image_id": "img_4", "image_path": str(image_paths[4]), "label": 0, "dataset_source": "synthetic", "patient_id": "p04"},
        {"image_id": "img_5", "image_path": str(image_paths[5]), "label": 1, "dataset_source": "synthetic", "patient_id": "p05"},
    ]

    csv_path = metadata_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "image_path", "label", "dataset_source", "patient_id"])
        writer.writeheader()
        writer.writerows(rows)

    return data_dir, csv_path


def test_dataset_metadata_loading_and_label_parsing(sample_metadata_dir):
    data_dir, csv_path = sample_metadata_dir
    samples = build_dataset_index(str(data_dir), metadata_file=str(csv_path))
    assert len(samples) == 6
    assert samples[0]["label"] == 0
    assert samples[0]["image_id"] == "img_0"
    assert samples[0]["dataset_source"] == "synthetic"

    dist = compute_class_distribution(samples)
    assert dist[0] == 3
    assert dist[1] == 3


def test_split_dataset_is_reproducible_and_patient_level(sample_metadata_dir):
    data_dir, csv_path = sample_metadata_dir
    samples = build_dataset_index(str(data_dir), metadata_file=str(csv_path))
    split_a = split_dataset(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    split_b = split_dataset(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    assert split_a == split_b

    train_ids = {row["image_id"] for row in split_a["train"]}
    val_ids = {row["image_id"] for row in split_a["validation"]}
    test_ids = {row["image_id"] for row in split_a["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_dataset_item_builds_graph_and_label(tmp_path):
    img_path = tmp_path / "sample.png"
    from PIL import Image
    img = np.full((256, 256, 3), 20, dtype=np.uint8)
    rr, cc = np.ogrid[:256, :256]
    center = (128, 128)
    mask = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= 90**2
    img[mask] = (50, 120, 70)
    Image.fromarray(img).save(img_path)

    samples = [{
        "image_path": str(img_path),
        "label": 1,
        "image_id": "sample-1",
        "dataset_source": "synthetic",
        "patient_id": "p42",
        "split": "train",
    }]
    dataset = RetinaGraphDataset(samples, cache_dir=tmp_path / "cache", graph_version="v1")
    item = dataset[0]
    assert item.x.ndim == 2
    assert item.edge_index.ndim == 2
    assert item.y.ndim == 0 or item.y.numel() == 1
    assert item.image_id == "sample-1"


def test_graph_batching_and_model_forward():
    model = RetinaGAT(in_channels=4, hidden_channels=8, num_classes=2, heads=2, edge_dim=3, dropout=0.0)
    model.eval()

    x = torch.rand((6, 4), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype=torch.long)
    edge_attr = torch.rand((5, 3), dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    with torch.no_grad():
        out, _ = model(x, edge_index, edge_attr=edge_attr, batch=batch)
    assert out.shape == (2, 2)

    batch_data = Batch()
    batch_data.x = x
    batch_data.edge_index = edge_index
    batch_data.edge_attr = edge_attr
    batch_data.batch = batch
    batch_data.y = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        out2, _ = model(batch_data.x, batch_data.edge_index, edge_attr=batch_data.edge_attr, batch=batch_data.batch)
    assert out2.shape == (2, 2)


def test_loss_backward_and_optimizer_update():
    model = RetinaGAT(in_channels=4, hidden_channels=8, num_classes=2, heads=2, edge_dim=3, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.rand((6, 4), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype=torch.long)
    edge_attr = torch.rand((5, 3), dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    y = torch.tensor([0, 1], dtype=torch.long)

    out, _ = model(x, edge_index, edge_attr=edge_attr, batch=batch)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


def test_checkpoint_save_load(tmp_path):
    model = RetinaGAT(in_channels=4, hidden_channels=8, num_classes=2, heads=2, edge_dim=3, dropout=0.0)
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3, "val_loss": 0.25}, ckpt)
    loaded = torch.load(ckpt, map_location="cpu")
    assert loaded["epoch"] == 3
    assert "model_state_dict" in loaded
