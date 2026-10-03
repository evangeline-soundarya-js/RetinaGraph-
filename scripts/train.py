import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from src.data.dataset import RetinaGraphDataset, build_dataset_index, split_dataset
from src.models.gat_model import RetinaGAT


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def build_model(model_cfg: dict):
    return RetinaGAT(
        in_channels=model_cfg.get("in_channels", 4),
        hidden_channels=model_cfg.get("hidden_channels", 16),
        num_classes=model_cfg.get("num_classes", 2),
        heads=model_cfg.get("heads", 2),
        edge_dim=model_cfg.get("edge_dim", 3),
        dropout=model_cfg.get("dropout", 0.2),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_examples += batch.num_graphs

    return total_loss / max(total_examples, 1)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total_examples += batch.num_graphs

    return total_loss / max(total_examples, 1)


def save_checkpoint(path: Path, model, optimizer, epoch, val_loss, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(checkpoint, path)


def run_training(args):
    set_seed(args.seed)
    device = get_device()

    dataset_root = Path(args.dataset_path or os.getenv("RETINAGRAPH_DATASET_PATH", "data"))
    metadata_file = args.metadata_file or os.getenv("RETINAGRAPH_METADATA_FILE")
    samples = build_dataset_index(str(dataset_root), metadata_file=metadata_file)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    split = split_dataset(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, random_seed=args.seed)
    train_samples = split["train"]
    val_samples = split["validation"]

    train_dataset = RetinaGraphDataset(train_samples, cache_dir=str(dataset_root / "processed" / "graphs"), graph_version="train-v1")
    val_dataset = RetinaGraphDataset(val_samples, cache_dir=str(dataset_root / "processed" / "graphs"), graph_version="val-v1")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = int(max(sample["label"] for sample in train_samples)) + 1 if train_samples else 2
    model_cfg = {
        "in_channels": 4,
        "hidden_channels": args.hidden_dim,
        "num_classes": num_classes,
        "heads": args.heads,
        "edge_dim": 3,
        "dropout": args.dropout,
    }
    model = build_model(model_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={args.learning_rate}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(Path("checkpoints") / "best_model.pt", model, optimizer, epoch, val_loss, model_cfg)
        save_checkpoint(Path("checkpoints") / "last_model.pt", model, optimizer, epoch, val_loss, model_cfg)

    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RetinaGraph GAT model on a labeled fundus dataset.")
    parser.add_argument("--dataset-path", default=os.getenv("RETINAGRAPH_DATASET_PATH", "data"), help="Base folder containing raw images and metadata.")
    parser.add_argument("--metadata-file", default=os.getenv("RETINAGRAPH_METADATA_FILE"), help="Optional CSV file with image_path,label,...")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test cap for a small dataset subset.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
