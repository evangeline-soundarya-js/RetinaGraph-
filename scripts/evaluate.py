import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import build_dataset_index, split_dataset
from src.evaluation.evaluator import (
    audit_split_integrity,
    evaluate_checkpoint,
    save_evaluation_outputs,
    train_baseline,
)
from src.evaluation.metrics import accuracy_score, binary_classification_metrics, confusion_matrix, macro_metrics


def _safe_softmax(logits):
    logits = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(logits, dim=1).numpy()


def _collect_metrics(y_true, y_pred, y_score):
    labels = sorted(set(y_true) | set(y_pred))
    _, matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }
    if len(labels) == 2:
        metrics.update(binary_classification_metrics(y_true, y_pred, y_score=y_score, positive_label=max(labels)))
    else:
        macro = macro_metrics(y_true, y_pred, labels=labels)
        metrics.update({"macro_precision": macro["macro_precision"], "macro_recall": macro["macro_recall"], "macro_f1": macro["macro_f1"], "per_class": macro["per_class"]})
    return metrics


def run_evaluation(args):
    dataset_root = Path(args.dataset_path or os.getenv("RETINAGRAPH_DATASET_PATH", "data"))
    metadata_file = args.metadata_file or os.getenv("RETINAGRAPH_METADATA_FILE")
    samples = build_dataset_index(str(dataset_root), metadata_file=metadata_file)
    splits = split_dataset(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, random_seed=args.seed)
    audit = audit_split_integrity(splits)
    test_graphs = [
        dataset_item for dataset_item in [
            __import__("src.data.dataset", fromlist=["RetinaGraphDataset"]).RetinaGraphDataset(
                splits["test"],
                cache_dir=str(dataset_root / "processed" / "graphs"),
                graph_version="eval-v1",
            )[index]
            for index in range(len(splits["test"]))
        ]
    ]
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_predictions, config = evaluate_checkpoint(checkpoint_path, test_graphs, device=args.device)
    y_true = model_predictions["true_label"]
    y_pred = model_predictions["predicted_label"]
    y_score = [p[1] if len(p) > 1 else p[0] for p in model_predictions["probabilities"]]
    metrics = _collect_metrics(y_true, y_pred, y_score)
    metrics["audit"] = audit
    metrics["config"] = config
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["seed"] = args.seed

    output_dir = Path(args.output_dir)
    save_evaluation_outputs(output_dir, model_predictions, metrics, confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred))))

    if args.baseline:
        train_graphs = [
            __import__("src.data.dataset", fromlist=["RetinaGraphDataset"]).RetinaGraphDataset(
                splits["train"],
                cache_dir=str(dataset_root / "processed" / "graphs"),
                graph_version="baseline-v1",
            )[index]
            for index in range(len(splits["train"]))
        ]
        baseline_model = train_baseline(train_graphs, num_classes=max(max(y_true), max(y_pred)) + 1, epochs=args.baseline_epochs)
        baseline_features = []
        baseline_true = []
        baseline_pred = []
        baseline_scores = []
        for graph in test_graphs:
            feature = graph.x.mean(dim=0).cpu().numpy()
            logits = baseline_model(torch.tensor(feature, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            baseline_pred.append(pred)
            baseline_true.append(int(graph.y.item()))
            baseline_scores.append(float(probs[0, pred].item()))
        baseline_metrics = _collect_metrics(baseline_true, baseline_pred, baseline_scores)
        metrics["baseline"] = baseline_metrics

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(json.dumps({"audit": audit, "metrics": metrics}, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RetinaGraph GAT checkpoint on the untouched test split.")
    parser.add_argument("--dataset-path", default=os.getenv("RETINAGRAPH_DATASET_PATH", "data"))
    parser.add_argument("--metadata-file", default=os.getenv("RETINAGRAPH_METADATA_FILE"))
    parser.add_argument("--checkpoint-path", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
