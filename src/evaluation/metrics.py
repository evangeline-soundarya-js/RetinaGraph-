from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_numpy(array: Iterable) -> np.ndarray:
    return np.asarray(list(array), dtype=np.int64)


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None):
    labels = list(labels) if labels is not None else sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    mapping = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[mapping[int(t)], mapping[int(p)]] += 1
    return labels, matrix


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true == y_pred))


def macro_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None) -> Dict[str, float]:
    labels = list(labels) if labels is not None else sorted(set(int(v) for v in y_true) | set(int(v) for v in y_pred))
    per_class = {}
    for label in labels:
        true_mask = np.asarray(y_true, dtype=int) == int(label)
        pred_mask = np.asarray(y_pred, dtype=int) == int(label)
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum((~true_mask) & pred_mask))
        fn = int(np.sum(true_mask & (~pred_mask)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(np.sum(true_mask))
        per_class[int(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    return {"macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "per_class": per_class}


def precision_recall_curve(y_true: Sequence[int], y_score: Sequence[float], positive_label: int = 1):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos_mask = y_true == positive_label
    if pos_mask.sum() == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0

    thresholds = np.unique(np.concatenate(([np.inf], y_score, [-np.inf])))
    precision = []
    recall = []
    for threshold in thresholds:
        pred_positive = y_score >= threshold
        tp = int(np.sum(pred_positive & pos_mask))
        fp = int(np.sum(pred_positive & (~pos_mask)))
        fn = int(np.sum((~pred_positive) & pos_mask))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        precision.append(p)
        recall.append(r)

    precision = np.asarray(precision)
    recall = np.asarray(recall)
    pr_auc = float(np.trapz(precision, recall))
    return precision, recall, thresholds, pr_auc


def roc_auc_score(y_true: Sequence[int], y_score: Sequence[float], positive_label: int = 1) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos_mask = y_true == positive_label
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return 0.5

    ranks = np.argsort(y_score)
    sorted_scores = y_score[ranks]
    sorted_true = y_true[ranks]
    rank_sum = 0.0
    for index, score in enumerate(sorted_scores):
        if sorted_true[index] == positive_label:
            rank_sum += index + 1
    u = pos_mask.sum() * neg_mask.sum()
    a = rank_sum - pos_mask.sum() * (pos_mask.sum() + 1) / 2.0
    auc = a / u
    return float(auc)


def binary_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]] = None, positive_label: int = 1) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    accuracy = accuracy_score(y_true, y_pred)

    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": recall,
        "specificity": specificity,
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score, positive_label=positive_label)
    return metrics
