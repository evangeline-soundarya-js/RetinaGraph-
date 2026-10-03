import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.extractor import FeatureExtractor
from src.graph.builder import GraphBuilder
from src.preprocessing.fundus_validator import FundusValidator
from src.preprocessing.image_processor import ImageProcessor


_LABEL_ALIASES = {
    "0": 0,
    "no dr": 0,
    "normal": 0,
    "healthy": 0,
    "1": 1,
    "dr": 1,
    "positive": 1,
    "present": 1,
    "mild": 1,
    "2": 2,
    "moderate": 2,
    "3": 3,
    "severe": 3,
    "4": 4,
    "proliferative": 4,
    "proliferative diabetic retinopathy": 4,
}


def parse_label(raw_label: Any) -> int:
    if raw_label is None:
        raise ValueError("Label is missing.")

    if isinstance(raw_label, (int, np.integer)):
        return int(raw_label)

    if isinstance(raw_label, float):
        return int(raw_label)

    text = str(raw_label).strip()
    if text == "":
        raise ValueError("Label is empty.")

    normalized = text.lower().replace("_", " ")
    if normalized in _LABEL_ALIASES:
        return _LABEL_ALIASES[normalized]

    for key, value in _LABEL_ALIASES.items():
        if key in normalized:
            return value

    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Unsupported label value: {raw_label!r}") from exc


def build_dataset_index(
    data_dir: str,
    metadata_file: Optional[str] = None,
    image_column: str = "image_path",
    label_column: str = "label",
    id_column: str = "image_id",
    patient_column: str = "patient_id",
    dataset_source_column: str = "dataset_source",
) -> List[Dict[str, Any]]:
    root = Path(data_dir)
    if metadata_file is None:
        candidates = [
            root / "metadata" / "metadata.csv",
            root / "metadata" / "labels.csv",
            root / "metadata" / "train.csv",
            root / "data.csv",
        ]
        chosen = next((item for item in candidates if item.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"No metadata file found under {root}. Set metadata_file explicitly or provide data/metadata/metadata.csv."
            )
        metadata_file = str(chosen)

    csv_path = Path(metadata_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    samples: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file is missing a header row: {csv_path}")

        required = {image_column, label_column}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Metadata file is missing required columns: {missing}")

        for row in reader:
            if row is None:
                continue
            image_value = row.get(image_column)
            if image_value is None or str(image_value).strip() == "":
                continue

            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = (csv_path.parent / image_path).resolve()

            sample = {
                "image_path": str(image_path),
                "label": parse_label(row.get(label_column)),
                "dataset_source": row.get(dataset_source_column, csv_path.stem),
                "image_id": row.get(id_column, image_path.stem),
                "patient_id": row.get(patient_column),
            }
            if sample["patient_id"] is not None and str(sample["patient_id"]).strip() == "":
                sample["patient_id"] = None
            if not Path(sample["image_path"]).exists():
                raise FileNotFoundError(f"Image file listed in metadata was not found: {sample['image_path']}")
            samples.append(sample)

    if not samples:
        raise ValueError(f"No labeled dataset rows were loaded from {csv_path}")
    return samples


def load_dataset_metadata(*args, **kwargs):
    return build_dataset_index(*args, **kwargs)


def compute_class_distribution(samples: Iterable[Dict[str, Any]], include_percentages: bool = False) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for sample in samples:
        label = int(sample["label"])
        counts[label] += 1

    ordered = dict(sorted(counts.items()))
    if not include_percentages:
        return ordered

    total = sum(ordered.values())
    return {
        "counts": ordered,
        "percentages": {label: value / total if total else 0.0 for label, value in ordered.items()},
    }


def split_dataset(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    total = len(samples)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/validation/test ratios must sum to 1.0")

    rng = np.random.default_rng(random_seed)
    patient_ids = [sample.get("patient_id") for sample in samples]
    has_patient_ids = bool(patient_ids) and all(pid is not None for pid in patient_ids)

    if has_patient_ids:
        grouped = defaultdict(list)
        for sample in samples:
            grouped[sample["patient_id"]].append(sample)
        patients = list(grouped.keys())
        rng.shuffle(patients)
        train_count = max(1, int(round(train_ratio * len(patients)))) if len(patients) > 1 else 1
        val_count = max(1, int(round(val_ratio * len(patients)))) if len(patients) > 1 else 0
        test_count = max(0, len(patients) - train_count - val_count)
        if train_count + val_count + test_count < len(patients):
            test_count += len(patients) - (train_count + val_count + test_count)

        train_patients = set(patients[:train_count])
        val_patients = set(patients[train_count : train_count + val_count])
        test_patients = set(patients[train_count + val_count : train_count + val_count + test_count])

        splits = {"train": [], "validation": [], "test": []}
        for patient_id, patient_samples in grouped.items():
            target = "train" if patient_id in train_patients else "validation" if patient_id in val_patients else "test"
            for sample in patient_samples:
                sample_copy = dict(sample)
                sample_copy["split"] = target
                splits[target].append(sample_copy)
        return splits

    shuffled = list(samples)
    rng.shuffle(shuffled)
    train_end = int(round(train_ratio * total))
    val_end = train_end + int(round(val_ratio * total))
    train = shuffled[:train_end]
    validation = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return {
        "train": [{**sample, "split": "train"} for sample in train],
        "validation": [{**sample, "split": "validation"} for sample in validation],
        "test": [{**sample, "split": "test"} for sample in test],
    }


class RetinaGraphDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        cache_dir: Optional[str] = None,
        graph_version: str = "v1",
        max_keypoints: int = 150,
        k_neighbors: int = 5,
    ):
        self.samples = list(samples)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data") / "processed" / "graphs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph_version = graph_version
        self.processor = ImageProcessor(target_size=(512, 512))
        self.extractor = FeatureExtractor(max_keypoints=max_keypoints)
        self.builder = GraphBuilder(k_neighbors=k_neighbors)
        self.validator = FundusValidator()

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, sample: Dict[str, Any]) -> Path:
        payload = {
            "image_path": sample.get("image_path"),
            "label": sample.get("label"),
            "image_id": sample.get("image_id"),
            "graph_version": self.graph_version,
            "k_neighbors": self.builder.k_neighbors,
            "target_size": list(self.processor.target_size),
        }
        digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _build_graph(self, sample: Dict[str, Any]):
        image_path = Path(sample["image_path"])
        with open(image_path, "rb") as handle:
            image_bytes = handle.read()

        try:
            with open(image_path, "rb") as handle:
                img = np.array(__import__("PIL").Image.open(handle))
            validation = self.validator.validate(img)
        except Exception:
            validation = {"status": "rejected", "reason": "Validator failed"}

        if validation.get("status") == "rejected":
            node_features = np.zeros((1, 4), dtype=np.float32)
            keypoints = np.zeros((1, 2), dtype=np.float32)
        else:
            prep = self.processor.process(image_bytes)
            feature_result = self.extractor.extract_features(prep["vessel_representation"])
            if feature_result["status"] == "insufficient_retinal_structure":
                node_features = np.zeros((1, 4), dtype=np.float32)
                keypoints = np.zeros((1, 2), dtype=np.float32)
            else:
                keypoints = feature_result["keypoints"]
                node_features = feature_result["node_features"]

        data = self.builder.build_graph(keypoints, node_features)
        data.image_path = str(image_path)
        data.image_id = sample.get("image_id", image_path.stem)
        data.patient_id = sample.get("patient_id")
        data.dataset_source = sample.get("dataset_source", "unknown")
        data.split = sample.get("split", "unspecified")
        data.label = int(sample["label"])
        data.y = torch.tensor(int(sample["label"]), dtype=torch.long)
        data.num_classes = max(2, int(np.max([sample.get("label", 0), 0])) + 1)
        return data

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        cache_path = self._cache_path(sample)
        if cache_path.exists():
            try:
                graph = torch.load(cache_path, map_location="cpu")
                if hasattr(graph, "y"):
                    return graph
            except Exception:
                pass

        graph = self._build_graph(sample)
        torch.save(graph, cache_path)
        return graph
