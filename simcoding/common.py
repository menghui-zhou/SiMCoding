"""Small helpers shared by the experiment entry points."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    return torch.device(name)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def limit_per_class(dataset, count, seed):
    if count is None:
        return dataset
    if count < 1:
        raise ValueError("The per-class limit must be positive.")
    from torch.utils.data import Subset
    labels = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    indices = np.concatenate([
        rng.permutation(np.flatnonzero(labels == c))[:count]
        for c in np.unique(labels)
    ])
    result = Subset(dataset, indices.tolist())
    result.targets = labels[indices].tolist()
    return result
