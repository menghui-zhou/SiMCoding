"""Logistic regression (C=1) and numerical rank measurements; no plotting."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    for images, target in loader:
        features.append(model(images.to(device)).cpu())
        labels.append(target.cpu())
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def fit_logistic(features, labels, seed=42):
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    classifier.fit(features, labels)
    return classifier


def classification_metrics(classifier, features, labels):
    prediction = classifier.predict(features)
    return {
        "accuracy": 100.0 * accuracy_score(labels, prediction),
        "balanced_accuracy": 100.0 * balanced_accuracy_score(labels, prediction),
    }


def effective_rank(features, cutoff=0.5):
    """Count singular values strictly above cutoff times the largest value."""
    singular_values = np.linalg.svd(features, compute_uv=False)
    if not len(singular_values) or singular_values[0] == 0:
        return 0
    return int(np.count_nonzero(singular_values > cutoff * singular_values[0]))


def rank_metrics(features, labels, cutoff=0.5):
    per_class = [effective_rank(features[labels == c], cutoff) for c in np.unique(labels)]
    return {
        "global_effective_rank": effective_rank(features, cutoff),
        "mean_class_effective_rank": float(np.mean(per_class)),
    }
