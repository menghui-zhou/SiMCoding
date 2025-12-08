import argparse
import os

import torch
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import dataset
import functionals as F
import utils


# ========= IMBALANCED-AWARE METRICS =========
def evaluate_metrics(y_true, y_pred, labels=None):
    """
    Returns a dict of metrics suited for imbalanced classification.
    - accuracy (for reference)
    - balanced_accuracy (macro recall)
    - macro_f1, weighted_f1
    - per-class precision, recall, f1, support (arrays aligned to `labels` order if provided)
    - confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    metrics["per_class_precision"] = p
    metrics["per_class_recall"] = r
    metrics["per_class_f1"] = f1
    metrics["per_class_support"] = s

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics


def print_metrics(name, metrics, labels=None, max_classes_to_show=20):
    print(f"\n=== {name} ===")
    print(f"accuracy:          {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1:          {metrics['macro_f1']:.4f}")
    print(f"weighted_f1:       {metrics['weighted_f1']:.4f}")

    # Per-class (truncate display if many classes)
    pcs = len(metrics["per_class_f1"])
    show = min(pcs, max_classes_to_show)
    if labels is None:
        labels = np.arange(pcs)
    print("\nper-class (showing first", show, "):")
    for i in range(show):
        print(
            f" class {labels[i]}: "
            f"prec={metrics['per_class_precision'][i]:.3f} "
            f"rec={metrics['per_class_recall'][i]:.3f} "
            f"f1={metrics['per_class_f1'][i]:.3f} "
            f"(n={metrics['per_class_support'][i]})"
        )

    # Confusion matrix (small preview)
    cm = metrics["confusion_matrix"]
    if cm.shape[0] <= max_classes_to_show and cm.shape[1] <= max_classes_to_show:
        print("\nconfusion matrix:\n", cm)


# ========= HELPERS =========
def _unit_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row (for cosine kNN)."""
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(nrm, eps, None)


# ========= MODELS (updated to use balanced metrics) =========
def svm(train_features, train_labels, test_features, test_labels):
    # class_weight balanced
    clf = LinearSVC(verbose=0, random_state=10, class_weight="balanced")
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    metrics = evaluate_metrics(test_labels, y_pred)
    print_metrics("SVM (LinearSVC, balanced)", metrics)
    return metrics


def knn(train_features, train_labels, test_features, test_labels, k=5):
    """k-NN using cosine similarity (with L2 row normalization)."""
    # Normalize to make dot product == cosine
    train_n = _unit_normalize_rows(np.asarray(train_features))
    test_n = _unit_normalize_rows(np.asarray(test_features))

    sim_mat = train_n @ test_n.T  # (N_train, N_test)
    topk_idx = torch.from_numpy(sim_mat).topk(k=k, dim=0).indices.numpy()  # (k, N_test)
    topk_pred = train_labels[topk_idx]  # (k, N_test)
    # majority vote along axis 0
    # use numpy apply_along_axis to mode
    from scipy.stats import mode
    test_pred = mode(topk_pred, axis=0, keepdims=False).mode  # (N_test,)

    metrics = evaluate_metrics(test_labels, test_pred)
    print_metrics(f"kNN (k={k}, cosine)", metrics)
    return metrics


def nearsub(train_features, train_labels, test_features, test_labels, n_comp=10):
    """Nearest subspace via TruncatedSVD."""
    scores_svd = []
    classes = np.unique(test_labels)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, classes=classes, stack=False)
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in np.arange(len(classes)):
        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) @ (test_features).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        scores_svd.append(score_svd_j)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    y_pred = classes[test_predict_svd]
    metrics = evaluate_metrics(test_labels, y_pred)
    print_metrics("Nearest Subspace (SVD)", metrics)
    return metrics


def nearsub_pca(train_features, train_labels, test_features, test_labels, n_comp=10):
    """Nearest subspace via PCA."""
    scores_pca = []
    classes = np.unique(test_labels)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, classes=classes, stack=False)
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in np.arange(len(classes)):
        pca = PCA(n_components=n_comp).fit(features_sort[j])
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) @ (test_features - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)
        scores_pca.append(score_pca_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    y_pred = classes[test_predict_pca]
    metrics = evaluate_metrics(test_labels, y_pred)
    print_metrics("Nearest Subspace (PCA)", metrics)
    return metrics


def baseline(train_features, train_labels, test_features, test_labels):
    # Use class_weight='balanced' where supported
    test_models = {
        'log_l2': LogisticRegression(
            penalty='l2', C=1.0, max_iter=10000, multi_class='multinomial',
            solver='lbfgs', class_weight='balanced'  # n_jobs not used by lbfgs
        ),
        'SVM_linear': LinearSVC(max_iter=10000, random_state=42, class_weight='balanced'),
        'SVM_RBF': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced'),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'SGD_log': SGDClassifier(loss='log', max_iter=10000, random_state=42, class_weight='balanced'),
    }
    for model_name, model in test_models.items():
        model.fit(train_features, train_labels)
        y_pred = model.predict(test_features)
        metrics = evaluate_metrics(test_labels, y_pred)
        print_metrics(f"Baseline - {model_name}", metrics)

def logistic_softmax(train_features, train_labels, test_features, test_labels,
                     C=1.0, tol=1e-4, max_iter=1000, random_state=0):
    """
    Multinomial logistic regression (softmax) with L2 regularization and
    class_weight='balanced' for imbalanced data.

    Returns:
        metrics (dict): same format as other classifiers (test-set only).
    """
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        tol=tol,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(train_features, train_labels)

    # Test-set predictions only (to match other classifier functions)
    y_pred = clf.predict(test_features)
    metrics = evaluate_metrics(test_labels, y_pred)

    print_metrics("Logistic-Softmax (multinomial, balanced)", metrics)
    return metrics