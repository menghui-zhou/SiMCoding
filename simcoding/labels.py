"""Original CIFAR-20 mapping and uniform label corruption."""
import numpy as np
import torch

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3,
                      9, 7, 11, 6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
                      12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16,
                      4, 17, 4, 2, 0, 17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12, 1,
                      9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 16, 19, 2, 4, 6,
                      19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
    return np.array(coarse_targets)[targets]

def _get_labels_array(ds):
    # For ImageFolder, labels used by __getitem__ live in ds.samples (and ds.imgs).
    if hasattr(ds, "samples") and isinstance(ds.samples, list) and len(ds.samples) > 0:
        return np.array([lab for _, lab in ds.samples], dtype=int)
    # Fallback for CIFAR/MNIST-style datasets
    return np.array(ds.targets, dtype=int)

def _set_label(ds, idx, new_lab: int):
    # Update the true source-of-truth for labels
    if hasattr(ds, "samples") and isinstance(ds.samples, list) and len(ds.samples) > 0:
        p, _ = ds.samples[idx]
        ds.samples[idx] = (p, int(new_lab))
        if hasattr(ds, "imgs"):  # keep alias in sync for older torchvision
            ds.imgs[idx] = (p, int(new_lab))
    # Keep ds.targets in sync and as list[int]
    if hasattr(ds, "targets"):
        if not isinstance(ds.targets, list):
            ds.targets = list(map(int, np.array(ds.targets, dtype=int).tolist()))
        ds.targets[idx] = int(new_lab)

def corrupt_labels_uniform(ds, ratio: float, seed: int = 42, include_self: bool = True):
    """
    Corrupt exactly `ratio` of samples by resampling their label from a uniform
    distribution over classes. If `include_self=True`, the new label may equal
    the old one (expected unchanged fraction ≈ 1/C).

    Works for ImageFolder (updates samples/imgs) and CIFAR/MNIST (updates targets).
    """
    assert 0 <= ratio <= 1
    rng = np.random.default_rng(seed)

    # Save originals once for sanity checks
    if not hasattr(ds, "targets_original"):
        ds.targets_original = _get_labels_array(ds).copy()

    labels = _get_labels_array(ds)
    n = len(labels)
    if n == 0:
        print("[LabelNoise] dataset is empty; nothing to corrupt.")
        return ds

    # Infer #classes robustly
    num_classes = getattr(ds, "num_classes", int(labels.max()) + 1)

    n_rand = int(round(n * ratio))
    idxs = rng.choice(n, size=n_rand, replace=False)

    if include_self:
        # Uniform over all classes [0, C-1], so some will remain unchanged.
        new_labels = rng.integers(0, num_classes, size=n_rand)
    else:
        # Force a different class (for reference)
        new_labels = rng.integers(0, num_classes - 1, size=n_rand)
        # bump >= old to skip the original class without rejection
        new_labels = np.where(new_labels >= labels[idxs], new_labels + 1, new_labels)

    # Apply updates
    for i, nl in zip(idxs, new_labels):
        _set_label(ds, int(i), int(nl))

    # Sanity report based on what __getitem__ will use
    used_labels_after = _get_labels_array(ds)
    changed = (used_labels_after != ds.targets_original).sum()
    eff = changed / n
    expect_eff = ratio * (1.0 - (1.0 / num_classes)) if include_self else ratio
    print(f"[LabelNoise] requested={ratio:.3f}, include_self={include_self} "
          f"actual_changed={changed}/{n} ({eff:.2%}), "
          f"expected≈{expect_eff:.2%} (C={num_classes})")
    return ds
