# 标准库
import os
import math
import random
import csv


import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC, LinearSVC
import umap
from torchvision import transforms

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_avg_saliency_as_classification(model, dataset, num_classes, save_dir, max_samples_per_class=500):
    """
    Compute average saliency maps for each class using up to 100 samples per class.
    Saliency is computed by backpropagating the predicted class score.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    saliency_sums = defaultdict(lambda: None)
    class_counts = defaultdict(int)

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = int(label)

        if class_counts[label] >= max_samples_per_class:
            continue  # Skip if already have enough samples for this class

        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True

        # Forward pass
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_score = output[0, pred_class]

        # Backward pass
        model.zero_grad()
        pred_score.backward()

        # Compute saliency
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Accumulate saliency for the true label
        if saliency_sums[label] is None:
            saliency_sums[label] = saliency
        else:
            saliency_sums[label] += saliency
        class_counts[label] += 1

        # Optional early break: all classes reached 100
        if len(class_counts) == num_classes and all(v >= max_samples_per_class for v in class_counts.values()):
            break

    # Save averaged saliency maps
    for class_label in saliency_sums:
        avg_saliency = saliency_sums[class_label] / class_counts[class_label]

        # Normalize
        avg_saliency = avg_saliency - avg_saliency.min()
        avg_saliency = avg_saliency / (avg_saliency.max() + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(avg_saliency, cmap='hot')
        plt.axis('off')
        # plt.title(f'Avg Saliency: Class {class_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'avg_saliency_class_{class_label}.png'))
        plt.close()

        print(f"Saved average saliency map for class {class_label} ({class_counts[class_label]} samples).")


def visualize_saliency_as_classification(model, dataset, num_classes, save_dir):
    """
    For each class, find the first sample and plot saliency map.
    """
    model.eval()
    # Storage for first sample indices per class
    class_to_index = {}

    # We assume dataset has __getitem__(i) returning (image, label)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)  # ⭐️关键修正
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break

    # Sanity check
    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")

    os.makedirs(save_dir, exist_ok=True)  # Only once outside the loop

    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True

        # Forward pass
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_score = output[0, pred_class]

        # Backward to get saliency
        model.zero_grad()
        pred_score.backward()

        # Compute saliency
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Convert input image for display
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # --- Save original image ---
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        #        plt.title(f'Original Class {class_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_class_{class_label}.png'))
        plt.close()

        # --- Save saliency map ---
        plt.figure(figsize=(4, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        #        plt.title(f'Saliency (Predicted: {pred_class})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'saliency_class_{class_label}.png'))
        plt.close()

        print(f"Saved original and saliency for class {class_label}.")


def visualize_saliency_as_sum_grad(model, dataset, num_classes, save_dir):
    """
    For each class, find the first sample and plot saliency map (embedding output).
    """
    model.eval()
    class_to_index = {}

    # Find first sample per class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)  # ⭐️关键修正
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break

    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")

    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True

        # Forward
        embedding = model(input_tensor)  # Shape: [1, D]

        # We sum all embedding entries
        scalar_score = embedding.sum()

        # Backward
        model.zero_grad()
        scalar_score.backward()

        # Saliency: max over channels
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Convert input image for display
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Plot
        plt.figure(figsize=(8, 4))
        plt.imshow(saliency, cmap='hot')
        # plt.title('Saliency Map (Embedding Sum)')
        plt.axis('off')

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'saliency_class_{class_label}.png'))
        plt.close()
        print(f"Saved saliency for class {class_label}.")


def visualize_saliency_as_grad_sum(model, dataset, num_classes, save_dir):
    """
    For each class, find the first sample and plot saliency map:
    Computes gradient for each output dimension separately, sums them up.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    class_to_index = {}

    # Find first sample per class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)  # 
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break

    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")

    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True

        # Forward pass
        embedding = model(input_tensor)  # Shape: [1, D]

        # Accumulate gradients over all embedding dimensions
        saliency_accum = torch.zeros_like(input_tensor)

        for dim in range(embedding.shape[1]):
            scalar_output = embedding[0, dim]
            model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()

            scalar_output.backward(retain_graph=True)
            saliency_accum += input_tensor.grad.data.abs()

        # Collapse across RGB channels using max
        saliency, _ = torch.max(saliency_accum, dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Convert image for plotting
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # --- Save original image ---
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        #        plt.title(f'Original Class {class_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_class_{class_label}.png'))
        plt.close()

        # --- Save saliency map ---
        plt.figure(figsize=(4, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        #        plt.title(f'Saliency (Grad Sum)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'saliency_class_{class_label}.png'))
        plt.close()

        print(f"Saved original and saliency for class {class_label}.")


def visualize_avg_saliency_as_grad_sum(model, dataset, num_classes, save_dir, max_samples_per_class=500):
    """
    For each class, compute average saliency map from up to 100 samples.
    Saliency is computed by summing gradients for each embedding dimension separately.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Storage for saliency sums and counts
    saliency_sums = defaultdict(lambda: None)
    class_counts = defaultdict(int)

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = int(label)

        if class_counts[label] >= max_samples_per_class:
            continue  # Skip if we've already collected 100 samples for this class

        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True

        # Forward pass to get embedding
        embedding = model(input_tensor)  # Shape: [1, D]

        # Compute saliency by summing per-dimension gradients
        saliency_accum = torch.zeros_like(input_tensor)

        for dim in range(embedding.shape[1]):
            scalar_output = embedding[0, dim]
            model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            scalar_output.backward(retain_graph=True)
            saliency_accum += input_tensor.grad.data.abs()

        # Collapse channels using max
        saliency, _ = torch.max(saliency_accum, dim=1)
        saliency = saliency.squeeze().cpu().numpy()

        # Accumulate
        if saliency_sums[label] is None:
            saliency_sums[label] = saliency
        else:
            saliency_sums[label] += saliency
        class_counts[label] += 1

        # Optional early stop if all classes reached 100
        if len(class_counts) == num_classes and all(c >= max_samples_per_class for c in class_counts.values()):
            break

    # Save average saliency maps
    for class_label in saliency_sums:
        avg_saliency = saliency_sums[class_label] / class_counts[class_label]

        # Normalize
        avg_saliency = avg_saliency - avg_saliency.min()
        avg_saliency = avg_saliency / (avg_saliency.max() + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(avg_saliency, cmap='hot')
        plt.axis('off')
        # plt.title(f'Avg Grad-Sum (Top {class_counts[class_label]}) - Class {class_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'avg_gradsum_top{class_counts[class_label]}_class_{class_label}.png'))
        plt.close()

        print(f"Saved average grad-sum saliency for class {class_label} using {class_counts[class_label]} samples.")


def compute_smoothgrad_saliency(model, input_tensor, mode='classification', n_samples=50, noise_std=0.1):
    """
    Compute SmoothGrad saliency map.

    mode:
        - 'classification': uses predicted class
        - 'embedding_sum': sum over embedding dimensions
        - 'embedding_separate': sum of per-dimension gradients
    """
    model.eval()
    saliency_accum = torch.zeros_like(input_tensor)

    for _ in range(n_samples):
        noise = torch.normal(mean=0, std=noise_std, size=input_tensor.shape).cuda()
        noisy_input = (input_tensor + noise).detach()
        noisy_input.requires_grad = True

        if mode == 'classification':
            output = model(noisy_input)
            pred_class = output.argmax(dim=1).item()
            scalar = output[0, pred_class]
            model.zero_grad()
            scalar.backward()
            saliency = noisy_input.grad.abs()

        elif mode == 'embedding_sum':
            embedding = model(noisy_input)
            scalar = embedding.sum()
            model.zero_grad()
            scalar.backward()
            saliency = noisy_input.grad.abs()

        elif mode == 'embedding_separate':
            embedding = model(noisy_input)
            saliency = torch.zeros_like(noisy_input)
            for dim in range(embedding.shape[1]):
                scalar = embedding[0, dim]
                model.zero_grad()
                if noisy_input.grad is not None:
                    noisy_input.grad.zero_()
                scalar.backward(retain_graph=True)
                saliency += noisy_input.grad.abs()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        saliency_accum += saliency

    saliency_mean = saliency_accum / n_samples
    # Collapse channels
    saliency_final, _ = torch.max(saliency_mean, dim=1)
    return saliency_final.squeeze().cpu().numpy()


def visualize_smoothed_saliency_as_grad_sum(model, dataset, num_classes, save_dir):
    """
    SmoothGrad version: For each class, compute saliency by summing gradients across
    all embedding dimensions separately (embedding_separate mode).
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    class_to_index = {}

    # Find the first sample for each class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break

    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")

    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()

        # Compute SmoothGrad saliency map
        saliency = compute_smoothgrad_saliency(
            model, input_tensor, mode='embedding_separate', n_samples=50, noise_std=0.1
        )

        # Convert input image for display
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # --- Save original image ---
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        # plt.title(f'Original Class {class_label}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_class_{class_label}.png'))
        plt.close()

        # --- Save SmoothGrad saliency map ---
        plt.figure(figsize=(4, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        # plt.title('SmoothGrad (Grad Sum)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'smooth_saliency_gradsum_class_{class_label}.png'))
        plt.close()

        print(f"Saved original and SmoothGrad (Grad Sum) saliency for class {class_label}.")


def visualize_smoothed_saliency_all_classes_one_figure(model, dataset, num_classes, save_dir):
    """
    SmoothGrad version: embedding_separate mode, all classes in one figure.
    """
    model.eval()
    class_to_index = {}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break

    fig, axes = plt.subplots(nrows=num_classes, ncols=1, figsize=(4, num_classes * 3))

    for row_idx, class_label in enumerate(sorted(class_to_index.keys())):
        idx = class_to_index[class_label]
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()

        saliency = compute_smoothgrad_saliency(
            model, input_tensor, mode='embedding_separate', n_samples=50, noise_std=0.1
        )

        ax = axes[row_idx] if num_classes > 1 else axes
        ax.imshow(saliency, cmap='hot')
        # ax.set_title(f'Class {class_label} Smooth Saliency')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'smooth_all_classes.png'))
    plt.close()
    print(f"Saved combined SmoothGrad saliency figure to {save_dir}")
