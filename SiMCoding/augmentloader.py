import os
import time
import sys

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class AugmentLoader:
    """Dataloader that includes augmentation functionality."""

    def __init__(self,
                 dataset,
                 batch_size,
                 sampler="random",
                 transforms=transforms.ToTensor(),
                 num_aug=0,
                 shuffle=False):

        # Save the dataset (train/test split handled outside)
        self.dataset = dataset
        # Total number of samples per batch (including augmented copies)
        self.batch_size = batch_size
        # Transformation(s) to apply to each image
        self.transforms = transforms
        # Sampler type: 'balance' (equal per class) or 'random'
        self.sampler = sampler
        # Number of augmentations per original image (>=1)
        self.num_aug = num_aug
        # Whether to shuffle indices (only for random sampler)
        self.shuffle = shuffle

    def __iter__(self):
        # Balanced sampler: ensure equal number of samples per class
        if self.sampler == "balance":
            sampler = BalanceSampler(self.dataset)
            # Number of *original* images per batch (before augmentation)
            num_img = self.batch_size // self.num_aug
            # Return an iterator object to generate batches
            return _Iter(self, sampler, num_img, self.num_aug)
        # Random sampler: sample images uniformly at random
        elif self.sampler == "random":
            # Truncate dataset so batch size divides evenly
            size = len(self.dataset.targets) // self.batch_size * self.batch_size
            sampler = RandomSampler(self.dataset, size, shuffle=self.shuffle)
            num_img = self.batch_size // self.num_aug
            return _Iter(self, sampler, num_img, self.num_aug)
        else:
            # Invalid sampler name
            raise NameError(f"sampler {self.sampler} not found.")

    def update_labels(self, targets):
        # Update dataset labels (useful for semi-supervised or relabeling)
        self.dataset.targets = targets

    def apply_augments(self, sample):
        # If no augmentation is specified, just apply the base transform once
        if self.num_aug is None:
            return self.transforms(sample).unsqueeze(0)
        # First augmentation = identity (raw tensor)
        batch_imgs = [transforms.ToTensor()(sample).unsqueeze(0)]
        # Apply remaining augmentations (randomized)
        for _ in range(self.num_aug - 1):
            transformed = self.transforms(sample)
            batch_imgs.append(transformed.unsqueeze(0))
        # Concatenate augmentations along batch dimension
        return torch.cat(batch_imgs, axis=0)


class _Iter():
    """Iterator that yields batches from AugmentLoader."""

    def __init__(self, loader, sampler, num_img, num_aug, size=None):
        self.loader = loader  # Reference to parent loader
        self.sampler = sampler  # Sampler object (balance/random)
        self.num_img = num_img  # Number of original images per batch
        self.num_aug = num_aug  # Number of augmentations per image
        self.size = size  # Dataset size (optional)

    def __next__(self):
        # Stop iteration if sampler says dataset is exhausted
        if self.sampler.stop():
            raise StopIteration
        batch_imgs = []
        batch_lbls = []
        batch_idx = []
        # Get original samples and labels from sampler
        sampled_imgs, sampled_lbls = self.sampler.sample(self.num_img)
        # For each image, apply augmentation pipeline
        for i in range(self.num_img):
            img_augments = self.loader.apply_augments(sampled_imgs[i])
            batch_imgs.append(img_augments)
            # Repeat label for each augmented version
            batch_lbls.append(np.repeat(sampled_lbls[i], self.num_aug))
            # Record image index for tracking
            batch_idx.append(np.repeat(i, self.num_aug))
        # Concatenate into tensors
        batch_imgs = torch.cat(batch_imgs, axis=0).float()
        batch_lbls = torch.from_numpy(np.hstack(batch_lbls))
        batch_idx = torch.from_numpy(np.hstack(batch_idx))
        # Return a tuple: (images, labels, indices)
        return (batch_imgs,
                batch_lbls,
                batch_idx)


class BalanceSampler():
    """Sampler ensuring equal samples from each class."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.size = len(self.dataset.targets)
        # Number of distinct classes in dataset
        self.num_classes = np.max(self.dataset.targets) + 1
        self.num_sampled = 0  # Counter for sampled images
        self.sort()  # Group dataset by class

    def sort(self):
        # Organize data into per-class buckets
        sorted_data = [[] for _ in range(self.num_classes)]
        for i, lbl in enumerate(self.dataset.targets):
            sorted_data[lbl].append(self.dataset[i][0])
        self.sorted_data = sorted_data
        # Labels are stored in arrays corresponding to class IDs
        self.sorted_labels = [np.repeat(i, len(sorted_data[i])) for i in range(self.num_classes)]

    def sample(self, num_imgs):
        # Must sample a multiple of num_classes
        num_imgs_per_class = num_imgs // self.num_classes
        assert num_imgs_per_class * self.num_classes == num_imgs, 'cannot sample uniformly'

        batch_imgs, batch_lbls = [], []
        for c in range(self.num_classes):
            img_c, lbl_c = self.sorted_data[c], self.sorted_labels[c]
            # Randomly sample indices within each class (with replacement)
            sample_indices = np.random.choice(len(img_c), num_imgs_per_class)
            for i in sample_indices:
                batch_imgs.append(img_c[i])
                batch_lbls.append(lbl_c[i])
        self.increment_step(num_imgs)
        return batch_imgs, batch_lbls

    def increment_step(self, num_imgs):
        # Update counter of sampled images
        self.num_sampled += num_imgs

    def stop(self):
        # Stop when sampled >= dataset size
        if self.num_sampled < self.size:
            return False
        return True


class RandomSampler():
    """Sampler that randomly selects samples from dataset."""

    def __init__(self, dataset, size, shuffle=False):
        self.dataset = dataset
        self.size = size
        self.shuffle = shuffle
        self.num_sampled = 0
        # Initialize sample index list
        self.sample_indices = self.reset_index()

    def reset_index(self):
        # If shuffle, randomly permute indices; else sequential
        if self.shuffle:
            return np.random.choice(len(self.dataset.targets), self.size, replace=False).tolist()
        else:
            return np.arange(self.size).tolist()

    def sample(self, num_img):
        # Pop off the next indices to form a batch
        indices = [self.sample_indices.pop(0) for _ in range(num_img)]
        batch_imgs, batch_lbls = [], []
        for i in indices:
            img, lbl = self.dataset[i]
            batch_imgs.append(img)
            batch_lbls.append(lbl)
        self.increment_step(num_img)
        return batch_imgs, batch_lbls

    def increment_step(self, num_img):
        # Increase counter of sampled images
        self.num_sampled += num_img

    def stop(self):
        # Stop when all samples used
        if self.num_sampled < self.size:
            return False
        return True
