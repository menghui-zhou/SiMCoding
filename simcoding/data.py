"""Datasets for representation learning and incremental learning."""
import tarfile
import urllib.request
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .common import limit_per_class
from .labels import sparse2coarse


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def imagenette_root(data_root, explicit=None, download=False):
    root = Path(explicit) if explicit else Path(data_root) / "imagenette" / "imagenette2-320"
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root
    if explicit or not download:
        raise FileNotFoundError(
            f"Expected {root}/train and {root}/val. Set --imagenette-dir or use --download."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "imagenette2-320.tgz"
    if not archive.exists():
        print(f"Downloading ImageNette to {archive}")
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    with tarfile.open(archive, "r:gz") as tf:
        destination = root.parent.resolve()
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError("Unsupported archive member.")
        tf.extractall(destination)
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        raise FileNotFoundError(f"ImageNette extraction failed at {root}")
    return root


def image_datasets(name, args):
    if name == "imagenette":
        root = imagenette_root(args.data_root, args.imagenette_dir, args.download)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(round(args.image_size * 1.25)),
            transforms.CenterCrop(args.image_size), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        train = datasets.ImageFolder(root / "train", transform=train_transform)
        train_eval = datasets.ImageFolder(root / "train", transform=test_transform)
        test = datasets.ImageFolder(root / "val", transform=test_transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Training and validation class folders must match.")
        num_classes = 10
    elif name == "cifar20":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()
        root = Path(args.data_root) / "cifar100"
        train = datasets.CIFAR100(root, train=True, transform=train_transform, download=args.download)
        train_eval = datasets.CIFAR100(root, train=True, transform=test_transform, download=False)
        test = datasets.CIFAR100(root, train=False, transform=test_transform, download=False)
        for dataset in (train, train_eval, test):
            dataset.targets = sparse2coarse(dataset.targets).tolist()
        num_classes = 20
    else:
        raise ValueError(name)
    if len(set(train.targets)) != num_classes:
        raise ValueError(f"Expected {num_classes} classes for {name}.")
    train = limit_per_class(train, args.train_per_class, args.seed)
    train_eval = limit_per_class(train_eval, args.train_per_class, args.seed)
    test = limit_per_class(test, args.test_per_class, args.seed)
    return train, train_eval, test, num_classes


class RepeatedDataset(Dataset):
    """Repeated stochastic augmentation of the same training samples."""

    def __init__(self, dataset, repeats=1):
        if repeats < 1:
            raise ValueError("repeats must be positive.")
        self.dataset, self.repeats = dataset, repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]


class LabeledDataset(Dataset):
    """Keep mutable noisy labels separate from the clean evaluation dataset."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = list(dataset.targets)
        self.num_classes = len(set(self.targets))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image, int(self.targets[index])
