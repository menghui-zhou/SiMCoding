"""Cleaned training/evaluation loop from train_sup_SiMCoding.py."""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .common import resolve_device, save_csv, save_json, set_seed
from .data import image_datasets, LabeledDataset, RepeatedDataset
from .evaluation import extract_features, fit_logistic, classification_metrics, rank_metrics
from .labels import corrupt_labels_uniform
from .loss import SiMCoding
from .models import ResNet18


def add_training_arguments(parser, dataset):
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--imagenette-dir", default=None, help="Directory containing train/ and val/.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=400 if dataset == "imagenette" else 1000)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-repeats", type=int, default=1,
                        help="Repeated stochastic augmentation; use 5 for fivefold augmentation.")
    parser.add_argument("--train-per-class", type=int, default=None, help="Optional small-run limit.")
    parser.add_argument("--test-per-class", type=int, default=None, help="Optional small-run limit.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional small-run limit per epoch.")


def train(train_loader, model, criterion, optimizer, device, max_batches=None):
    model.train()
    total, count = 0.0, 0
    for index, (images, labels) in enumerate(train_loader):
        if max_batches is not None and index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = model(images)
        loss = criterion(features, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered.")
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        count += 1
    if count == 0:
        raise ValueError("No training batches were processed.")
    return total / count


def run_training(args, dataset, eps2, output, noise=0.0, measure_rank=False):
    if args.epochs < 1 or args.batch_size < 2 or args.feature_dim < 1:
        raise ValueError("epochs/feature-dim must be positive and batch-size must be at least 2.")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output = Path(output)
    if (output / "checkpoint.pt").exists():
        raise FileExistsError(f"Run already exists at {output}; choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    train_set, train_eval, test_set, num_classes = image_datasets(dataset, args)
    train_set = LabeledDataset(train_set)
    corrupt_labels_uniform(train_set, noise, seed=args.seed,
                           include_self=not getattr(args, "exclude_original_label", False))
    # Logistic regression is fitted to training features and their training labels only.
    train_eval = LabeledDataset(train_eval)
    train_eval.targets = list(train_set.targets)
    augmented = RepeatedDataset(train_set, args.train_repeats)
    train_loader = DataLoader(augmented, batch_size=min(args.batch_size, len(augmented)),
                              shuffle=True, drop_last=True, num_workers=args.workers)
    eval_loader = DataLoader(train_eval, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.workers)
    model = ResNet18(args.feature_dim).to(device)
    criterion = SiMCoding(eps2)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    save_json(output / "config.json", dict(vars(args), dataset=dataset, eps2=eps2,
              label_noise=noise, classifier="LogisticRegression", classifier_C=1.0))
    history = []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader, model, criterion, optimizer, device, args.max_batches)
        history.append({"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.6f}", flush=True)
    train_features, train_labels = extract_features(model, eval_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    classifier = fit_logistic(train_features, train_labels, args.seed)
    train_metrics = classification_metrics(classifier, train_features, train_labels)
    test_metrics = classification_metrics(classifier, test_features, test_labels)
    result = {"method": "SiMCoding", "dataset": dataset, "eps2": eps2,
              "label_noise": noise, "seed": args.seed,
              "train_accuracy": train_metrics["accuracy"],
              "test_accuracy": test_metrics["accuracy"],
              "train_balanced_accuracy": train_metrics["balanced_accuracy"],
              "test_balanced_accuracy": test_metrics["balanced_accuracy"]}
    if measure_rank:
        result.update(rank_metrics(train_features, train_labels, cutoff=0.5))
    result["seconds"] = time.monotonic() - start
    torch.save({"model": model.state_dict(), "eps2": eps2, "feature_dim": args.feature_dim,
                "epoch": args.epochs}, output / "checkpoint.pt")
    save_csv(output / "training.csv", history)
    save_json(output / "metrics.json", result)
    print(result, flush=True)
    return result
