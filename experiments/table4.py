"""Five-task incremental SiM-ReduNet experiments on MNIST and ImageNette."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from simcoding.common import limit_per_class, resolve_device, save_csv, save_json, set_seed
from simcoding.data import imagenette_root
from simcoding.evaluation import classification_metrics, fit_logistic
from simcoding.incremental import imagenette_core, mnist_core


def load_data(args):
    if args.dataset == "mnist":
        transform = transforms.ToTensor()
        train = datasets.MNIST(args.data_root, train=True, transform=transform, download=args.download)
        test = datasets.MNIST(args.data_root, train=False, transform=transform, download=args.download)
    else:
        root = imagenette_root(args.data_root, args.imagenette_dir, args.download)
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
        ])
        train = datasets.ImageFolder(root / "train", transform=transform)
        test = datasets.ImageFolder(root / "val", transform=transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Training and validation class folders must match.")
    if len(set(np.asarray(train.targets).tolist())) != 10:
        raise ValueError("Table 4 requires ten classes.")
    train = limit_per_class(train, args.train_per_class, args.seed)
    test = limit_per_class(test, args.test_per_class, args.seed)

    def materialize(dataset):
        images, labels = [], []
        for x, y in DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers):
            images.append(x.flatten(1).numpy())
            labels.append(y.numpy())
        return np.concatenate(images).astype(np.float32), np.concatenate(labels)

    return *materialize(train), *materialize(test)


@torch.no_grad()
def infer(block, samples, batch_size, dataset):
    """Use the original layer updates, loading each layer once per evaluation."""
    features = torch.nn.functional.normalize(torch.from_numpy(samples), dim=1)
    for layer in range(block.layers):
        if dataset == "mnist":
            E, Cs, gam, _ = block.load_layer(layer)
        else:
            block.load_weights(layer)
            block.load_gam(layer)
            E, Cs, gam = block.E, block.Cs, block.gam
        chunks = []
        for chunk in features.split(batch_size):
            Z = chunk.to(block.device)
            expd = Z @ E.T
            comp = torch.stack([Z @ C.T for C in Cs], dim=0)
            clus, _ = block.nonlinear(comp, gam)
            chunks.append(torch.nn.functional.normalize(Z + block.eta * (expd - clus), dim=1).cpu())
        features = torch.cat(chunks)
    return features.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "imagenette"], required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--imagenette-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--eps2", type=float, default=None, help="Defaults: MNIST 80; ImageNette 300.")
    parser.add_argument("--eta", type=float, default=None, help="Defaults: MNIST 0.5; ImageNette 10.")
    parser.add_argument("--image-size", type=int, default=32, help="ImageNette input size.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Evaluation batch size.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=None)
    parser.add_argument("--output", default="outputs/table4")
    args = parser.parse_args()
    args.eps2 = args.eps2 if args.eps2 is not None else (80.0 if args.dataset == "mnist" else 300.0)
    args.eta = args.eta if args.eta is not None else (0.5 if args.dataset == "mnist" else 10.0)
    if min(args.layers, args.batch_size, args.image_size, args.eps2, args.eta) <= 0:
        parser.error("layers, batch-size, image-size, eps2 and eta must be positive")
    output = Path(args.output) / args.dataset
    if (output / "weights").exists():
        raise FileExistsError(f"Run already exists at {output}; choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    save_json(output / "config.json", dict(vars(args), classifier="LogisticRegression", classifier_C=1.0))
    Xtr, ytr, Xte, yte = load_data(args)
    core = mnist_core if args.dataset == "mnist" else imagenette_core
    block = core.Vector(layers=args.layers, eta=args.eta, eps=args.eps2, device=resolve_device(args.device))
    architecture = core.Architecture([block], model_dir=str(output), num_classes=10, batch_size=args.batch_size)
    rows = []
    for task in range(1, 6):
        new_mask = (ytr >= 2 * (task - 1)) & (ytr < 2 * task)
        architecture.init_loss()
        with torch.no_grad():
            if args.dataset == "mnist":
                block.fit_task_incremental(Xtr[new_mask], ytr[new_mask])
            else:
                Z = block.preprocess(Xtr[new_mask])
                y = torch.tensor(ytr[new_mask], dtype=torch.int64, device=block.device)
                for layer in range(block.layers):
                    Z, _ = block.forward_incremental(layer, Z, y)
        # Classifier refitting uses accumulated training features, never test labels.
        train_mask, test_mask = ytr < 2 * task, yte < 2 * task
        train_features = infer(block, Xtr[train_mask], args.batch_size, args.dataset)
        test_features = infer(block, Xte[test_mask], args.batch_size, args.dataset)
        classifier = fit_logistic(train_features, ytr[train_mask], args.seed)
        labels = yte[test_mask]
        first_task = labels < 2
        all_seen = classification_metrics(classifier, test_features, labels)
        task1_accuracy = 100.0 * float(np.mean(classifier.predict(test_features[first_task]) == labels[first_task]))
        row = {"method": "SiM-ReduNet", "dataset": args.dataset, "task": task,
               "task1_accuracy": task1_accuracy, "seen_accuracy": all_seen["accuracy"]}
        rows.append(row)
        print(row, flush=True)
        save_csv(output / "sessions.csv", rows)
    accuracies = [row["task1_accuracy"] for row in rows]
    summary = {"method": "SiM-ReduNet", "dataset": args.dataset}
    summary.update({f"Task {i+1}": value for i, value in enumerate(accuracies)})
    summary["Avg. Acc."] = float(np.mean(accuracies))
    summary["Drop (%)"] = (100.0 * (accuracies[0] - accuracies[-1]) / accuracies[0]
                            if accuracies[0] else None)
    save_csv(output / "table4.csv", [summary])


if __name__ == "__main__":
    main()
