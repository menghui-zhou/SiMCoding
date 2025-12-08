# incremental_mnist.py
import os
import argparse
import numpy as np
from torchvision import datasets, transforms

from redunet import Architecture, Vector
import evaluate
import utils
import functionals as F  # uses your F.normalize

def load_mnist_numpy(root="./data", train=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root=root, train=train, download=True, transform=tfm)
    X = ds.data.numpy().astype(np.float32) / 255.0
    y = ds.targets.numpy().astype(np.int64)
    m = X.shape[0]
    X = X.reshape(m, -1)  # flatten to vectors
    return X, y

def split_tasks_by_class(X, y, pairs=((0,1),(2,3),(4,5),(6,7),(8,9))):
    tasks = []
    for a, b in pairs:
        idx = np.where((y == a) | (y == b))[0]
        tasks.append((X[idx], y[idx]))
    return tasks

def run_inference_through_all_layers(block, Z):
    for layer in range(block.layers):
        block.load_weights(layer)
        block.load_gam(layer)
        expd = Z @ block.E.T
        comp = np.stack([Z @ C.T for C in block.Cs])
        clus, _ = block.nonlinear(comp)
        Z = F.normalize(Z + block.eta * (expd - clus))
    return Z

def incremental_fit_mnist(model_dir, layers=30, eta=0.5, eps=0.1, batch_size=4096):
    # data
    Xtr, ytr = load_mnist_numpy(train=True)
    Xte, yte = load_mnist_numpy(train=False)
    num_classes = 10

    # model (single Vector block)
    vec = Vector(layers, eta=eta, eps=eps)
    arch = Architecture([vec], model_dir=model_dir, num_classes=num_classes, batch_size=batch_size)

    # define tasks: 5 increments, 2 classes each
    train_tasks = split_tasks_by_class(Xtr, ytr)

    # === Incremental training across tasks ===
    for task_id, (Xt, yt) in enumerate(train_tasks):
        print(f"\n=== Task {task_id+1}/{len(train_tasks)}: classes {np.unique(yt)} ===")
        block = arch[0]
        block.load_arch(arch, 0)
        arch.init_loss()

        # features BEFORE layer 0
        Z = block.preprocess(Xt)
        y = yt.copy()

        # Per-layer: update stats with NEW data at input to this layer, recompute E/C/gamma, forward NEW data
        for layer in range(block.layers):
            Z, y_pred = block.forward_incremental(layer, Z, y)

        # save per-task training loss curve
        utils.save_loss(arch.loss_dict, model_dir, f"train_task{task_id+1}")


        # Evaluate on seen classes so far
        seen = np.unique(np.concatenate([tt[1] for tt in train_tasks[:task_id+1]]))
        mask_te = np.isin(yte, seen)
        mask_tr = np.isin(ytr, seen)  # <-- ADD: use train set from seen classes too

        Xte_seen, yte_seen = Xte[mask_te], yte[mask_te]
        Xtr_seen, ytr_seen = Xtr[mask_tr], ytr[mask_tr]  # <-- ADD

        # Build features via current ReduNet for both train & test (seen classes only)
        Ztr = block.preprocess(Xtr_seen)                 # <-- ADD
        Ztr = run_inference_through_all_layers(block, Ztr)  # <-- ADD

        Zte = block.preprocess(Xte_seen)
        Zte = run_inference_through_all_layers(block, Zte)

        print('Accuracy after this task (seen classes):')
        # Existing metrics
        _, acc_svm = evaluate.svm(Ztr, ytr_seen, Zte, yte_seen)     # (train on Ztr, test on Zte)
        acc_knn = evaluate.knn(Ztr, ytr_seen, Zte, yte_seen, k=5)
        acc_svd = evaluate.nearsub(Ztr, ytr_seen, Zte, yte_seen, n_comp=1)

        # NEW: Logistic Softmax (multinomial logistic regression)
        acc_log_tr, acc_log_te = evaluate.logistic_softmax(Ztr, ytr_seen, Zte, yte_seen)

        acc = {
            "svm": float(acc_svm),
            "knn": float(acc_knn),
            "nearsub-svd": float(acc_svd),
            "logistic-softmax_train": float(acc_log_tr),  # optional: store train acc too
            "logistic-softmax_test": float(acc_log_te)
        }
        utils.save_params(model_dir, acc, name=f"acc_seen_after_task{task_id+1}.json")


    print("\nDone incremental training.")
    return arch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--save_dir", type=str, default="./saved_models/")
    ap.add_argument("--tail", type=str, default="mnist_inc")
    args = ap.parse_args()

    model_dir = os.path.join(args.save_dir, "mnist_inc",
                             f"layers{args.layers}_eps{args.eps}_eta{args.eta}_{args.tail}")
    os.makedirs(model_dir, exist_ok=True)
    utils.save_params(model_dir, vars(args))

    incremental_fit_mnist(model_dir, layers=args.layers, eta=args.eta, eps=args.eps)