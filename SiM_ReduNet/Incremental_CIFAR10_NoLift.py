# incremental_cifar10_gpu_nolift.py
# GPU ReduNet (Vector only, no lift/pooling) with incremental learning on CIFAR-10.

import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision import datasets, transforms

# ---------- small utils ----------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def row_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)

# ---------- Architecture ----------
class Architecture:
    def __init__(self, blocks, model_dir, num_classes, batch_size=4096):
        self.blocks = blocks
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        for b, block in enumerate(self.blocks):
            block.load_arch(self, b)
        self.init_loss()

    def __call__(self, Z, y=None):
        for b, block in enumerate(self.blocks):
            self.init_loss()
            Z = block.preprocess(Z)
            Z = block(Z, y)
            Z = block.postprocess(Z)
        return Z

    def __getitem__(self, i): return self.blocks[i]
    def init_loss(self):
        self.loss_dict = {"loss_total": [], "loss_expd": [], "loss_comp": []}
    def update_loss(self, layer, loss_total, loss_expd, loss_comp):
        self.loss_dict["loss_total"].append(loss_total)
        self.loss_dict["loss_expd"].append(loss_expd)
        self.loss_dict["loss_comp"].append(loss_comp)
        print(f"layer: {layer} | loss_total: {loss_total:.6f} | loss_expd: {loss_expd:.6f} | loss_comp: {loss_comp:.6f}")

# ---------- Vector (GPU, incremental) ----------
@dataclass
class LayerStats:
    S_total: torch.Tensor  # (d,d)
    m_total: int
    S_j: torch.Tensor      # (k,d,d)
    m_j: torch.Tensor      # (k,)

class Vector:
    def __init__(self, layers, eta, eps, lmbda=500, device=None):
        self.layers = layers
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.E_list, self.Cs_list, self.gam_list, self.stats_list = [], [], [], []

    def load_arch(self, arch, block_id):
        self.arch, self.block_id = arch, block_id
        self.num_classes = arch.num_classes

    # ----- helpers -----
    def _as_tensor(self, X):
        return X if torch.is_tensor(X) and X.device == self.device else torch.tensor(X, dtype=torch.float32, device=self.device)
    def _as_labels(self, y):
        return y if torch.is_tensor(y) and y.device == self.device else torch.tensor(y, dtype=torch.int64, device=self.device)
    def _ensure_len(self, L):
        while len(self.gam_list) < L: self.gam_list.append(None)
        while len(self.E_list)   < L: self.E_list.append(None)
        while len(self.Cs_list)  < L: self.Cs_list.append(None)

    # ----- forward API -----
    def __call__(self, Z, y=None):
        Z = self.preprocess(Z)
        if y is not None: y = self._as_labels(y)
        for layer in range(self.layers):
            Z, y_hat = self.forward(layer, Z, y)
            self.arch.update_loss(layer, *self.compute_loss(Z, (y if y is not None else y_hat)))
        return self.postprocess(Z)

    def forward(self, layer, Z, y=None):
        if y is not None:
            self.init(Z, y, layer)
        E, Cs, gam = self.E_list[layer], self.Cs_list[layer], self.gam_list[layer]
        expd = Z @ E.T
        comp = torch.stack([Z @ C.T for C in Cs], dim=0)  # (k,m,d)
        clus, y_pred = self.nonlinear(comp, gam)
        Z = row_normalize(Z + self.eta * (expd - clus))
        return Z, y_pred

    # ----- incremental per-layer -----
    def forward_incremental(self, layer, Z_new, y_new):
        Z_new = self._as_tensor(Z_new); y_new = self._as_labels(y_new)
        self._ensure_len(layer+1)
        d = Z_new.shape[1]
        if layer >= len(self.stats_list): self.stats_list.extend([None]*(layer-len(self.stats_list)+1))
        stats = self.stats_list[layer]
        if stats is None:
            stats = LayerStats(
                S_total=torch.zeros((d,d), dtype=torch.float32, device=self.device),
                m_total=0,
                S_j=torch.zeros((self.num_classes,d,d), dtype=torch.float32, device=self.device),
                m_j=torch.zeros((self.num_classes,), dtype=torch.int64, device=self.device)
            )
        # update stats
        stats.S_total += Z_new.T @ Z_new
        stats.m_total += int(Z_new.shape[0])
        for j in range(self.num_classes):
            idx = (y_new == j)
            if idx.any():
                Zj = Z_new[idx, :]
                stats.S_j[j] += Zj.T @ Zj
                stats.m_j[j] += int(idx.sum())
        # recompute weights
        self.compute_from_stats(stats, layer)
        self.stats_list[layer] = stats

        # forward NEW batch
        E, Cs, gam = self.E_list[layer], self.Cs_list[layer], self.gam_list[layer]
        expd = Z_new @ E.T
        comp = torch.stack([Z_new @ C.T for C in Cs], dim=0)
        clus, y_pred = self.nonlinear(comp, gam)
        Z_out = row_normalize(Z_new + self.eta * (expd - clus))
        self.arch.update_loss(layer, *self.compute_loss(Z_out, y_new))
        return Z_out, y_pred

    # ----- build from data/stats -----
    def init(self, Z, y, layer):
        m = max(1, int(y.numel()))
        gam = torch.stack([(y == j).sum() for j in range(self.num_classes)], dim=0).to(torch.float32) / m
        E = self._compute_E_from_Z(Z)
        Cs = self._compute_Cs_from_Zy(Z, y)
        self._ensure_len(layer+1)
        self.gam_list[layer] = gam
        self.E_list[layer]   = E
        self.Cs_list[layer]  = Cs

    def compute_from_stats(self, stats: LayerStats, layer: int):
        m_total = max(1, stats.m_total); d = stats.S_total.shape[0]
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        gam = (stats.m_j.to(torch.float32) / m_total).to(self.device)
        c = d / (m_total * self.eps)
        E = c * torch.linalg.inv(I + c * stats.S_total)
        Cs = []
        for j in range(self.num_classes):
            mj = int(stats.m_j[j].item())
            if mj <= 0:
                Cs.append(torch.zeros((d, d), device=self.device, dtype=torch.float32))
            else:
                cj = d / (mj * self.eps)
                Cs.append(cj * torch.linalg.inv(I + cj * stats.S_j[j]))
        Cs = torch.stack(Cs, dim=0)
        self._ensure_len(layer+1)
        self.gam_list[layer] = gam; self.E_list[layer] = E; self.Cs_list[layer] = Cs

    def _compute_E_from_Z(self, Z):
        m, d = Z.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        c = d / (m * self.eps)
        return c * torch.linalg.inv(I + c * (Z.T @ Z))

    def _compute_Cs_from_Zy(self, Z, y):
        m, d = Z.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        Cs = []
        for j in range(self.num_classes):
            idx = (y == j)
            mj = int(idx.sum())
            if mj == 0:
                Cs.append(torch.zeros((d, d), device=self.device, dtype=torch.float32))
            else:
                Zj = Z[idx, :]
                cj = d / (mj * self.eps)
                Cs.append(cj * torch.linalg.inv(I + cj * (Zj.T @ Zj)))
        return torch.stack(Cs, dim=0)

    # ----- loss & nonlinearity -----
    def compute_loss(self, Z, y):
        m, d = Z.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        c = d / (m * self.eps)
        _, logdet = torch.linalg.slogdet(I + c * (Z.T @ Z))
        loss_expd = logdet / 2.0
        loss_comp = 0.0
        # use latest gam (prior) if available
        gam = self.gam_list[-1] if self.gam_list and self.gam_list[-1] is not None else torch.zeros(self.num_classes, device=self.device)
        for j in range(self.num_classes):
            idx = (y == j)
            mj = int(idx.sum())
            if mj == 0: continue
            Zj = Z[idx, :]
            cj = d / (mj * self.eps)
            _, logdet_j = torch.linalg.slogdet(I + cj * (Zj.T @ Zj))
            loss_comp = loss_comp + gam[j] * (logdet_j / 2.0)
        return float((loss_expd - loss_comp).item()), float(loss_expd.item()), float(loss_comp.item() if hasattr(loss_comp, "item") else loss_comp)

    def nonlinear(self, Bz, gam):
        # Bz: (k,m,d); gam: (k,)
        norm = torch.linalg.vector_norm(Bz, dim=2).clamp_min(1e-8)  # (k,m)
        pred = torch.softmax(-self.lmbda * norm, dim=0)             # (k,m)
        y = torch.argmax(pred, dim=0)                               # (m,)
        gam_b = gam.view(gam.shape[0], 1, 1)                        # (k,1,1)
        out = torch.sum(gam_b * Bz * pred.view(pred.shape[0], pred.shape[1], 1), dim=0)  # (m,d)
        return out, y

    # ----- pre/post -----
    def preprocess(self, X):
        # X can be (N,3,32,32) np or torch → flatten to (N,3072) and normalize
        if torch.is_tensor(X): X = X.to(self.device, dtype=torch.float32)
        else:                  X = torch.tensor(X, dtype=torch.float32, device=self.device)
        m = X.shape[0]
        X = X.view(m, -1)
        return row_normalize(X)

    def postprocess(self, X): return row_normalize(X)

# ---------- Evaluators (CPU / sklearn) ----------
def eval_svm(Ztr, ytr, Zte, yte):
    from sklearn.svm import LinearSVC
    clf = LinearSVC(random_state=10)
    clf.fit(Ztr, ytr)
    return clf.score(Ztr, ytr), clf.score(Zte, yte)

def eval_knn_cosine(Ztr, ytr, Zte, yte, k=5):
    # cosine sim = dot if normalized
    sim = Ztr @ Zte.T                          # (ntr, nte)
    # top-k indices per column
    topk_idx = np.argpartition(sim, -k, axis=0)[-k:]  # (k, nte)
    preds = ytr[topk_idx]                      # (k, nte)
    # majority vote (pure numpy)
    # Count votes per class for each column
    classes = np.unique(ytr)
    vote = np.zeros((classes.size, preds.shape[1]), dtype=int)
    for i, c in enumerate(classes):
        vote[i] = (preds == c).sum(axis=0)
    yhat = classes[vote.argmax(axis=0)]
    acc = (yhat == yte).mean()
    return float(acc)

def eval_nearsub_svd(Ztr, ytr, Zte, yte, n_comp=1):
    from sklearn.decomposition import TruncatedSVD
    classes = np.unique(yte)
    fd = Ztr.shape[1]
    n_comp = min(max(1, n_comp), fd-1)
    scores = []
    eye = np.eye(fd, dtype=Ztr.dtype)
    for c in classes:
        Zc = Ztr[ytr == c]
        svd = TruncatedSVD(n_components=n_comp).fit(Zc)
        sub = svd.components_.T
        resid = (eye - sub @ sub.T) @ Zte.T   # (d, nte)
        scores.append(np.linalg.norm(resid, ord=2, axis=0))
    pred = classes[np.argmin(np.vstack(scores), axis=0)]
    return float((pred == yte).mean())

def eval_logistic_softmax(Ztr, ytr, Zte, yte, C=1.0, max_iter=1000, tol=1e-4):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty="l2", C=C, tol=tol, max_iter=max_iter,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(Ztr, ytr)
    return float(clf.score(Ztr, ytr)), float(clf.score(Zte, yte))

# ---------- Data ----------
def load_cifar10_numpy(root="./data", train=True):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    X = ds.data.astype(np.float32) / 255.0            # (N,32,32,3)
    y = np.array(ds.targets, dtype=np.int64)
    X = np.transpose(X, (0, 3, 1, 2))                 # (N,3,32,32)
    return X, y

def split_tasks_by_class(X, y, pairs=((0,1),(2,3),(4,5),(6,7),(8,9))):
    tasks = []
    for a, b in pairs:
        idx = np.where((y == a) | (y == b))[0]
        tasks.append((X[idx], y[idx]))
    return tasks

# ---------- Inference (Vector only) ----------
def run_inference(vec: Vector, X_np: np.ndarray):
    Z = vec.preprocess(X_np)  # torch on device (N,3072)
    for layer in range(vec.layers):
        E, Cs, gam = vec.E_list[layer], vec.Cs_list[layer], vec.gam_list[layer]
        expd = Z @ E.T
        comp = torch.stack([Z @ C.T for C in Cs], dim=0)
        clus, _ = vec.nonlinear(comp, gam)
        Z = row_normalize(Z + vec.eta * (expd - clus))
    return Z

# ---------- Training loop ----------
def incremental_fit_cifar10_nolift(model_dir, layers=1, eta=0.5, eps=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Xtr, ytr = load_cifar10_numpy(train=True)
    Xte, yte = load_cifar10_numpy(train=False)
    num_classes = 10

    vec = Vector(layers=layers, eta=eta, eps=eps, device=device)
    arch = Architecture([vec], model_dir=model_dir, num_classes=num_classes)

    train_tasks = split_tasks_by_class(Xtr, ytr)

    for task_id, (Xt, yt) in enumerate(train_tasks):
        print(f"\n=== Task {task_id+1}/5: classes {np.unique(yt)} ===")
        arch.init_loss()

        Z = vec.preprocess(Xt)  # torch (N,3072)
        y = torch.tensor(yt, dtype=torch.int64, device=device)

        for layer in range(vec.layers):
            Z, _ = vec.forward_incremental(layer, Z, y)

        # Evaluate on seen classes so far
        seen = np.unique(np.concatenate([tt[1] for tt in train_tasks[:task_id+1]]))
        mask_te = np.isin(yte, seen); mask_tr = np.isin(ytr, seen)
        Xte_seen, yte_seen = Xte[mask_te], yte[mask_te]
        Xtr_seen, ytr_seen = Xtr[mask_tr], ytr[mask_tr]

        Ztr = run_inference(vec, Xtr_seen).detach().cpu().numpy()
        Zte = run_inference(vec, Xte_seen).detach().cpu().numpy()

        print("Accuracy after this task (seen classes):")
        acc_svm_tr, acc_svm = eval_svm(Ztr, ytr_seen, Zte, yte_seen)
        print(f"SVM: {acc_svm:.3f}")
        acc_knn = eval_knn_cosine(Ztr, ytr_seen, Zte, yte_seen, k=5)
        print(f"kNN: {acc_knn:.3f}")
        acc_svd = eval_nearsub_svd(Ztr, ytr_seen, Zte, yte_seen, n_comp=1)
        print(f"SVD: {acc_svd:.3f}")
        acc_log_tr, acc_log = eval_logistic_softmax(Ztr, ytr_seen, Zte, yte_seen)
        print(f"Logistic-Softmax: {acc_log:.3f}")

        report = {
            "task": int(task_id+1), "seen_classes": list(map(int, seen)),
            "svm": float(acc_svm), "knn": float(acc_knn), "svd": float(acc_svd),
            "logistic_softmax": float(acc_log)
        }
        save_json(report, os.path.join(model_dir, f"acc_after_task{task_id+1}.json"))

    print("\nDone incremental CIFAR-10 training.")

# ---------- Main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8, help="Vector layers (memory ~400MB per layer at d=3072)")
    ap.add_argument("--eta", type=float, default=5)
    ap.add_argument("--eps", type=float, default=228.64)
    ap.add_argument("--save_dir", type=str, default="./saved_models/")
    ap.add_argument("--tail", type=str, default="cifar10_inc_gpu_nolift")
    args = ap.parse_args()

    model_dir = os.path.join(args.save_dir, "cifar10_inc_nolift",
                             f"layers{args.layers}_eps{args.eps}_eta{args.eta}_{args.tail}")
    os.makedirs(model_dir, exist_ok=True)
    save_json(vars(args), os.path.join(model_dir, "args.json"))

    incremental_fit_cifar10_nolift(model_dir, layers=args.layers, eta=args.eta, eps=args.eps)