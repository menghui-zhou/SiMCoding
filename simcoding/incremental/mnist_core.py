"""SiM-ReduNet core extracted from Modified_Incre_MNIST.py."""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def row_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class Architecture:
    def __init__(self, blocks, model_dir, num_classes, batch_size=4096):
        self.blocks = blocks
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        for b, block in enumerate(self.blocks):
            block.load_arch(self, b)
        self.init_loss()

    def __getitem__(self, i): return self.blocks[i]

    def init_loss(self):
        self.loss_dict = {"loss_total": [], "loss_expd": [], "loss_comp": []}

    def update_loss(self, layer, loss_total, loss_expd, loss_comp):
        self.loss_dict["loss_total"].append(loss_total)
        self.loss_dict["loss_expd"].append(loss_expd)
        self.loss_dict["loss_comp"].append(loss_comp)
        print(f"layer: {layer:03d} | loss_total: {loss_total:.6f} | loss_expd: {loss_expd:.6f} | loss_comp: {loss_comp:.6f}")

@dataclass
class LayerState:
    Sigma_j: torch.Tensor  # (K, d, d)  where Sigma_j[j] = Z_j^T Z_j  (PyTorch uses Z as (m,d))
    m_j: torch.Tensor

class Vector:
    def __init__(self, layers, eta, eps, lmbda=500, device=None):
        """
        eps here is eps^2 in the paper (i.e., alpha = d/(m*eps))
        """
        self.layers = layers
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # in-memory cache: layer -> LayerState
        self.layer_states = {}

    # ---- arch context ----
    def load_arch(self, arch, block_id):
        self.arch, self.block_id = arch, block_id
        self.num_classes = arch.num_classes

    # ---- paths ----
    def _w_dir(self): return os.path.join(self.arch.model_dir, "weights")
    def _w_path(self, layer): return os.path.join(self._w_dir(), f"{self.block_id}_{layer}.pt")

    # ---- save/load state ----
    def save_layer(self, layer, E, Cs, gam, state: LayerState):
        ensure_dir(self._w_dir())
        torch.save({
            "E": E.detach().cpu(),
            "Cs": Cs.detach().cpu(),
            "gam": gam.detach().cpu(),
            "Sigma_j": state.Sigma_j.detach().cpu(),
            "m_j": state.m_j.detach().cpu(),
        }, self._w_path(layer))

    def load_layer(self, layer):
        ckpt = torch.load(self._w_path(layer), map_location=self.device)
        E = ckpt["E"].to(self.device)
        Cs = ckpt["Cs"].to(self.device)
        gam = ckpt["gam"].to(self.device)
        Sigma_j = ckpt["Sigma_j"].to(self.device)
        m_j = ckpt["m_j"].to(self.device)
        return E, Cs, gam, LayerState(Sigma_j=Sigma_j, m_j=m_j)

    def has_layer(self, layer):
        return os.path.exists(self._w_path(layer))

    # ---- pre/post & type helpers ----
    def preprocess(self, X):
        if torch.is_tensor(X):
            X = X.to(self.device, dtype=torch.float32)
        else:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        m = X.shape[0]
        X = X.view(m, -1)        # flatten (MNIST: 784)
        return row_normalize(X)

    def postprocess(self, X): return row_normalize(self._as_tensor(X))

    def _as_tensor(self, X):
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)

    def _as_labels(self, y):
        return torch.as_tensor(y, dtype=torch.int64, device=self.device)

    # ---- loss ----
    def compute_loss(self, Z, y):
        # Z: (m,d), y: (m,)
        m, d = Z.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)

        c = d / (max(1, m) * self.eps)
        _, logdet = torch.linalg.slogdet(I + c * (Z.T @ Z))
        loss_expd = logdet / 2.0

        loss_comp = 0.0
        # gamma from batch label counts (consistent with your previous training-loss logging)
        m_j = torch.stack([(y == j).sum() for j in range(self.num_classes)], dim=0).to(torch.float32)
        gam = (m_j / max(1, int(y.numel()))).to(self.device)

        for j in range(self.num_classes):
            idx = (y == j)
            mj = int(idx.sum())
            if mj == 0:
                continue
            Zj = Z[idx, :]
            cj = d / (mj * self.eps)
            _, logdet_j = torch.linalg.slogdet(I + cj * (Zj.T @ Zj))
            loss_comp = loss_comp + gam[j] * (logdet_j / 2.0)

        loss_total = loss_expd - loss_comp
        return float(loss_total.item()), float(loss_expd.item()), float(loss_comp.item())

    # ---- inference nonlinearity (soft membership) ----
    def nonlinear(self, Bz, gam):
        """
        Bz: (k, m, d) where Bz[j] = Z @ Cj^T
        gam: (k,)
        """
        norm = torch.linalg.vector_norm(Bz, dim=2).clamp_min(1e-8)  # (k,m)
        pred = torch.softmax(-self.lmbda * norm, dim=0)             # (k,m)
        y = torch.argmax(pred, dim=0)                               # (m,)
        gam_b = gam.view(-1, 1, 1)                                  # (k,1,1)
        out = torch.sum(gam_b * Bz * pred.view(pred.shape[0], pred.shape[1], 1), dim=0)  # (m,d)
        return out, y

    # ---- build E, Cs from covariance (paper style) ----
    def compute_E_from_Sigma(self, Sigma, m_total):
        # Sigma = Z^T Z  (d,d)
        d = Sigma.shape[0]
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        m_total = max(1, int(m_total))
        alpha = d / (m_total * self.eps)
        E = alpha * torch.linalg.inv(I + alpha * Sigma)
        return E

    def compute_Cs_from_Sigma_j(self, Sigma_j, m_j):
        # Sigma_j: (K,d,d), m_j: (K,)
        K, d, _ = Sigma_j.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        Cs = torch.zeros((K, d, d), device=self.device, dtype=torch.float32)
        for j in range(K):
            mj = int(m_j[j].item())
            if mj <= 0:
                continue
            alpha_j = d / (mj * self.eps)
            Cs[j] = alpha_j * torch.linalg.inv(I + alpha_j * Sigma_j[j])
        return Cs

    # ---- hard-label training layer update (linear per class) ----
    def train_layer_update(self, Z, y, E, Cs, gam):
        """
        Z: (m,d), y: (m,)
        Update uses hard membership: clus_i = gam[y_i] * (Z_i @ C_{y_i}^T)
        """
        expd = Z @ E.T  # (m,d)

        # gather Cs[y]: (m,d,d)
        # Same per-class update without allocating an (m,d,d) tensor.
        comp_y = torch.zeros_like(Z)
        for j in torch.unique(y):
            mask = y == j
            comp_y[mask] = Z[mask] @ Cs[int(j)].T
        clus = (gam[y].unsqueeze(1)) * comp_y

        Z_out = row_normalize(Z + self.eta * (expd - clus))
        return Z_out

    # ---------------- Paper-faithful incremental fit on ONE task ----------------
    def fit_task_incremental(self, X_task_np: np.ndarray, y_task_np: np.ndarray):
        """
        Performs Algorithm-1 style incremental update over all layers,
        updating stored per-layer covariances and saving E, Cs, gam per layer.

        Key: old-class covariances are propagated analytically, no old samples needed.
        """
        Z = self.preprocess(X_task_np)              # (m,d)
        y = self._as_labels(y_task_np)              # (m,)
        K = self.num_classes
        m_total = int(y.numel())
        d = Z.shape[1]

        # counts for this task (these become the m_j used for all layers in this normalization scheme)
        m_j_task = torch.zeros((K,), dtype=torch.int64, device=self.device)
        for j in range(K):
            m_j_task[j] = (y == j).sum()

        # layer 0 initial Sigma_j for this task
        Sigma_j_new = torch.zeros((K, d, d), dtype=torch.float32, device=self.device)
        for j in range(K):
            idx = (y == j)
            if idx.any():
                Zj = Z[idx, :]               # (mj,d)
                Sigma_j_new[j] = Zj.T @ Zj   # (d,d)

        # We will update all layers sequentially.
        for layer in range(self.layers):
            # 1) Load old stored state at this layer (if exists). Otherwise zeros.
            if self.has_layer(layer):
                _, _, _, old_state = self.load_layer(layer)
                Sigma_j_old = old_state.Sigma_j
                m_j_old = old_state.m_j
                # sanity: dimensions
                if Sigma_j_old.shape[-1] != d:
                    raise RuntimeError(f"Dim mismatch at layer {layer}: stored d={Sigma_j_old.shape[-1]} vs current d={d}")
            else:
                Sigma_j_old = torch.zeros((K, d, d), dtype=torch.float32, device=self.device)
                m_j_old = torch.zeros((K,), dtype=torch.int64, device=self.device)

            # 2) Merge: joint-training equivalent Sigma_j and counts at this layer (before propagation)
            Sigma_j_joint = Sigma_j_old + Sigma_j_new
            m_j_joint = m_j_old + m_j_task

            # 3) Compute gam, E, Cs from joint covariances at this layer
            m_total_joint = int(m_j_joint.sum().item())
            gam = (m_j_joint.to(torch.float32) / max(1, m_total_joint)).to(self.device)  # (K,)
            Sigma_total = Sigma_j_joint.sum(dim=0)                                       # (d,d)

            E = self.compute_E_from_Sigma(Sigma_total, m_total_joint)
            Cs = self.compute_Cs_from_Sigma_j(Sigma_j_joint, m_j_joint)

            # 4) Save current-layer parameters and state (this is the trained network at this layer)
            state_here = LayerState(Sigma_j=Sigma_j_joint, m_j=m_j_joint)
            self.save_layer(layer, E, Cs, gam, state_here)

            # 5) Forward update real task features (hard labels) to get next-layer features Z_{l+1,new}
            Z_next = self.train_layer_update(Z, y, E, Cs, gam)

            # Log loss on the updated features (optional but consistent with your previous logging)
            self.arch.update_loss(layer, *self.compute_loss(Z_next, y))

            # 6) Propagate NEW-task covariances to next layer using paper recursion:
            #    For each class j: T = L Sigma L^T, then trace-normalize.
            Sigma_j_new_next = torch.zeros_like(Sigma_j_new)
            for j in range(K):
                mj = int(m_j_task[j].item())  # counts of new-task samples in class j
                if mj <= 0:
                    continue
                Lj = torch.eye(d, device=self.device, dtype=torch.float32) + self.eta * E - self.eta * gam[j] * Cs[j]
                T = Lj @ Sigma_j_new[j] @ Lj.T
                tr = torch.trace(T).clamp_min(1e-12)
                Sigma_j_new_next[j] = (mj / tr) * T

            # 7) Move to next layer: Z and Sigma_j_new update
            Z = Z_next
            Sigma_j_new = Sigma_j_new_next

        return

    # ---------------- Inference through all layers ----------------
    def inference_all_layers(self, X_np: np.ndarray):
        """
        Soft-membership inference (Eq. 8 style) through saved layers.
        """
        Z = self.preprocess(X_np)  # (m,d)
        for layer in range(self.layers):
            E, Cs, gam, _ = self.load_layer(layer)
            expd = Z @ E.T  # (m,d)
            # comp: (k,m,d) where comp[j] = Z @ Cj^T
            comp = torch.stack([Z @ C.T for C in Cs], dim=0)  # (K, m, d)
            clus, _ = self.nonlinear(comp, gam)
            Z = row_normalize(Z + self.eta * (expd - clus))
        return Z
