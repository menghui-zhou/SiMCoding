"""SiM-ReduNet core extracted from Incremental_ImageNette_Resize28*28*3.py."""
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

@dataclass
class LayerStats:
    S_total: torch.Tensor
    m_total: int
    S_j: torch.Tensor
    m_j: torch.Tensor

class Vector:
    def __init__(self, layers, eta, eps, lmbda=500, device=None):
        self.layers = layers
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def load_arch(self, arch, block_id):
        self.arch, self.block_id = arch, block_id
        self.num_classes = arch.num_classes

    def _w_dir(self): return os.path.join(self.arch.model_dir, "weights")
    def _w_path(self, layer): return os.path.join(self._w_dir(), f"{self.block_id}_{layer}.pt")
    def _gam_path(self, layer): return os.path.join(self._w_dir(), f"{self.block_id}_{layer}_gam.pt")
    def save_weights(self, layer):
        os.makedirs(self._w_dir(), exist_ok=True)
        torch.save({"E": self.E.detach().cpu(), "Cs": self.Cs.detach().cpu()}, self._w_path(layer))
    def load_weights(self, layer):
        w = torch.load(self._w_path(layer), map_location=self.device)
        self.E = w["E"].to(self.device)
        self.Cs = w["Cs"].to(self.device)
    def save_gam(self, layer):
        os.makedirs(self._w_dir(), exist_ok=True)
        torch.save({"gam": self.gam.detach().cpu()}, self._gam_path(layer))
    def load_gam(self, layer):
        g = torch.load(self._gam_path(layer), map_location=self.device)
        self.gam = g["gam"].to(self.device)

    def __call__(self, Z, y=None):
        Z = self.preprocess(Z)
        if y is not None:
            y = self._as_labels(y)
        for layer in range(self.layers):
            Z, y_hat = self.forward(layer, Z, y)
            self.arch.update_loss(layer, *self.compute_loss(Z, (y if y is not None else y_hat)))
        return self.postprocess(Z)

    def forward(self, layer, Z, y=None):
        Z = self._as_tensor(Z)
        if y is not None:
            self.init(Z, y)
            self.save_weights(layer)
            self.save_gam(layer)
        else:
            self.load_weights(layer)
            self.load_gam(layer)
        expd = Z @ self.E.T
        comp = torch.stack([Z @ C.T for C in self.Cs], dim=0)
        clus, y_pred = self.nonlinear(comp, self.gam)
        Z = row_normalize(Z + self.eta * (expd - clus))
        return Z, y_pred

    def forward_incremental(self, layer, Z_new, y_new):
        Z_new = self._as_tensor(Z_new)
        y_new = self._as_labels(y_new)
        d = Z_new.shape[1]

        stats = getattr(self, "stats", {})
        st = stats.get(layer, None)
        if st is None:
            st = LayerStats(
                S_total=torch.zeros((d,d), dtype=torch.float32, device=self.device),
                m_total=0,
                S_j=torch.zeros((self.num_classes,d,d), dtype=torch.float32, device=self.device),
                m_j=torch.zeros((self.num_classes,), dtype=torch.int64, device=self.device)
            )

        st.S_total += Z_new.T @ Z_new
        st.m_total += int(Z_new.shape[0])
        for j in range(self.num_classes):
            idx = (y_new == j)
            if idx.any():
                Zj = Z_new[idx, :]
                st.S_j[j] += Zj.T @ Zj
                st.m_j[j] += int(idx.sum())

        self.compute_from_stats(st)

        stats[layer] = st
        self.stats = stats
        self.save_weights(layer)
        self.save_gam(layer)

        expd = Z_new @ self.E.T
        comp = torch.stack([Z_new @ C.T for C in self.Cs], dim=0)
        clus, y_pred = self.nonlinear(comp, self.gam)
        Z_out = row_normalize(Z_new + self.eta * (expd - clus))

        self.arch.update_loss(layer, *self.compute_loss(Z_out, y_new))
        return Z_out, y_pred

    def init(self, Z, y):
        self.compute_gam(y)
        self.compute_E(Z)
        self.compute_Cs(Z, y)

    def compute_gam(self, y):
        m_j = torch.stack([(y == j).sum() for j in range(self.num_classes)], dim=0).to(torch.float32)
        m = max(1, int(y.numel()))
        self.gam = (m_j / m).to(self.device)

    def compute_E(self, X):
        m, d = X.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        c = d / (m * self.eps)
        self.E = c * torch.linalg.inv(I + c * (X.T @ X))

    def compute_Cs(self, X, y):
        m, d = X.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        Cs = []
        for j in range(self.num_classes):
            idx = (y == j)
            mj = int(idx.sum())
            if mj == 0:
                Cs.append(torch.zeros((d, d), device=self.device, dtype=torch.float32))
            else:
                Zj = X[idx, :]
                cj = d / (mj * self.eps)
                Cs.append(cj * torch.linalg.inv(I + cj * (Zj.T @ Zj)))
        self.Cs = torch.stack(Cs, dim=0)

    def compute_from_stats(self, st: LayerStats):
        m_total = max(1, st.m_total)
        d = st.S_total.shape[0]
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        self.gam = (st.m_j.to(torch.float32) / m_total).to(self.device)
        c = d / (m_total * self.eps)
        self.E = c * torch.linalg.inv(I + c * st.S_total)
        Cs = []
        for j in range(self.num_classes):
            mj = int(st.m_j[j].item())
            if mj <= 0:
                Cs.append(torch.zeros((d, d), device=self.device, dtype=torch.float32))
            else:
                cj = d / (mj * self.eps)
                Cs.append(cj * torch.linalg.inv(I + cj * st.S_j[j]))
        self.Cs = torch.stack(Cs, dim=0)

    def compute_loss(self, Z, y):
        m, d = Z.shape
        I = torch.eye(d, device=self.device, dtype=torch.float32)
        c = d / (m * self.eps)
        _, logdet = torch.linalg.slogdet(I + c * (Z.T @ Z))
        loss_expd = logdet / 2.0
        loss_comp = 0.0
        for j in range(self.num_classes):
            idx = (y == j)
            mj = int(idx.sum())
            if mj == 0:
                continue
            Zj = Z[idx, :]
            cj = d / (mj * self.eps)
            _, logdet_j = torch.linalg.slogdet(I + cj * (Zj.T @ Zj))
            loss_comp = loss_comp + self.gam[j] * (logdet_j / 2.0)
        return float((loss_expd - loss_comp).item()), float(loss_expd.item()), float(loss_comp.item() if hasattr(loss_comp, "item") else loss_comp)

    def nonlinear(self, Bz, gam):
        norm = torch.linalg.vector_norm(Bz, dim=2).clamp_min(1e-8)
        pred = torch.softmax(-self.lmbda * norm, dim=0)
        y = torch.argmax(pred, dim=0)
        gam_b = gam.view(-1, 1, 1)
        out = torch.sum(gam_b * Bz * pred.view(pred.shape[0], pred.shape[1], 1), dim=0)
        return out, y

    def preprocess(self, X):
        if torch.is_tensor(X): X = X.to(self.device, dtype=torch.float32)
        else:                  X = torch.tensor(X, dtype=torch.float32, device=self.device)
        m = X.shape[0]
        X = X.view(m, -1)
        return row_normalize(X)

    def postprocess(self, X): return row_normalize(self._as_tensor(X))
    def _as_tensor(self, X):
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)
    def _as_labels(self, y):
        return torch.as_tensor(y, dtype=torch.int64, device=self.device)

def run_inference_through_all_layers(block: Vector, Z_np: np.ndarray):
    Z = block.preprocess(Z_np)
    for layer in range(block.layers):
        block.load_weights(layer)
        block.load_gam(layer)
        expd = Z @ block.E.T
        comp = torch.stack([Z @ C.T for C in block.Cs], dim=0)
        clus, _ = block.nonlinear(comp, block.gam)
        Z = row_normalize(Z + block.eta * (expd - clus))
    return Z
