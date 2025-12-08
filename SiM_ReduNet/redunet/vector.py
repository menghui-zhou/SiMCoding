import os
import numpy as np
import functionals as F
from scipy.special import softmax

# --- Add to the top of your file ---
from dataclasses import dataclass
import json


@dataclass
class LayerStats:
    # Sufficient statistics for one layer
    # S_total = Z^T Z   (d x d)
    # m_total = total #samples
    # S_j[j] = Z_j^T Z_j   (d x d), m_j[j] = #samples in class j
    S_total: np.ndarray
    m_total: int
    S_j: np.ndarray  # shape (k, d, d)
    m_j: np.ndarray  # shape (k,)


class Vector:
    """
    Forward-constructed ReduNet block for vector (non-convolutional) features.

    Each forward layer does:
      1) Build expansion/compression filters (E, {C_j}) from current features Z (if labels available).
      2) Expansion: E Z
      3) Compression / clustering: σ([C_j Z]) via soft assignment over classes
      4) Residual update: Z <- Z + η (E Z - σ([C_j Z]))
      5) Projection to sphere: normalize rows of Z
    Loss tracked per layer: ΔR = R - Rc, with R and Rc as log-det coding rates.
    """

    def __init__(self, layers, eta, eps, lmbda=500):
        self.layers = layers  # number of ReduNet layers
        self.eta = eta  # step size for residual update
        self.eps = eps  # epsilon used in alpha = d / (m * eps)
        self.lmbda = lmbda  # temperature for softmax in nonlinear()

    def __call__(self, Z, y=None):
        """
        Run Z through 'layers' forward-constructed steps.
        If y is given for the first pass, each layer constructs and saves E, {C_j}, and γ.
        If y is None (inference), loads precomputed weights and γ for each layer.
        """
        for layer in range(self.layers):
            Z, y_approx = self.forward(layer, Z, y)
            # Track losses using predicted labels y_approx at this layer
            self.arch.update_loss(layer, *self.compute_loss(Z, y_approx))
        return Z

    def forward(self, layer, Z, y=None):
        """
        One ReduNet layer:
          - If labels available: initialize and save E, C, γ for this layer.
          - Else: load previously saved E, C, γ.
          - Compute expansion E Z and compression stack [C_j Z].
          - Nonlinear soft assignment and aggregation.
          - Residual update + normalization.
        Returns updated features and the soft-assignment argmax labels.
        """
        if y is not None:
            self.init(Z, y)
            self.save_weights(layer)
            self.save_gam(layer)
        else:
            self.load_weights(layer)
            self.load_gam(layer)

        # Expansion term: shape (m, d)
        expd = Z @ self.E.T

        # Compression terms for each class: stack shape (k, m, d)
        comp = np.stack([Z @ C.T for C in self.Cs])

        # Soft clustering / nonlinear aggregation
        clus, y_approx = self.nonlinear(comp)

        # Residual update towards expansion and away from compression
        Z = Z + self.eta * (expd - clus)

        # Project back to the unit sphere (row-wise)
        Z = F.normalize(Z)
        return Z, y_approx

    def load_arch(self, arch, block_id):
        """
        Attach training/eval context (for logging, saving paths, class count).
        """
        self.arch = arch
        self.block_id = block_id
        self.num_classes = self.arch.num_classes

    def init(self, Z, y):
        """
        Build quantities from current features:
          - γ (class priors)
          - E (expansion filter)
          - {C_j} (class-wise compression filters)
        """
        self.compute_gam(y)
        self.compute_E(Z)
        self.compute_Cs(Z, y)

    def compute_gam(self, y):
        """
        γ_j = m_j / m : class prior (fraction of samples in class j).
        """
        inspect1 = []  # start with an empty list
        for j in range(self.num_classes):  # loop over each class index j = 0, 1, ..., k-1
            mask = (y == j)  # boolean array: True where sample label == j
            inspect2 = mask.nonzero()
            inspect3 = inspect2[0]  # the first dimension of a tuple
            indices = mask.nonzero()[0]  # indices of samples in class j
            count = indices.size  # number of samples in class j
            inspect1.append(count)  # add this count to the list

        # after loop, m_j is a list with num_classes elements, each the count of samples per class
        m_j = [(y == j).nonzero()[0].size for j in range(self.num_classes)]
        self.gam = np.array(m_j) / y.size

    def compute_E(self, X):
        """
        E = α (I + α Z Z^T)^(-1), where:
          - X is (m, d), Z := X^T is (d, m)
          - α = d / (m * eps)
        """
        m, d = X.shape  # m is the number of samples, d is the dimension of features
        Z = X.T  # d * m
        I = np.eye(d)
        c = d / (m * self.eps)
        E = c * np.linalg.inv(I + c * (Z @ Z.T))
        self.E = E

    def compute_Cs(self, X, y):
        """
        For each class j:
          C_j = α_j (I + α_j Z_j Z_j^T)^(-1), where:
            - Z_j are columns of Z for samples in class j
            - α_j = d / (m_j * eps)
        """
        m, d = X.shape  # m is the number of samples and d is the dimesnion of features
        Z = X.T  # d * m
        I = np.eye(d)
        Cs = np.empty((self.num_classes, d, d))
        for j in range(self.num_classes):
            idx = (y == int(j))
            Z_j = Z[:, idx]  # (d, m_j)
            m_j = Z_j.shape[1]
            if m_j == 0:
                # Optional: provide a safe default (e.g., identity scaled to zero influence)
                Cs[j] = np.zeros((d, d))
                continue
            c_j = d / (m_j * self.eps)
            C = c_j * np.linalg.inv(I + c_j * (Z_j @ Z_j.T))
            Cs[j] = C
        self.Cs = Cs

    def compute_loss(self, Z, y):
        """
        Compute coding-rate objective components:
          R  = 0.5 * log det(I + α Z^T Z)
          Rc = sum_j γ_j * 0.5 * log det(I + α_j Z_j^T Z_j)
          ΔR = R - Rc
        Returns (ΔR, R, Rc).
        """
        m, d = Z.shape
        I = np.eye(d)

        # Overall rate R
        c = d / (m * self.eps)
        logdet = np.linalg.slogdet(I + c * (Z.T @ Z))[1]
        loss_expd = logdet / 2.0

        # Class-wise rate Rc
        loss_comp = 0.0
        for j in np.arange(self.num_classes):
            idx = (y == int(j))
            Z_j = Z[idx, :]
            m_j = Z_j.shape[0]
            if m_j == 0:
                continue
            c_j = d / (m_j * self.eps)
            logdet_j = np.linalg.slogdet(I + c_j * (Z_j.T @ Z_j))[1]
            loss_comp += self.gam[j] * (logdet_j / 2.0)

        return loss_expd - loss_comp, loss_expd, loss_comp

    def preprocess(self, X):
        """
        Flatten input samples and normalize to the unit sphere.
        """
        m = X.shape[0]
        X = X.reshape(m, -1)
        return F.normalize(X)

    def postprocess(self, X):
        """
        Normalize outputs to the unit sphere.
        """
        return F.normalize(X)

    def nonlinear(self, Bz):
        """
        Soft assignment and aggregation over class-specific projections.

        Args:
          Bz: array of shape (k, m, d), where each slice Bz[j] = C_j Z.

        Steps:
          - Compute per-class norms ‖C_j z_i‖ for all samples i.
          - Softmax over classes with temperature λ to get π̂_j(z_i).
          - Predict ŷ_i = argmax_j π̂_j(z_i).
          - Aggregate σ([C_j z]) = Σ_j γ_j * (C_j Z) * π̂_j(z), broadcasting over dims.
        """
        # Collapse any extra trailing dims (for generality) and compute norms per class/sample
        inspect1 = Bz.reshape(Bz.shape[0], Bz.shape[1], -1)  # reshape will return a new view
        norm = np.linalg.norm(Bz.reshape(Bz.shape[0], Bz.shape[1], -1), axis=2)
        # np.clip(x, a_min, a_max)
        norm = np.clip(norm, 1e-8, norm)  # numerical safety

        # Soft assignment: higher norm => lower probability (hence negative sign)
        pred = softmax(-self.lmbda * norm, axis=0)  # shape (k, m)
        y = np.argmax(pred, axis=0)  # predicted labels (m,)

        # Broadcast γ to match Bz dims
        inspect2 = Bz.shape
        inspect3 = len(inspect2)  # the length is simply the number of elements in the tuple
        inspect4 = np.arange(2, inspect3)
        axes = tuple(np.arange(2, len(Bz.shape)))  # dims beyond (k, m)
        # Expand the shape of an array.
        gam = np.expand_dims(self.gam, tuple(np.arange(1, len(Bz.shape))))  # shape (k, 1, [1...]),

        # Weighted sum over classes with soft assignments
        # Σ_j γ_j   (C_j z_i)   π̂_j(z_i)
        inspect5 = np.expand_dims(pred, axes)
        out = np.sum(gam * Bz * np.expand_dims(pred, axes), axis=0)  # shape (m, d)
        return out, y

    def save_weights(self, layer):
        """
        Save E and the stack of Cs for this layer to disk.
        Stored as a single .npy with E at index 0 and Cs at [1:].
        """
        weights = np.vstack([
            self.E[np.newaxis, :],  # (1, d, d), make the dimension same as Cs
            self.Cs  # (k, d, d)
        ])
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}.npy")
        np.save(save_path, weights)

    def load_weights(self, layer):
        """
        Load E and Cs for a given layer from disk.
        """
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}.npy")
        weights = np.load(save_path, allow_pickle=False)
        self.E = weights[0]
        self.Cs = weights[1:]
        return self.E, self.Cs

    def save_gam(self, layer):
        """
        Save γ (class priors) for this layer.
        """
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}_gam.npy")
        np.save(save_path, self.gam)

    def load_gam(self, layer):
        """
        Load γ (class priors) for this layer.
        """
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}_gam.npy")
        self.gam = np.load(save_path, allow_pickle=False)
        return self.gam

    # ---- incremental-learning helpers (paths) ----
    def _stats_dir(self):
        return os.path.join(self.arch.model_dir, "stats")

    def _stats_path(self, layer):
        return os.path.join(self._stats_dir(), f"{self.block_id}_{layer}.npz")

    def save_stats(self, layer, stats):
        os.makedirs(self._stats_dir(), exist_ok=True)
        np.savez_compressed(self._stats_path(layer),
                            S_total=stats.S_total, m_total=stats.m_total,
                            S_j=stats.S_j, m_j=stats.m_j)

    def load_stats(self, layer):
        path = self._stats_path(layer)
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=False)
        return LayerStats(S_total=data["S_total"],
                          m_total=int(data["m_total"]),
                          S_j=data["S_j"], m_j=data["m_j"])

    def _init_empty_stats(self, d):
        k = self.num_classes
        return LayerStats(S_total=np.zeros((d, d), dtype=float),
                          m_total=0,
                          S_j=np.zeros((k, d, d), dtype=float),
                          m_j=np.zeros((k,), dtype=int))

    def update_stats_with_batch(self, stats, Z, y):
        """
        Accumulate sufficient statistics from NEW data only.
        Z: (m, d) at THIS layer's input; y: (m,)
        """
        m, d = Z.shape
        stats.S_total += Z.T @ Z
        stats.m_total += m
        for j in range(self.num_classes):
            idx = (y == int(j))
            if np.any(idx):
                Zj = Z[idx, :]
                stats.S_j[j] += Zj.T @ Zj
                stats.m_j[j] += int(idx.sum())
        return stats

    def compute_from_stats(self, stats):
        """
        Recompute gamma, E, and C_j from sufficient statistics only.
        """
        m_total = max(1, stats.m_total)  # safety
        self.gam = stats.m_j / m_total

        d = stats.S_total.shape[0]
        I = np.eye(d)

        # E = α (I + α S_total)^(-1),   α = d / (m * eps)
        c = d / (m_total * self.eps)
        self.E = c * np.linalg.inv(I + c * stats.S_total)

        # C_j = α_j (I + α_j S_j)^(-1), α_j = d / (m_j * eps)
        Cs = np.zeros((self.num_classes, d, d), dtype=float)
        for j in range(self.num_classes):
            mj = int(stats.m_j[j])
            if mj <= 0:
                Cs[j] = np.zeros((d, d), dtype=float)
                continue
            cj = d / (mj * self.eps)
            Cs[j] = cj * np.linalg.inv(I + cj * stats.S_j[j])
        self.Cs = Cs
        return self.E, self.Cs, self.gam

    def forward_incremental(self, layer, Z_new, y_new):
        """
        Incremental update for ONE layer with ONLY the new task's data:
          1) Load existing stats (if any); else init.
          2) Update stats with (Z_new, y_new) at THIS layer's input space.
          3) Recompute (E, C_j, γ) from stats; save weights & γ & stats.
          4) Compute expd/comp/soft-agg and residual update for Z_new only.
        Returns Z_out for the new data and y_approx (preds for monitoring).
        """
        # load or init stats
        d = Z_new.shape[1]
        stats = self.load_stats(layer)
        if stats is None:
            stats = self._init_empty_stats(d)

        # update stats with the NEW task batch
        stats = self.update_stats_with_batch(stats, Z_new, y_new)

        # recompute weights from stats (no access to old raw data)
        self.compute_from_stats(stats)

        # save everything (re-uses your existing helpers)
        self.save_weights(layer)
        self.save_gam(layer)
        self.save_stats(layer, stats)

        # forward only the NEW batch through this updated layer
        expd = Z_new @ self.E.T
        comp = np.stack([Z_new @ C.T for C in self.Cs])
        clus, y_approx = self.nonlinear(comp)
        Z_out = F.normalize(Z_new + self.eta * (expd - clus))

        # (optional) track loss on the NEW batch using y_new
        self.arch.update_loss(layer, *self.compute_loss(Z_out, y_new))
        return Z_out, y_approx
