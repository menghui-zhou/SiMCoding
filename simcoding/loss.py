"""The original SiMCoding objective with device-independent class selection."""
import torch


class SiMCoding(torch.nn.Module):
    """Negative rate reduction; eps2 denotes epsilon squared in the paper.

    Replaces dense diagonal membership matrices with equivalent boolean masks.
    The smaller of the sample Gram and feature Gram matrices is used.
    """

    def __init__(self, eps2, gam1=1.0, gam2=1.0):
        super().__init__()
        if eps2 <= 0:
            raise ValueError("eps2 must be positive.")
        self.eps2 = eps2
        self.gam1 = gam1
        self.gam2 = gam2

    def coding_rate(self, features, multiplier=1.0):
        m, d = features.shape
        gram = features @ features.T if m < d else features.T @ features
        eye = torch.eye(gram.shape[0], device=features.device, dtype=features.dtype)
        return 0.5 * torch.logdet(eye + multiplier * d / (m * self.eps2) * gram)

    def forward(self, features, labels):
        if features.ndim != 2 or len(features) != len(labels) or not len(features):
            raise ValueError("Expected a nonempty (samples, features) tensor and matching labels.")
        expansion = self.coding_rate(features, self.gam1)
        compression = features.new_zeros(())
        for label in torch.unique(labels):
            group = features[labels == label]
            compression = compression + len(group) / len(features) * self.coding_rate(group)
        return -self.gam2 * expansion + compression
