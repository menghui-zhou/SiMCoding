"""Fast checks for objective equivalence, labels, ranks and incremental updates."""
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from simcoding.data import LabeledDataset
from simcoding.evaluation import effective_rank, fit_logistic
from simcoding.incremental import mnist_core, imagenette_core
from simcoding.labels import corrupt_labels_uniform, sparse2coarse
from simcoding.loss import SiMCoding


def reference_loss(features, labels, eps2):
    """Direct feature-Gram expression, independent of the small-Gram implementation."""
    m, d = features.shape
    eye = torch.eye(d, dtype=features.dtype)
    expansion = torch.logdet(eye + d / (m * eps2) * features.T @ features) / 2
    compression = 0.0
    for label in torch.unique(labels):
        group = features[labels == label]
        compression += len(group) / m * torch.logdet(
            eye + d / (len(group) * eps2) * group.T @ group) / 2
    return -expansion + compression


class CoreTests(unittest.TestCase):
    def test_loss_and_gradient_match_direct_objective(self):
        for m, d in [(6, 13), (13, 6)]:
            torch.manual_seed(5)
            X = F.normalize(torch.randn(m, d, dtype=torch.float64), dim=1).requires_grad_()
            labels = torch.arange(m) % 3
            actual = SiMCoding(3.2)(X, labels)
            expected = reference_loss(X, labels, 3.2)
            torch.testing.assert_close(actual, expected)
            actual_grad, = torch.autograd.grad(actual, X, retain_graph=True)
            expected_grad, = torch.autograd.grad(expected, X)
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_cifar20_contains_twenty_groups_of_five(self):
        groups = sparse2coarse(list(range(100)))
        np.testing.assert_array_equal(np.bincount(groups), np.full(20, 5))

    def test_noise_wrapper_updates_labels_used_by_getitem(self):
        class TinyDataset:
            targets = [0, 1] * 10
            def __len__(self): return 20
            def __getitem__(self, i): return torch.ones(3), self.targets[i]
        original = TinyDataset()
        wrapped = LabeledDataset(original)
        corrupt_labels_uniform(wrapped, 1.0, seed=4, include_self=False)
        self.assertTrue(all(wrapped[i][1] != original[i][1] for i in range(20)))
        self.assertEqual(original.targets, [0, 1] * 10)

    def test_effective_rank_uses_strict_relative_cutoff(self):
        self.assertEqual(effective_rank(np.diag([4., 2., 1.])), 1)
        self.assertEqual(effective_rank(np.zeros((4, 3))), 0)

    def test_logistic_C_is_one(self):
        clf = fit_logistic(np.array([[-2.], [-1.], [1.], [2.]]), np.array([0, 0, 1, 1]))
        self.assertEqual(clf.C, 1.0)

    def test_incremental_cores_accumulate_new_class_counts(self):
        for core in [mnist_core, imagenette_core]:
            with tempfile.TemporaryDirectory() as folder:
                block = core.Vector(layers=2, eta=0.5, eps=3.0, device=torch.device("cpu"))
                core.Architecture([block], folder, num_classes=4)
                for task in range(2):
                    torch.manual_seed(task)
                    X = np.random.default_rng(task).normal(size=(8, 6)).astype(np.float32)
                    y = np.tile([task * 2, task * 2 + 1], 4)
                    if core is mnist_core:
                        block.fit_task_incremental(X, y)
                    else:
                        Z = block.preprocess(X)
                        for layer in range(block.layers):
                            Z, _ = block.forward_incremental(layer, Z, torch.tensor(y))
                if core is mnist_core:
                    E, Cs, gam, state = block.load_layer(0)
                    self.assertEqual(int(state.m_j.sum()), 16)
                else:
                    E, Cs, gam = block.E, block.Cs, block.gam
                    self.assertEqual(block.stats[0].m_total, 16)
                self.assertTrue(torch.isfinite(E).all())
                self.assertTrue(torch.isfinite(Cs).all())
                torch.testing.assert_close(gam, torch.full((4,), 0.25))

    def test_memory_efficient_class_update_matches_original(self):
        block = mnist_core.Vector(layers=1, eta=0.5, eps=3.0, device=torch.device("cpu"))
        torch.manual_seed(1)
        Z, E, Cs = torch.randn(8, 6), torch.randn(6, 6), torch.randn(3, 6, 6)
        y, gam = torch.arange(8) % 3, torch.tensor([.4, .3, .3])
        gathered = torch.bmm(Z.unsqueeze(1), Cs[y].transpose(1, 2)).squeeze(1)
        expected = F.normalize(Z + block.eta * (Z @ E.T - gam[y, None] * gathered), dim=1)
        torch.testing.assert_close(block.train_layer_update(Z, y, E, Cs, gam), expected)


if __name__ == "__main__":
    unittest.main()
