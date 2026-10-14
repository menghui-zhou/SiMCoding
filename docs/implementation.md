# Relationship to the research code

This release was extracted from the authors' SiMCoding and Incremental_ReduNet experiment directories. It retains the original SiMCoding objective, ResNet-18 feature network, CIFAR-20 label mapping, uniform label-corruption function, and the separate MNIST/ImageNette incremental layer updates. Original file hashes are recorded in `source_manifest.json`.

| Release file | Original implementation |
| --- | --- |
| `simcoding/loss.py` | `loss_SiMCoding.py` |
| `simcoding/models.py` | `architectures/resnet_cifar.py` (BasicBlock, ResNet, ResNet18) |
| `simcoding/labels.py` | `train_func.py` (coarse mapping), `corrupt.py` (uniform corruption) |
| `simcoding/train.py` | training and SGD loop in `train_sup_SiMCoding.py` |
| `simcoding/incremental/mnist_core.py` | `Modified_Incre_MNIST.py` |
| `simcoding/incremental/imagenette_core.py` | `Incremental_ImageNette_Resize28*28*3.py` |

The release removes unrelated tasks, baseline implementations, duplicated classifier evaluations, plotting dependencies, and automatic checkpoint discovery. The original dense membership matrices in the loss are replaced by equivalent label masks, and CPU/CUDA placement is explicit. The MNIST per-class update avoids allocating a separate d x d matrix for every sample; a numerical test compares it against the original gathered-matrix operation. Inference is evaluated in batches using the original layer update equations.

The experiment wrappers save CSV tables and use logistic regression with C=1. Evaluation uses deterministic transforms and includes every selected test sample. The supervised ImageNette default is 128 pixels, following the manuscript; the currently stored research-script transform uses 64 pixels, which can be selected with `--image-size 64`. CIFAR training repetition is explicit through `--train-repeats`, rather than hidden in a dataset loader. These choices make the package easier to run and inspect; the release does not guarantee an exact match to the paper's numeric entries.

## Validation

The server test environment uses Python 3.10.13, PyTorch 2.10.0.dev20250916+cu128, torchvision 0.25.0.dev20250916+cu128, NumPy 1.26.3, and scikit-learn 1.3.0. This is the authors' existing environment, not a requirement to install nightly builds. A compatible stable PyTorch/torchvision pair is sufficient to run the code.

Validation includes loss-value/gradient agreement with the direct rate-reduction formula, coarse-label mapping, label-corruption behavior, the relative-rank cutoff, C=1 logistic regression, and covariance updates across two incremental tasks. Small end-to-end runs exercise the table entry points on existing datasets. Full 500-epoch and 50-layer experiments are not rerun as part of packaging.
