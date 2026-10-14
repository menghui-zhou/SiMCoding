# SiMCoding

Code for **Less Structure is More: Minimal Representations for Supervised Learning** (NeurIPS 2026), by Menghui Zhou, Vitaveska Lanfranchi, and Po Yang.

This repository provides SiMCoding implementations for supervised representation learning and incremental learning, with training, evaluation, and CSV result output.

| Experiment | Dataset | Entry point | Result file |
| --- | --- | --- | --- |
| Coding precision, effective rank, and accuracy | ImageNette | `experiments.precision_sweep` | `outputs/precision_sweep/precision_sweep.csv` |
| Robustness to label noise | CIFAR-20 (coarse CIFAR-100 labels) | `experiments.label_noise` | `outputs/label_noise/label_noise.csv` |
| Incremental learning | MNIST and ImageNette | `experiments.incremental` | `outputs/incremental/<dataset>/incremental.csv` |

All final classifiers are **logistic regression with `C=1.0`, `solver="lbfgs"`, and `max_iter=1000`**, fitted on training features and labels only. Accuracy is reported as a percentage. No pretrained backbone is required.

The SiMCoding model architectures for PD Activity and 3D MRI are available in [`simcoding/architectures`](simcoding/architectures).

## Installation

Use Python 3.10 or later. A CUDA-capable GPU is recommended for the full experiments.

```bash
git clone https://github.com/menghui-zhou/SiMCoding.git
cd SiMCoding
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a particular CUDA version, install a matching PyTorch/torchvision pair using the [official PyTorch instructions](https://docs.pytorch.org/get-started/locally/) before installing the remaining requirements. Run commands from the repository root; alternatively, install with `python -m pip install -e .`.

## Datasets

The `--download` flag downloads MNIST and CIFAR-100 through torchvision, and the 320-pixel version of [ImageNette](https://github.com/fastai/imagenette) from its official download URL. CIFAR-20 uses the original fine-to-coarse label mapping in `simcoding/labels.py`. No dataset is distributed in this repository; the original dataset terms apply.

The default layout is:

```text
data/
  MNIST/raw/                  # MNIST
  cifar100/cifar-100-python/  # CIFAR-20
  imagenette/imagenette2-320/
    train/<class>/*.JPEG
    val/<class>/*.JPEG
```

Use `--data-root /path/to/data` to change the root. For an existing ImageNette dataset, use `--imagenette-dir /path/to/imagenette2-320`; this directory must contain `train/` and `val/` with the same ten class folders. The official validation split is used as the held-out test split.

## Coding precision on ImageNette

```bash
python -m experiments.precision_sweep --download
```

This trains separate models with `eps2 / 16`, `eps2 / 8`, `eps2 / 4`, `eps2 / 2`, and `eps2`, where `eps2=53.2` by default. **`--eps2` always means epsilon squared**, not epsilon. Each row reports training/test accuracy, global effective rank, and mean class-wise effective rank. Effective rank counts singular values strictly larger than 50% of the largest singular value, computed from training features.

The defaults are ResNet-18, 512-dimensional normalized features, 128-pixel images, 500 epochs, batch size 400, and SGD with learning rate 0.01, momentum 0.9, and weight decay 0.0005. The learning rate is divided by ten after epochs 200 and 400.

A single setting can be run with:

```bash
python -m experiments.precision_sweep --download --divisors 1 --output outputs/precision_sweep_single
```

## CIFAR-20 label noise

```bash
python -m experiments.label_noise --download --train-repeats 5
```

This runs noise ratios `0, 0.1, 0.2, 0.3, 0.4, 0.5`, with `eps2=38.2`, ResNet-18, 512-dimensional features, 500 epochs, and batch size 1000. The optimizer uses the same settings as the coding-precision experiment. Training augmentation uses `RandomCrop(32, padding=8)` and horizontal flips. `--train-repeats 5` provides fivefold stochastic augmentation; omit it for a shorter single-pass training dataset.

Label corruption preserves the original code's uniform resampling convention: selected labels can be reassigned to their original class. Consequently, the fraction of labels actually changed is approximately `ratio * 19/20`. Use `--exclude-original-label` to force every selected label to change. Only training labels are corrupted; the test labels remain clean. Logistic regression is fitted to the training labels used by the model. CSV files include both ordinary and balanced accuracy, so the metric is explicit.

To run only two noise levels:

```bash
python -m experiments.label_noise --download --noise-ratios 0 0.2 --output outputs/label_noise_short
```

## Incremental learning

```bash
python -m experiments.incremental --dataset mnist --download
python -m experiments.incremental --dataset imagenette --download
```

The original MNIST and ImageNette SiM-ReduNet updates are retained as separate cores. Each experiment introduces two classes per task in the order `(0,1), (2,3), (4,5), (6,7), (8,9)`. There are 50 layers by default. The original script defaults are `eps2=80, eta=0.5` for MNIST and `eps2=300, eta=10` for ImageNette. ImageNette is resized to 32 x 32 RGB pixels. Parameters can be changed with `--layers`, `--eps2`, `--eta`, and `--image-size`.

`incremental.csv` reports **Task 1 test accuracy after each of the five sessions**, its arithmetic mean (`Avg. Acc.`), and the relative forgetting drop:

```text
Drop (%) = 100 * (Task 1 accuracy after session 1 - after session 5)
                 / Task 1 accuracy after session 1
```

`sessions.csv` additionally records accuracy on all classes seen so far. The final logistic classifier uses all seen training features; stored covariance statistics drive the incremental representation updates.

The vector implementation stores dense covariance matrices and layer weights. Full ImageNette experiments require substantial memory and disk space because the flattened input dimension is 3072. Use fewer layers, smaller images, or the small-run options below when first checking the installation.

## Small runs and output

For an inexpensive end-to-end check with a small subset:

```bash
python -m experiments.precision_sweep --download --epochs 1 --divisors 1 \
  --image-size 32 --feature-dim 32 --batch-size 20 --workers 0 \
  --train-per-class 4 --test-per-class 2 --max-batches 1 \
  --output outputs/check_precision_sweep

python -m experiments.label_noise --download --epochs 1 --noise-ratios 0 0.2 \
  --feature-dim 32 --batch-size 40 --workers 0 \
  --train-per-class 8 --test-per-class 2 --max-batches 1 \
  --output outputs/check_label_noise

python -m experiments.incremental --dataset mnist --download --layers 2 \
  --train-per-class 4 --test-per-class 2 --workers 0 \
  --output outputs/check_incremental

python -m experiments.incremental --dataset imagenette --download --layers 2 \
  --image-size 8 --train-per-class 4 --test-per-class 2 --workers 0 \
  --output outputs/check_incremental
```

These small runs check the training and evaluation pipeline. Use the full defaults for longer experiments. For repeated runs, change `--seed` and use a different `--output` directory. Runs do not silently reuse existing checkpoints. Configuration JSON, numerical training logs, metrics, and checkpoints are stored beneath the output directory; no figures are generated.

Fast numerical checks can be run with:

```bash
python -m unittest discover -s tests -v
```

See [implementation notes](docs/implementation.md) for the relationship to the original research scripts and the tested environment.

## Citation and acknowledgments

Please cite our paper when using this code:

```bibtex
@inproceedings{zhou2026simcoding,
  title={Less Structure is More: Minimal Representations for Supervised Learning},
  author={Zhou, Menghui and Lanfranchi, Vitaveska and Yang, Po},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

Our incremental implementation builds on the rate-reduction and ReduNet principles studied by [Yu et al.](https://proceedings.neurips.cc/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html), [Chan et al.](https://arxiv.org/abs/2010.14765), and Wu et al. (2021). We thank these authors for their foundational work. This work was supported by Innovate UK Project 10081615.
