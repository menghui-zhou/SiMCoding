import numpy as np  # Importing the numpy library for numerical operations
import torch  # Importing the torch library for tensor operations


import numpy as np

# --- small helpers to read/write labels in a dataset-agnostic way ---

def _get_labels_array(ds):
    # For ImageFolder, labels used by __getitem__ live in ds.samples (and ds.imgs).
    if hasattr(ds, "samples") and isinstance(ds.samples, list) and len(ds.samples) > 0:
        return np.array([lab for _, lab in ds.samples], dtype=int)
    # Fallback for CIFAR/MNIST-style datasets
    return np.array(ds.targets, dtype=int)

def _set_label(ds, idx, new_lab: int):
    # Update the true source-of-truth for labels
    if hasattr(ds, "samples") and isinstance(ds.samples, list) and len(ds.samples) > 0:
        p, _ = ds.samples[idx]
        ds.samples[idx] = (p, int(new_lab))
        if hasattr(ds, "imgs"):  # keep alias in sync for older torchvision
            ds.imgs[idx] = (p, int(new_lab))
    # Keep ds.targets in sync and as list[int]
    if hasattr(ds, "targets"):
        if not isinstance(ds.targets, list):
            ds.targets = list(map(int, np.array(ds.targets, dtype=int).tolist()))
        ds.targets[idx] = int(new_lab)

# --- the corrupter you want: uniform over classes (allows no-op reassignments) ---

def corrupt_labels_uniform(ds, ratio: float, seed: int = 42, include_self: bool = True):
    """
    Corrupt exactly `ratio` of samples by resampling their label from a uniform
    distribution over classes. If `include_self=True`, the new label may equal
    the old one (expected unchanged fraction ≈ 1/C).

    Works for ImageFolder (updates samples/imgs) and CIFAR/MNIST (updates targets).
    """
    assert 0 <= ratio <= 1
    rng = np.random.default_rng(seed)

    # Save originals once for sanity checks
    if not hasattr(ds, "targets_original"):
        ds.targets_original = _get_labels_array(ds).copy()

    labels = _get_labels_array(ds)
    n = len(labels)
    if n == 0:
        print("[LabelNoise] dataset is empty; nothing to corrupt.")
        return ds

    # Infer #classes robustly
    num_classes = getattr(ds, "num_classes", int(labels.max()) + 1)

    n_rand = int(round(n * ratio))
    idxs = rng.choice(n, size=n_rand, replace=False)

    if include_self:
        # Uniform over all classes [0, C-1], so some will remain unchanged.
        new_labels = rng.integers(0, num_classes, size=n_rand)
    else:
        # Force a different class (for reference)
        new_labels = rng.integers(0, num_classes - 1, size=n_rand)
        # bump >= old to skip the original class without rejection
        new_labels = np.where(new_labels >= labels[idxs], new_labels + 1, new_labels)

    # Apply updates
    for i, nl in zip(idxs, new_labels):
        _set_label(ds, int(i), int(nl))

    # Sanity report based on what __getitem__ will use
    used_labels_after = _get_labels_array(ds)
    changed = (used_labels_after != ds.targets_original).sum()
    eff = changed / n
    expect_eff = ratio * (1.0 - (1.0 / num_classes)) if include_self else ratio
    print(f"[LabelNoise] requested={ratio:.3f}, include_self={include_self} "
          f"actual_changed={changed}/{n} ({eff:.2%}), "
          f"expected≈{expect_eff:.2%} (C={num_classes})")
    return ds

def default_corrupt(trainset, ratio, seed):
    """Corrupt labels in trainset.

    Parameters:
        trainset (torch.data.dataset): trainset where labels is stored
        ratio (float): ratio of labels to be corrupted. 0 to corrupt no labels;
                            1 to corrupt all labels
        seed (int): random seed for reproducibility

    Returns:
        trainset (torch.data.dataset): trainset with updated corrupted labels

    """
    np.random.seed(seed)  # Set the random seed for numpy for reproducibility
    train_labels = np.asarray(trainset.targets)  # Convert trainset targets to numpy array
    num_classes = np.max(train_labels) + 1  # Calculate the number of unique classes
    n_train = len(train_labels)  # Get the number of training samples
    # n_rand = int(len(trainset.data) * ratio)  # Calculate the number of labels to corrupt
    n_rand = int(len(trainset.targets) * ratio)
    randomize_indices = np.random.choice(range(n_train), size=n_rand,
                                         replace=False)  # Randomly select indices to corrupt
    # randomly corrupt
    train_labels[randomize_indices] = np.random.choice(np.arange(num_classes), size=n_rand,
                                                       replace=True)  # Corrupt the labels
    trainset.targets = torch.tensor(train_labels).int()  # Update trainset targets with corrupted labels
    return trainset  # Return the corrupted trainset



def shift_corrupt(trainset, ratio, seed):
    """Corrupt labels in trainset by cyclically shifting a portion of them.

    Parameters:
        trainset (torch.utils.data.Dataset): trainset where labels are stored
        ratio (float): ratio of labels to be cyclically shifted. 0 to shift no labels;
                            1 to shift all labels
        seed (int): random seed for reproducibility

    Returns:
        trainset (torch.utils.data.Dataset): trainset with updated cyclically shifted labels

    """
    np.random.seed(seed)  # Set the random seed for reproducibility

    # Convert trainset targets to numpy array
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1  # Calculate the number of unique classes
    n_train = len(train_labels)  # Get the number of training samples

    # Calculate the number of labels to cyclically shift
    n_shift = int(n_train * ratio)

    # Randomly select indices to apply the cyclic shift
    shift_indices = np.random.choice(range(n_train), size=n_shift, replace=False)

    # Apply cyclic shift: label k becomes (k + 1) % num_classes, last label cycles to 0
    train_labels[shift_indices] = (train_labels[shift_indices] + 1) % num_classes

    # Update trainset targets with cyclically shifted labels
    trainset.targets = torch.tensor(train_labels).int()

    return trainset


def cyclic_corrupt(trainset, ratio, seed):
    """
    对trainset中部分标签进行循环打乱，即随机选取一部分标签进行 (标签 + 1) % num_classes 的操作。

    参数：
        trainset: 包含targets属性的训练数据集
        ratio: 要打乱标签的比例（0到1之间）
        seed: 随机种子，确保结果可复现

    返回：
        trainset: 更新后标签部分被循环打乱的训练数据集
    """
    assert 0 <= ratio <= 1., 'ratio is bounded between 0 and 1'  # Ensure ratio is between 0 and 1
    np.random.seed(seed)
    # 将标签转换为numpy数组
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1
    n_train = len(train_labels)
    # 计算要打乱的样本数量
    n_shift = int(n_train * ratio)
    # 随机选取需要打乱的索引
    shift_indices = np.random.choice(n_train, size=n_shift, replace=False)
    # 对选中的标签进行循环打乱
    train_labels[shift_indices] = (train_labels[shift_indices] + 1) % num_classes
    # 更新trainset的targets
    trainset.targets = torch.tensor(train_labels).int()
    return trainset


## https://github.com/shengliu66/ELR/blob/909687a4621b742cb5b8b44872d5bc6fce38bdd3/ELR/data_loader/cifar10.py#L82
def asymmetric_noise(trainset, ratio, seed):
    assert 0 <= ratio <= 1., 'ratio is bounded between 0 and 1'  # Ensure ratio is between 0 and 1
    np.random.seed(seed)  # Set the random seed for numpy for reproducibility
    train_labels = np.array(trainset.targets)  # Convert trainset targets to numpy array
    train_labels_gt = train_labels.copy()  # Make a copy of the original labels
    for i in range(trainset.num_classes):  # Iterate over each class
        indices = np.where(train_labels == i)[0]  # Get indices of the current class
        np.random.shuffle(indices)  # Shuffle the indices
        for j, idx in enumerate(indices):  # Iterate over the shuffled indices
            if j < ratio * len(indices):  # Check if the index should be corrupted
                #                 self.noise_indx.append(idx)  # This line is commented out, it's not needed
                # Class-specific corruption rules
                if i == 9:
                    train_labels[idx] = 1  # truck -> automobile
                elif i == 2:
                    train_labels[idx] = 0  # bird -> airplane
                elif i == 3:
                    train_labels[idx] = 5  # cat -> dog
                elif i == 5:
                    train_labels[idx] = 3  # dog -> cat
                elif i == 4:
                    train_labels[idx] = 7  # deer -> horse
    trainset.targets = torch.tensor(train_labels).int()  # Update trainset targets with corrupted labels
    return trainset  # Return the corrupted trainset


# https://github.com/xiaoboxia/T-Revision/blob/b984283b884c13eb59ed0f8d435f4eda548ab26a/data/utils.py#L125
# noisify_pairflip call the function "multiclass_noisify"
# This function applies pair-flip noise to the labels of a given training dataset,
# introducing mislabeling between adjacent classes based on a specified noise level.
def noisify_pairflip(trainset, noise, seed=None):
    """mistakes:
        flip in the pair
    """
    y_train = np.array(trainset.targets)  # Convert trainset targets to numpy array
    nb_classes = np.unique(trainset.targets).size  # Calculate the number of unique classes
    P = np.eye(nb_classes)  # Initialize the transition matrix as identity matrix
    n = noise  # Noise level

    if n > 0.0:  # If noise level is greater than 0
        # Modify the transition matrix for pair flipping
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=seed)  # Apply noise
        actual_noise = (y_train_noisy != y_train).mean()  # Calculate actual noise
        assert actual_noise > 0.0  # Ensure noise is applied
        #         print('Actual noise %.2f' % actual_noise)  # This line is commented out, it's not needed
        y_train = y_train_noisy  # Update the labels with noisy labels

    trainset.targets = torch.tensor(y_train)  # Update trainset targets with noisy labels
    return trainset  # Return the corrupted trainset


# https://github.com/xiaoboxia/T-Revision/blob/b984283b884c13eb59ed0f8d435f4eda548ab26a/data/utils.py#L149
def noisify_multiclass_symmetric(trainset, noise, seed=10):
    """mistakes:
        flip in the symmetric way
    """
    y_train = np.array(trainset.targets)  # Convert trainset targets to numpy array
    nb_classes = np.unique(y_train).size  # Calculate the number of unique classes
    P = np.ones((nb_classes, nb_classes))  # Initialize the transition matrix with ones
    n = noise  # Noise level
    P = (n / (nb_classes - 1)) * P  # Adjust transition matrix for symmetric noise

    if n > 0.0:  # If noise level is greater than 0
        # Modify the transition matrix for symmetric noise
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=seed)  # Apply noise
        actual_noise = (y_train_noisy != y_train).mean()  # Calculate actual noise
        assert actual_noise > 0.0  # Ensure noise is applied
        #         print('Actual noise %.2f' % actual_noise)  # This line is commented out, it's not needed
        y_train = y_train_noisy  # Update the labels with noisy labels

    trainset.targets = torch.tensor(y_train)  # Update trainset targets with noisy labels
    return trainset  # Return the corrupted trainset


#### Helper
def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    #     print (np.max(y), P.shape[0])  # This line is commented out, it's not needed
    assert P.shape[0] == P.shape[1]  # Ensure P is a square matrix
    assert np.max(y) < P.shape[0]  # Ensure max label is less than number of classes

    # row stochastic matrix
    assert np.allclose(P.sum(axis=1), np.ones(P.shape[1]))  # Ensure rows sum to 1
    assert (P >= 0.0).all()  # Ensure all probabilities are non-negative

    m = y.shape[0]  # Number of samples
    #     print(m)  # This line is commented out, it's not needed
    new_y = y.copy()  # Make a copy of the labels
    flipper = np.random.RandomState(random_state)  # Initialize random state

    for idx in np.arange(m):  # Iterate over each sample
        i = y[idx]  # Get the current label
        # draw a vector with only a 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]  # Flip the label based on transition matrix
        new_y[idx] = np.where(flipped == 1)[0]  # Update the label

    return new_y  # Return the noisy labels


