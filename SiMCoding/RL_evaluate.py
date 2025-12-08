import argparse
import os
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import cluster
import train_func as tf
import utils

from sklearn.metrics import balanced_accuracy_score

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from sklearn.metrics import normalized_mutual_info_score
import torch

from sklearn.metrics import balanced_accuracy_score
from cluster import ElasticNetSubspaceClustering

import torch
from sklearn.metrics import balanced_accuracy_score


def svm_linear(args, train_features, train_labels, test_features, test_labels):
    train_features = train_features.cpu()
    train_labels = train_labels.cpu()
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()

    svm = LinearSVC(verbose=0, random_state=42, dual=True)
    svm.fit(train_features, train_labels)
    test_pred = svm.predict(test_features)
    balanced_acc = balanced_accuracy_score(test_labels, test_pred)
    return balanced_acc


def knn(args, train_features, train_labels, test_features, test_labels,
        k=None, test_bs=1024, train_bs=8192, normalize=True):
    if k is None:
        k = args.k

    train = train_features.to('cpu', dtype=torch.float32).contiguous()
    test = test_features.to('cpu', dtype=torch.float32).contiguous()
    labels = train_labels.to('cpu', dtype=torch.long)

    N_test = test.size(0)
    preds = []

    for i in range(0, N_test, test_bs):
        t_batch = test[i:i + test_bs]

        best_scores = torch.full((t_batch.size(0), k), -float("inf"))
        best_idx = torch.full((t_batch.size(0), k), -1, dtype=torch.long)

        for j in range(0, train.size(0), train_bs):
            tr_block = train[j:j + train_bs]
            sims = t_batch @ tr_block.T
            local_scores, local_idx = sims.topk(min(k, tr_block.size(0)), dim=1)
            local_idx += j
            merged_scores = torch.cat([best_scores, local_scores], dim=1)
            merged_idx = torch.cat([best_idx, local_idx], dim=1)
            best_scores, keep_pos = merged_scores.topk(k, dim=1)
            best_idx = merged_idx.gather(1, keep_pos)
            del sims, local_scores, local_idx, merged_scores, merged_idx

        topk_labels = labels[best_idx]
        preds.append(torch.mode(topk_labels, dim=1).values)

    y_pred = torch.cat(preds).numpy()
    y_true = test_labels.cpu().numpy()
    return balanced_accuracy_score(y_true, y_pred)


def linear_classifier(args, train_features, train_labels, test_features, test_labels):
    train_features = train_features.cpu()
    train_labels = train_labels.cpu()
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(
        random_state=42, max_iter=1000, multi_class='auto', solver='lbfgs'
    )
    classifier.fit(train_features, train_labels)
    test_pred = classifier.predict(test_features)
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(test_labels.numpy(), test_pred)
    return balanced_acc


def nearcenter(args, train_features, train_labels, test_features, test_labels):
    train_features = train_features.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_features = test_features.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    num_classes = train_labels.max() + 1
    feature_dim = train_features.shape[1]

    class_means = np.zeros((num_classes, feature_dim))
    for j in range(num_classes):
        class_means[j] = train_features[train_labels == j].mean(axis=0)

    distances = []
    for j in range(num_classes):
        diff = test_features - class_means[j]
        dist_j = np.linalg.norm(diff, axis=1)
        distances.append(dist_j)

    distances = np.stack(distances, axis=0)
    test_predict = np.argmin(distances, axis=0)
    balanced_acc = balanced_accuracy_score(test_labels, test_predict)
    return balanced_acc


def get_all_acc(args, train_features, train_labels, test_features, test_labels):
    acc_svm_linear = svm_linear(args, train_features, train_labels, test_features, test_labels)
    acc_knn = knn(args, train_features, train_labels, test_features, test_labels)
    acc_ncc = nearcenter(args, train_features, train_labels, test_features, test_labels)
    acc_linear = linear_classifier(args, train_features, train_labels, test_features, test_labels)
    return acc_svm_linear, acc_knn, acc_ncc, acc_linear


def ensc(args, test_features, test_labels):
    if isinstance(test_features, torch.Tensor):
        test_features_np = test_features.cpu().numpy()
    else:
        test_features_np = test_features

    model = ElasticNetSubspaceClustering(
        n_clusters=args.n,
        algorithm='lasso_lars',
        active_support=True,
        active_support_params={'support_size': 50, 'maxiter': 100}
    )

    model.fit(test_features_np)
    pred_labels = model.labels_
    return pred_labels


def normalised_mutual_information(pred_labels, test_labels, average_method='geometric'):
    nmi = normalized_mutual_info_score(test_labels, pred_labels, average_method=average_method)
    return nmi


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        confusion_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    acc = confusion_matrix[row_ind, col_ind].sum() / y_pred.size
    return acc


def adjusted_rand_index(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)


def get_all_slefsup_acc(args, test_features, test_labels):
    test_features = test_features.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    save_path = os.path.join(args.model_dir, 'ensc_pred_labels.npy')

    if not os.path.exists(save_path):
        pred_label = ensc(args, test_features, test_labels)
        np.save(save_path, pred_label)
    else:
        pred_label = np.load(save_path)

    NMI = normalised_mutual_information(pred_label, test_labels)
    ACC = clustering_accuracy(pred_label, test_labels)
    ARI = adjusted_rand_index(pred_label, test_labels)

    return NMI, ACC, ARI
