# Import necessary libraries and modules
import argparse  # Used for parsing command line arguments
import os  # Provides functions for interacting with the operating system
import time

# Scientific computing and deep learning libraries
import numpy as np  # Fundamental package for scientific computing with Python
import torch  # PyTorch library for deep learning
from torch.utils.data import DataLoader  # DataLoader class for loading data in batches
from augmentloader import AugmentLoader  # Custom module for data augmentation, not standard in PyTorch
import torch.optim as optim  # Submodule that contains standard optimization operations like SGD, Adam
import torch.optim.lr_scheduler as lr_scheduler  # Submodule for learning rate scheduling

# Import custom modules for specific functionalities
import train_func as tf  # Contains functions related to model training (e.g., loading models, data transformations)
from loss import MaximalCodingRateReduction  # Custom loss function for training
from loss_RL import RoughLearning
import utils  # Utility functions (e.g., for initializing training pipeline, saving parameters)
from utils import save_model
import sys
from feature_visualisation import feature_similarity_analysis, extract_features
import logging
from Inf_quantity import calculate_information_quantification
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pandas as pd
import RL_evaluate

import random
from torch.utils.data import DataLoader, Subset

from torchvision import transforms

print = logging.info

torch.cuda.empty_cache()
import matplotlib.pyplot as plt


def evaluate_and_save_metrics(epoch, model, train_loader, test_loader, args):
    """
    Evaluate and save metrics including coding length, spectral entropy,
    variable information, and classification accuracy during training.

    Parameters:
    - epoch (int): Current epoch.
    - model (torch.nn.Module): The model being trained.
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    - args: Argument parser with relevant training configurations.
    """

    # Extract train features and labels
    train_features, train_labels = extract_features(model, train_loader)

    # # Calculate information quantification
    # overall_coding_length, overal_spectral_entropy, overall_var_information, coding_length, spectral_entropy, var_information = \
    #     calculate_information_quantification(train_features, train_labels, args)
    # to_print = f'Train: [{epoch}]\tCoding Length {coding_length:.3f}\t' \
    #            f'Spectral Entropy {spectral_entropy:.3f}\tVar Information {var_information:.3f}'
    # print(to_print)
    # sys.stdout.flush()
    # utils.save_information_quantity(args.model_dir, epoch, coding_length, spectral_entropy, var_information)

    # Calculate information quantification
    (
        overall_coding_length,
        overall_spectral_entropy,
        overall_var_information,
        coding_length,
        spectral_entropy,
        var_information
    ) = calculate_information_quantification(train_features, train_labels, args)

    # Prepare formatted string for logging (Overall first)
    to_print = (
        f"Train: [Epoch {epoch}] "
        f"\tOverall Coding Length: {overall_coding_length:.3f} "
        f"\tOverall Spectral Entropy: {overall_spectral_entropy:.3f} "
        f"\tOverall Var Information: {overall_var_information:.3f}"
        f"\tCoding Length: {coding_length:.3f} "
        f"\tSpectral Entropy: {spectral_entropy:.3f} "
        f"\tVar Information: {var_information:.3f}"
    )

    # Print the formatted string
    print(to_print)
    sys.stdout.flush()

    # Save the information quantities to a file or log
    utils.save_information_quantity(
        args.model_dir,
        epoch,
        overall_coding_length,
        overall_spectral_entropy,
        overall_var_information,
        coding_length,
        spectral_entropy,
        var_information
    )

    # Evaluate train set accuracy
    train_accuracies = RL_evaluate.get_all_acc(args, train_features, train_labels, train_features, train_labels)
    to_print = f'Train: [{epoch}]\t' \
               f'Train Accuracy: Linear_SVM {train_accuracies[0]:.4f}\t' \
               f'KNN {train_accuracies[1]:.4f}\t' \
               f'NCC {train_accuracies[2]:.4f}\t' \
               f'LogisticSR {train_accuracies[3]:.4f}'
    print(to_print)
    sys.stdout.flush()
    utils.save_accuracy(args.model_dir, epoch, *train_accuracies, filename='train_accuracy.csv')

    # Extract test features and labels
    test_features, test_labels = extract_features(model, test_loader)

    # Evaluate test set accuracy
    test_accuracies = RL_evaluate.get_all_acc(args, train_features, train_labels, test_features, test_labels)
    to_print = f'Test: [{epoch}]\t' \
               f'Test Accuracy: Linear_SVM {test_accuracies[0]:.4f}\t' \
               f'KNN {test_accuracies[1]:.4f}\t' \
               f'NCC {test_accuracies[2]:.4f}\t' \
               f'LogisticSR {test_accuracies[3]:.4f}'
    print(to_print)
    sys.stdout.flush()
    utils.save_accuracy(args.model_dir, epoch, *test_accuracies, filename='test_accuracy.csv')

    # Perform feature similarity analysis at specified intervals
    if epoch % args.save_fea_ana_freq == 0:
        feature_similarity_analysis(train_features, train_labels, args.model_dir, epoch)


def parse_option():
    # Initialize the argument parser for command line argument parsing
    parser = argparse.ArgumentParser(description='Supervised Learning with Deep Neural modelworks')

    # Define command line arguments for various hyperparameters and training configurations
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers for loading data')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Model architecture, e.g., Resnet18 (resnet18), MNISTNet (minisnet)')
    parser.add_argument('--fd', type=int, default=512, help='Feature dimension for the model')
    parser.add_argument('--data', type=str, default='imagenettiny', help='Dataset name, e.g., CIFAR10')
    parser.add_argument('--epo', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--bs', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--mom', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay for regularization')
    parser.add_argument('--gam1', type=float, default=1., help='Gamma1 parameter for loss function tuning')
    parser.add_argument('--gam2', type=float, default=1., help='Gamma2 parameter for loss function tuning')
    parser.add_argument('--eps', type=float, default=0.5, help='Epsilon parameter for the loss function')
    parser.add_argument('--corrupt', type=str, default="default", help='Data corruption mode')
    parser.add_argument('--lcr', type=float, default=0, help='Label corruption ratio')
    parser.add_argument('--noise_std', type=float, default=0, help='Feature Gaussian noise std')
    parser.add_argument('--lcs', type=int, default=42, help='Seed for label corruption randomization')
    parser.add_argument('--trail', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--transform', type=str, default='test', help='Training Data transformation to apply')
    parser.add_argument('--test_transform', type=str, default='test', help='Test Data transformation to apply')
    parser.add_argument('--save_dir', type=str, default='./saved_models/ImageNetTiny/',
                        help='Directory to save trained models')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory containing dataset')
    parser.add_argument('--pretrain_dir', type=str, default=None, help='Directory of pretrained model to load')
    parser.add_argument('--pretrain_epo', type=int, default=None, help='Epoch of pretrained model to load')

    parser.add_argument('--optimizer', type=str, default='SGD', help='two options, SGD or Adam')
    # parser.add_argument('--resume', type=str, default='', help='resume ckpt path')

    parser.add_argument('--print_freq', type=int, default=60, help='print frequency in terms of every batch')
    parser.add_argument('--save_measure_freq', type=int, default=50, help='save measurement in terms of epoch')
    parser.add_argument('--save_model_freq', type=int, default=500, help='save mode frequency in terms of epoch')
    parser.add_argument('--save_fea_ana_freq', type=int, default=500, help='conduct feature analysis (time-consuming)')
    # parser.add_argument('--save_curr_freq', type=int, default=1, help='save frequency for the last epoch')

    #############################  evaluate model performance   ##########################################
    parser.add_argument('--k', type=int, default=20, help='top k components for kNN')
    parser.add_argument('--n', type=int, default=10, help='number of clusters for cluster (default: 10)')
    parser.add_argument('--n_comp', type=int, default=30, help='number of components for PCA (default: 30)')

    parser.add_argument('--resume', type=str,
                        default='',
                        help='resume ckpt path')

    # Parse the command line arguments
    args = parser.parse_args()

    # Construct a directory name based on training configuration for organized saving of models
    args.model_name = 'RL-eps:{}_sup_{}+{}_{}_epo_{}_bs_{}_opt_{}_lr_{}_mom_{}_wd_{}_gam1_{}_gam2_{}_lcr_{}_trail_{}'.format(
        args.eps, args.arch, args.fd, args.data, args.epo, args.bs, args.optimizer, args.lr, args.mom,
        args.wd, args.gam1, args.gam2, args.lcr, args.trail)

    args.model_dir = os.path.join(args.save_dir, args.model_name)

    # find the trained model
    folder_path = os.path.join(args.model_dir, "checkpoints")
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if "curr_epoch.pth" in file_name:
                args.resume = folder_path + "/curr_epoch.pth"
                args.need_train = False
                break

    if len(args.resume):
        args.model_name = args.resume.split('/')[-3]

    # Initialize the training setup, such as creating necessary directories and setting up logging
    utils.init_pipeline(args.model_dir)
    utils.save_params(args.model_dir, vars(args))  # Save initial training parameters for later reference

    # Configure logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    print(f"Model name: {args.model_name}")
    print(f"Arguments: {args}")

    return args


def set_loader(args):
    # Load dataset transformations and prepare the training dataset with optional label corruption
    base_transforms = tf.load_transforms(args.transform)
    gaussian_noise = tf.AddGaussianNoise(mean=0, std=args.noise_std)
    train_transform = transforms.Compose([
        base_transforms,
        gaussian_noise
    ])
    train_set = tf.load_trainset(args.data, train_transform, train=True, path=args.data_dir)  # trainset.num_classes
    # train_set = tf.corrupt_labels(args.corrupt)(train_set, args.lcr, args.lcs)
    from corrupt import corrupt_labels_uniform
    # save originals & corrupt
    train_set = corrupt_labels_uniform(train_set, args.lcr, args.lcs)

    test_transforms = tf.load_transforms(args.test_transform)
    test_set = tf.load_trainset(args.data, test_transforms, train=False, path=args.data_dir)  # trainset.num_classes
    # train_set = tf.corrupt_labels(args.corrupt)(train_set, args.lcr, args.lcs)

    # Initialize DataLoader for batching and loading the training data
    train_loader = DataLoader(train_set, batch_size=args.bs, drop_last=True, num_workers=args.n_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, drop_last=True, num_workers=args.n_workers, shuffle=False)

    return train_loader, train_set.num_classes, test_loader


def set_model(args):
    # Prepare for Training Section
    # Load a pretrained model if specified, otherwise initialize a new model with the specified architecture
    if args.pretrain_dir is not None:
        model, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
        utils.update_params(args.model_dir, args.pretrain_dir)
    else:
        model = tf.load_architectures(args.arch, args.fd, args.n_class)

    # Set up the loss function, optimizer, and learning rate scheduler for the training
    criterion = RoughLearning(gam1=args.gam1, gam2=args.gam2, eps=args.eps)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # Save initial training parameters for later reference
    utils.save_params(args.model_dir, vars(args))

    return model, criterion


def train(train_loader, model, criterion, optimiser, epoch, args):
    model.train()

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    sum_grad_norm = 0.0
    batch_count = 0

    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        batch_imgs, batch_labels = data_tuple

        data_time.update(time.time() - end)  # update the data loading time

        # if torch.cuda.is_available():
        #     batch_imgs = batch_imgs.cuda(non_blocking=True)
        #     batch_labels = batch_labels.cuda(non_blocking=True)

        features = model(batch_imgs)
        total_loss_rough, loss_rough, loss_precise, total_loss_precise = criterion(features, batch_labels)
        losses.update(total_loss_rough.item(), args.bs)

        optimiser.zero_grad()
        total_loss_rough.backward()

        ############ 检查梯度
        # Compute gradient norm for this batch
        batch_grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                batch_grad_norm_sq += p.grad.data.norm().item() ** 2
        batch_grad_norm = batch_grad_norm_sq ** 0.5
        sum_grad_norm += batch_grad_norm
        batch_count += 1

        optimiser.step()

        batch_time.update(time.time() - end)
        end = time.time()

        current_lr = optimiser.param_groups[0]['lr']
        if (idx + 1) % args.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'lr {lr:.5f}\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                lr=current_lr,
                loss=losses
            )
            print(to_print)
            sys.stdout.flush()

        utils.save_state(args.model_dir, epoch, idx, total_loss_rough.item(), *loss_rough, *loss_precise,
                         total_loss_precise.item())

    avg_grad_norm = sum_grad_norm / batch_count if batch_count > 0 else 0.0

    return avg_grad_norm


def set_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    # Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    return optimizer


def main():
    utils.set_seed(42)
    args = parse_option()

    train_loader, num_classes, test_loader = set_loader(args)
    args.n_class = num_classes
    model, criterion = set_model(args)
    optimizer = set_optimizer(args, model)
    if args.optimizer == 'SGD':
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400, 500], gamma=0.1)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epo, eta_min=1e-6)

    # Previous parameters for diff
    prev_params = None

    start_epoch = 1
    if len(args.resume):
        ckpt_state = torch.load(args.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {args.resume}!")

    for epoch in range(start_epoch, args.epo + 1):
        avg_grad_norm = train(train_loader, model, criterion, optimizer, epoch, args)
        # Adjust the learning rate based on the scheduler
        scheduler.step(epoch)
        # Save a checkpoint after each epoch


        # Save model parameter diff
        current_params = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        if prev_params is None:
            diff = 0.0
        else:
            total_diff_sq = 0.0
            for name in current_params:
                d = current_params[name] - prev_params[name]
                total_diff_sq += torch.norm(d).item() ** 2
            diff = total_diff_sq ** 0.5

        prev_params = current_params
        utils.save_parameter_flow(args.model_dir, epoch, avg_grad_norm, diff)

        model.eval()
        if epoch % args.save_measure_freq == 0 or epoch == 1:
            evaluate_and_save_metrics(epoch, model, train_loader, test_loader, args)

        if epoch % args.save_model_freq == 0:
            # utils.save_ckpt(args.model_dir, model, epoch)
            save_file = os.path.join(args.model_dir, 'checkpoints',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

        save_file = os.path.join(args.model_dir, 'checkpoints', 'curr_epoch.pth')
        save_model(model, optimizer, args, epoch, save_file)

    model.eval()
    evaluate_and_save_metrics(args.epo, model, train_loader, test_loader, args)

    print("Training complete.\n\n\n")  # Signal the end of training


if __name__ == '__main__':
    main()
