import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os
import argparse
import numpy as np

from sketched_kernels import SketchedKernels
from lowrank_feats import LowrankFeats
from learning_kernel_alignment import LearningKernelAlignment
from ridge_regression import RidgeRegression
from utils import *
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    # Get arguments from the command line
    parser = argparse.ArgumentParser(description='PyTorch CWT sketching kernel matrices')

    parser.add_argument('--datapath', type=str,
                                help='absolute path to the dataset')
    parser.add_argument('--modelname', type=str,
                                help='model name')

    parser.add_argument('--seed', default=0, type=int,
                                help='random seed for sketching')
    parser.add_argument('--task', default='cifar10', type=str,
                                choices=['cifar10', 'cifar100', 'svhn', 'stl10', 'cub200', 'kuzushiji49'],
                                help='the name of the dataset, cifar10 or cifar100 or svhn or stl10')

    parser.add_argument('--bsize', default=800, type=int,
                                help='batch size for computing the kernel')

    parser.add_argument('--M', '--num-buckets-sketching', default=2048, type=int,
                                help='number of buckets in Sketching')
    parser.add_argument('--T', '--num-buckets-per-sample', default=4, type=int,
                                help='number of buckets each data sample is sketched to')

    parser.add_argument('--freq_print', default=10, type=int,
                                help='frequency for printing the progress')

    args = parser.parse_args()

    # Set the backend and the random seed for running our code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    # The size of images for training and testing ImageNet models
    imgsize = 224

    # Generate a dataloader that iteratively reads data
    # Load a model, either pretrained or not
    loader = {
        "train": load_dataset(args.task, "train", args.bsize, args.datapath, imgsize),
        "test":  load_dataset(args.task, "test",  args.bsize, args.datapath, imgsize)
    }
    net = load_model(device, args.modelname, pretrained=True)

    # # # # # # # # # # # # # # # #
    # Nystroem with CountSketch for low-rank approximation
    # # # # # # # # # # # # # # # #

    # Set the model to be in the evaluation mode. VERY IMPORTANT!
    net.eval()
    csm = SketchedKernels(net, loader["train"], imgsize, device, args.M, args.T, args.freq_print)
    csm.compute_sketched_kernels()

    lowrank_feats = {}
    targets = {}

    for split in ["train", "test"]:
        proj = LowrankFeats(net, loader[split], csm.projection_matrices, csm.sketched_matrices, imgsize, device, args.freq_print)
        proj.compute_lowrank_feats()
        lowrank_feats[split] = proj.lowrank_feats
        targets[split] = proj.targets
        del proj

    del csm, net, loader

    for layer_id in lowrank_feats["train"]:
        mean_vec = lowrank_feats["train"][layer_id].mean(axis=0, keepdims=True)
        lowrank_feats["train"][layer_id] -= mean_vec
        lowrank_feats["test"][layer_id]  -= mean_vec
        # TODO: normalisation
        factor = np.linalg.norm(lowrank_feats["train"][layer_id].T @ lowrank_feats["train"][layer_id])
        lowrank_feats["train"][layer_id] /= factor
        lowrank_feats["test"][layer_id] /= factor


    # # # # # # # # # # # # # # # #
    # Learning kernel alignment for finding a convex combination
    # # # # # # # # # # # # # # # #

    lka = LearningKernelAlignment()
    train_targets = targets["train"]
    train_onehot = np.zeros((train_targets.size, train_targets.max()+1))
    train_onehot[np.arange(train_targets.size),train_targets] = 1
    lka.compute_alignment(lowrank_feats["train"], train_onehot)
    mu = lka.mu
    print(["{:.02f} ".format(m) for m in mu])
    sorted_indices = np.argsort(mu)[::-1]

    train_features = []
    test_features  = []

    for index in sorted_indices:
        if mu[index] != 0:
            train_features.append((mu[index] ** 0.5) * lowrank_feats["train"][index])
            test_features.append( (mu[index] ** 0.5) * lowrank_feats["test"][index] )

    train_features = np.concatenate(train_features, axis=1)
    test_features  = np.concatenate(test_features,  axis=1)

    # # # # # # # # # # # # # # # #
    # Nystroem again!
    # # # # # # # # # # # # # # # #

    factor = np.max(np.linalg.norm(train_features, axis=1))
    train_features /= factor
    test_features /= factor
    dim = min(args.M * 2, train_features.shape[0])
    print("Nystroem dim is {}".format(dim))
    # gamma = 1./np.max(np.linalg.norm(train_features, axis=1))
    gamma = 1.
    approx = Nystroem(kernel='rbf', gamma=gamma, n_components=dim)
    approx.fit(train_features)
    train_features = approx.transform(train_features)
    test_features  = approx.transform(test_features)

    # # # # # # # # # # # # # # # #
    # Ridge regression with cross validation
    # # # # # # # # # # # # # # # #

    style = 'c' if dim < train_features.shape[0] else 'k'
    clf = GridSearchCV(RidgeRegression(),
                            {
                                'alpha':[10**i for i in range(-7, 0)],
                                'style':[style],
                            },
                        n_jobs=4)

    clf.fit(train_features, train_onehot)
    test_pred_ = clf.predict(test_features)
    # print(test_pred_)
    acc = sum(np.argmax(test_pred_, axis=-1) == targets["test"])  * 1.0 / len(targets["test"])
    print(acc)
