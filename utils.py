import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from enum import Enum
import numpy as np
import scipy.linalg
from scipy._lib._util import check_random_state, rng_integers

class DatasetsNames(Enum):
    cifar10     = lambda x: {"train" : True if x == "train" else False}
    cifar100    = lambda x: {"train" : True if x == "train" else False}
    cub200      = lambda x: {"train" : True if x == "train" else False}
    kuzushiji49 = lambda x: {"train" : True if x == "train" else False}

    svhn     = lambda x: {"split" : x}
    stl10    = lambda x: {"split" : x}


@torch.jit.script
def competitive_learning(SA, feats, lr):
    # type: (torch.Tensor, torch.Tensor, float) -> torch.Tensor
    bsize = feats.size(0)

    for i in range(bsize):
        neighbour = torch.argmin((feats[i:i+1] - SA).norm(dim=1, p=2.), dim=0)
        SA[neighbour] += lr * (feats[i] - SA[neighbour])

    return SA



def pca(data_mat, proj_mat=None, portion=0.995):

    r"""
    Principal Component Analysis through Singular Value Decomposition

    Parameters
    ------
    data_mat : (n_samples, d_samples) numpy array
        Input data matrix for SVD

    proj_mat : (d_samples, d_projection) numpy array
        Projection matrix for dimensionality reduction

    portion : float
        Portion of spectrum to remain 

    Return 
    ------
    data_mat : (n_samples, d_samples) numpy array
        Projected data matrix
    
    proj_mat : (d_samples, d_projection) numpy array
        Projection matrix found by PCA

    """

    if proj_mat is None:
        _, s, v = scipy.linalg.svd(data_mat, full_matrices=False, compute_uv=True)
        eigvalues = s ** 2.
        nnz = ((np.cumsum(eigvalues) / eigvalues.sum()) <= portion).sum()
        proj_mat = (v.T)[:,:nnz]
    data_mat = data_mat @ proj_mat

    return data_mat, proj_mat


def sjlt_mat(k, d, T, seed=None):

    r"""
    Sparse Johnson-Lindenstrauss Transformation via a stack of CountSketch methods

    Parameters
    ------
    k : int
        Projection dimension

    d : int
        Dimension of input data

    T : int
        Number of hash tables to deploy

    seed : int
        random seed

    Return 
    ------
    sjlt : (k, d) PyTorch Sparse Tensor

    """

    rng = check_random_state(seed)
    cols = rng.choice(k, d * T, replace=True)
    rows = np.vstack([np.arange(d) for t in range(T)]).transpose().reshape(-1)
    cols, rows = list(cols), list(rows)
    signs = list((rng.randn(T * d) > 0.) * 2. - 1.)
    indices = torch.LongTensor([rows, cols])
    signs = torch.FloatTensor(signs)

    sjlt = torch.sparse.FloatTensor(indices, signs, torch.Size([d, k])).transpose(1,0).div_(T ** 0.5)
    
    return sjlt



def load_model(device, modelname, pretrained=True):

    r"""
    """

    model = getattr(models, modelname)(pretrained=pretrained).to(device)

    return model


def load_dataset(name, split, batchsize, datapath, imgsize):

    r"""
    """

    # standard preprocessing steps for imagenet models

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor(),
        normalize,
    ])

    # transform = transform_train if train else transform_test
    # for our purpose, we only consider the testing case.
    transform = transform_test

    if name == "kuzushiji49":
        from kuzushiji49 import KUZUSHIJI49
        func_ = KUZUSHIJI49
    elif name == "cub200":
        from cub200 import CUB200
        func_ = CUB200
    else:
        func_ = getattr(torchvision.datasets, name.upper())

    kws = {
            "root": datapath,
            "download": True,
            "transform": transform,
    }

    kws = {**kws, **getattr(DatasetsNames, name)(split)}

    dataset = func_(**kws)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    return loader
