import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch
import torch.nn as nn

import numpy as np
from abc import ABC
import scipy, scipy.linalg

from utils import *

class FrequentDirections(ABC):

    r"""
    For each layer, Clarkson-Woodruff Transformation (CWT) is applied to hash feature vectors to a small number of buckets,
    which are used as the subsampled data samples in the Nyström method. 

    Reference: 
    Clarkson, K.L., & Woodruff, D.P. (2013). 
    Low rank approximation and regression in input sparsity time. 
    The Annual ACM Symposium on Theory of Computing (STOC) 2013.

    Jagadeesan, M. (2019). 
    Understanding Sparse JL for Feature Hashing. 
    Neural Information Processing Systems (NeurIPS) 2019.
    """

    def __init__(self, model, loader, args):

        r"""
        Initialise variables
        """
        self.model = model
        self.loader = loader

        self.imgsize = args.imgsize
        self.device = args.device
        self.M = args.M
        self.T = args.T
        self.factor = args.factor
        self.feature_hashing = args.feature_hashing
        self.freq_print = args.freq_print

        self.max_feats = self.M * self.factor
        self.hashing_matrices = {}

        self.n_samples = 0
        # self.projected_matrices = {}
        self.projection_matrices = {}
        # self.mean_vectors = {}


    def cwt_sketching(self, feats, layer_id):

        r"""
        Compute CWT for a batch of feature vectors

        Parameters
        ---------
        feats : (batchsize, n_channels, height, width) PyTorch Tensor
            A batch of feature vectors 
        layer_id : int
            The index of the layer from which the feature vectors are produced

        Returns
        --------
        None

        """

        batchsize  = feats.size(0)
        feats = feats.data.view(batchsize, -1)
        batchsize, dim = feats.size()

        if dim > self.max_feats and self.feature_hashing:
            if layer_id not in self.hashing_matrices:
                self.hashing_matrices[layer_id] = sjlt_mat(self.max_feats, dim, self.T).to(self.device)
            feats = torch.sparse.mm(self.hashing_matrices[layer_id], feats.T).T
            dim = self.max_feats 
        else:
            self.hashing_matrices[layer_id] = None

        if dim > self.M:
            # if layer_id not in self.sketched_matrices:
                # self.sketched_matrices[layer_id] = torch.zeros(self.M, dim).float()
            # self.sketched_matrices[layer_id] += torch.sparse.mm(self.sjlt, feats).type(torch.FloatTensor)
            if layer_id not in self.projection_matrices:
                self.projection_matrices[layer_id] = {
                    "s": torch.zeros(dim, self.M),
                    "q": torch.qr(torch.randn(dim, self.M))[0],
                }
            q = self.projection_matrices[layer_id]["q"].to(self.device)
            s = self.projection_matrices[layer_id]["s"].to(self.device)

            temp = feats.T @ (feats @ q) / batchsize
            s += temp
            q, r = torch.qr(s)

            self.projection_matrices[layer_id]["s"] = s.cpu()
            self.projection_matrices[layer_id]["q"] = q.cpu()
        else:
            # self.sketched_matrices[layer_id] = None
            self.projection_matrices[layer_id] = None
        
        # if layer_id not in self.mean_vectors:
            # self.mean_vectors[layer_id] = 0.
        # self.mean_vectors[layer_id] += feats.sum(axis=0)

        del feats
        torch.cuda.empty_cache()


    def forward_with_layerwise_hooks(self, input_features):

        r"""
        Forward function of a ResNet model

        Parameters
        ---------
        input_features : (batchsize, 3, height, width) PyTorch Tensor
            A batch of input images

        Returns
        --------
        None

        """
        batchsize = input_features.size(0)

        layer_id = 0
        self.cwt_sketching(input_features, layer_id)

        out = self.model.conv1(input_features)

        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        layer_id += 1
        self.cwt_sketching(out, layer_id)

        # The residual blocks in a ResNet model are grouped into four stages
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for mod in layer:
                out = mod(out)
                layer_id += 1
                self.cwt_sketching(out, layer_id)

        out = out.mean(dim=(2,3))
        layer_id += 1
        self.cwt_sketching(out, layer_id)


    def compute_sketched_mat(self):

        r"""
        Forward data samples in a dataset to a pretrained neural network

        Parameters
        ---------
        None

        Returns
        --------
        None

        """

        self.model.eval()

        with torch.no_grad():
            # Iterate through the data loader in batches:
            for batch_idx, (data, target) in enumerate(self.loader):

                # load a batch of data samples
                data, target = data.to(self.device), target.to(self.device)

                batchsize = data.size(dim=0)
                self.n_samples += batchsize

                # sample a random projection matrix
                self.sjlt = sjlt_mat(self.M, batchsize, self.T).to(self.device)
                self.forward_with_layerwise_hooks(data)

                if batch_idx % self.freq_print == 0:
                    print("finished {:d}/{:d}".format(batch_idx, len(self.loader)))


    def compute_sketched_kernels(self):

        r"""
        The main function to compute both the subsampled dataset and the projection matrix for the Nyström method
        """

        self.compute_sketched_mat()

        for layer_id in self.projection_matrices:
            # self.mean_vectors[layer_id] /= self.n_samples

            if self.projection_matrices[layer_id] is not None:
                self.projection_matrices[layer_id] = self.projection_matrices[layer_id]["q"]
                # # Nyström
                # mat = self.sketched_matrices[layer_id].type(torch.cuda.FloatTensor)
                # mat -= self.mean_vectors[layer_id].type(torch.cuda.FloatTensor)
                # self.sketched_matrices[layer_id] = mat.type(torch.FloatTensor)

                # temp = mat @ mat.T
                # temp = np.float32(temp.cpu().numpy())

                # pinv = scipy.linalg.pinvh(temp)
                # w, v = scipy.linalg.eigh(pinv)
                # eigvecs, eigvals = np.float32(v[:,::-1]), np.float32(w[::-1].clip(0.) ** 0.5)
                # nnz = sum(eigvals != 0)

                # projection = eigvecs[:,:nnz] * eigvals[:nnz].reshape(1, -1)
                # self.projection_matrices[layer_id] = torch.from_numpy(projection)

            torch.cuda.empty_cache()
