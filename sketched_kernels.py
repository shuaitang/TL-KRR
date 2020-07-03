import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch
import torch.nn as nn

import numpy as np
from abc import ABC
import scipy, scipy.linalg

@torch.jit.script
def hashing(SA, row_maps, sign, data, T):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int) -> torch.Tensor
    num_hashes = row_maps.size(0)

    for i in range(num_hashes):
        SA[row_maps[i]] += data[int(i / T)] * sign[i]
    return SA


class SketchedKernels(ABC):

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

    def __init__(self, model, loader, imgsize, device, M, T, freq_print):

        r"""
        Initialise variables
        """
        self.model = model
        self.loader = loader
        self.imgsize = imgsize
        self.device = device
        self.M = M
        self.T = T
        self.freq_print = freq_print

        self.n_samples = 0
        self.sketched_matrices = {}
        self.projection_matrices = {}


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
        feats = feats.data.view(batchsize, -1).type(torch.FloatTensor)
        batchsize, dim = feats.size()

        if dim <= self.M:
            self.sketched_matrices[layer_id] = None
        else:
            if layer_id not in self.sketched_matrices:
                self.sketched_matrices[layer_id] = torch.zeros(self.M, dim).float()

            _row_map = torch.from_numpy(np.random.choice(self.M, batchsize * self.T, replace=True)).long()
            _sign = torch.randn(batchsize * self.T).sign()
            self.sketched_matrices[layer_id] = hashing(self.sketched_matrices[layer_id], _row_map, _sign, feats, self.T)

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
                self.forward_with_layerwise_hooks(data)

                batchsize = data.size(dim=0)
                self.n_samples += batchsize

                if batch_idx % self.freq_print == 0:
                    print("finished {:d}/{:d}".format(batch_idx, len(self.loader)))


    def compute_sketched_kernels(self):

        r"""
        The main function to compute both the subsampled dataset and the projection matrix for the Nyström method
        """

        self.compute_sketched_mat()

        for layer_id in range(len(self.sketched_matrices)):
            if self.sketched_matrices[layer_id] == None:
                self.projection_matrices[layer_id] = None
            else:

                # Nyström
                mat = self.sketched_matrices[layer_id].type(torch.cuda.FloatTensor)
                temp = mat @ mat.T
                temp = np.float32(temp.cpu().numpy())

                pinv = scipy.linalg.pinvh(temp)
                w, v = scipy.linalg.eigh(pinv)
                eigvecs, eigvals = np.float32(v[:,::-1]), np.float32(w[::-1].clip(0.) ** 0.5)
                nnz = sum(eigvals != 0)

                projection = eigvecs[:,:nnz] * eigvals[:nnz].reshape(1, -1)
                self.projection_matrices[layer_id] = torch.from_numpy(projection)

            torch.cuda.empty_cache()
