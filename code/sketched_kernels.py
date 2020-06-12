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



    def cwt_matrix(self, n_rows, n_cols, T):

        r"""
        """

        all_rows = []
        all_cols = []
        all_signs = []

        for t in range(T):

            chunk = int(n_rows / T)
            shift = int(t * chunk)

            rows = torch.randint(shift, shift+chunk, (1, n_cols))
            cols = torch.arange(n_cols).view(1,-1)
            signs = torch.randn(n_cols).sign().float()

            all_rows.append(rows)
            all_cols.append(cols)
            all_signs.append(signs)

        rows = torch.cat(all_rows, dim=1)
        cols = torch.cat(all_cols, dim=1)
        pos = torch.cat([rows.long(), cols.long()], dim=0)
        signs = torch.cat(all_signs, dim=0)
        cwt = torch.sparse.FloatTensor(pos, signs, torch.Size([n_rows, n_cols])).div( T ** 0.5 )
        return cwt


    def compute_sketched_mat(self):

        r"""
        """

        self.model.eval()
        total = 0

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
        """

        self.compute_sketched_mat()

        for layer_id in range(len(self.sketched_matrices)):
            if self.sketched_matrices[layer_id] == None:
                self.projection_matrices[layer_id] = None
            else:

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
