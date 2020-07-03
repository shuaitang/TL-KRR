import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch
import torch.nn as nn

import numpy as np
from abc import ABC

class LowrankFeats(ABC):

    r"""
    """

    def __init__(self, model, loader, projection_matrices, sketched_matrices, imgsize, device, freq_print):

        r"""
        Initialise variables
        """
        self.model = model
        self.loader = loader
        self.projection_matrices = projection_matrices
        self.sketched_matrices = sketched_matrices

        self.imgsize = imgsize
        self.device = device
        self.freq_print = freq_print

        self.lowrank_feats = {key: [] for key in self.projection_matrices}
        self.targets = []


    def project_feats(self, feats, layer_id):

        batchsize  = feats.size(0)
        feats = feats.data.view(batchsize, -1).type(torch.cuda.FloatTensor)

        if self.projection_matrices[layer_id] is not None:
            sketched_feats = self.sketched_matrices[layer_id].type(torch.cuda.FloatTensor)
            projection_matrix = self.projection_matrices[layer_id].type(torch.cuda.FloatTensor)

            temp = feats @ sketched_feats.T
            del sketched_feats
            temp = temp @ projection_matrix
            del projection_matrix
            torch.cuda.empty_cache()

            temp = temp.cpu().float()
        else:
            temp = feats.cpu().float()

        self.lowrank_feats[layer_id].append(temp)


    def forward_with_layerwise_hooks(self, input_features):

        r"""
        """
        batchsize = input_features.size(0)

        layer_id = 0
        self.project_feats(input_features, layer_id)

        out = self.model.conv1(input_features)

        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        layer_id += 1
        self.project_feats(out, layer_id)

        # The residual blocks in a ResNet model are grouped into four stages
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for mod in layer:
                out = mod(out)
                layer_id += 1
                self.project_feats(out, layer_id)

        out = out.mean(dim=(2,3))
        layer_id += 1
        self.project_feats(out, layer_id)


    def compute_projections(self):

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

                self.targets.append(target.cpu())

                if batch_idx % self.freq_print == 0:
                    print("finished {:d}/{:d}".format(batch_idx, len(self.loader)))


    def compute_lowrank_feats(self):

        r"""
        """

        self.compute_projections()

        for layer_id in self.lowrank_feats:
            self.lowrank_feats[layer_id] = torch.cat(self.lowrank_feats[layer_id], dim=0).float().numpy()

        self.targets = torch.cat(self.targets, dim=0).numpy()
        del self.projection_matrices, self.sketched_matrices
        torch.cuda.empty_cache()
