import os
os.environ["OMP_NUM_THREADS"] = "4"

import torch
import torch.nn as nn

import numpy as np
from abc import ABC

class LowrankFeats(ABC):

    r"""
    For each layer, given a subsampled dataset and a projection matrix, 
    the module computes a low-rank approximation of feature vectors from that layer.
    """

    def __init__(self, model, loader, projection_matrices, mean_vectors, sketched_matrices, hashing_matrices, args):

        r"""
        Initialise variables

        Parameters
        ---------
        model    : PyTorch model instance
            A ResNet model 
            (We only tested on ResNet18 and ResNet34 pretrained on the ImageNet dataset, 
            but it should work for all pretrained ResNet models)

        loader : PyTorch DataLoader instance
            A data loader for the dataset of a downstream task

        projection_matrices : (n_layers) dict
            A dictionary of projection matrices obtained through the Nyström method for dimensionality reduction.
            The number of elements is equal to the number of layers one wants to accumulate.

        mean_vectors : (n_layers) dict
            A dictionary of mean vectors for each layer.

        sketched_matrices : (n_layers) dict
            A dictionary of data summaries obtained through CWT (Clarkson-Woodruff Transformation), 
            of which each serves as subsampled data points for Nyström.
        
        imgsize : int
            The size of the input images

        device : str
            GPU or CPU. GPU is recommended for forwarding samples through a neural network as it is faster.
            (We actually kinda ignore this argument in this class. :shrug:)

        freq_print : int
            The printing frequency.
            

        """
        self.model = model
        self.loader = loader
        self.projection_matrices = projection_matrices
        self.mean_vectors = mean_vectors
        self.sketched_matrices = sketched_matrices
        self.hashing_matrices = hashing_matrices

        self.imgsize = args.imgsize
        self.device = args.device
        self.freq_print = args.freq_print

        self.lowrank_feats = {key: [] for key in self.projection_matrices}
        self.targets = []


    def project_feats(self, feats, layer_id):

        r"""
        Project a batch of feature vectors into the low-dimensional space 
        obtained through the Nyström method.

        Parameters
        ---------
        feats    : (batchsize, n_channels, height, width) PyTorch Tensor
            A batch of feature vectors 
        layer_id : int
            The index of the layer from which the feature vectors are produced
        
        Returns
        --------
        None

        """


        batchsize  = feats.size(0)
        feats = feats.data.view(batchsize, -1).type(torch.cuda.FloatTensor)

        if self.hashing_matrices[layer_id] is not None:
            feats = torch.sparse.mm(self.hashing_matrices[layer_id], feats.T).T

        mean_vector = self.mean_vectors[layer_id].type(torch.cuda.FloatTensor)
        feats -= mean_vector

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
        Forward function to collect feature vectors at multiple layers

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
        Compute low-rank approximations of features at multiple layers for all data samples

        Parameters
        ---------
        None
        
        Returns
        --------
        None

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
        Main function to call to compute low-rank approximations

        Parameters
        ---------
        None

        Returns
        --------
        None

        """

        self.compute_projections()

        for layer_id in self.lowrank_feats:
            self.lowrank_feats[layer_id] = torch.cat(self.lowrank_feats[layer_id], dim=0).float().numpy()

        self.targets = torch.cat(self.targets, dim=0).numpy()
        del self.projection_matrices, self.sketched_matrices
        torch.cuda.empty_cache()
