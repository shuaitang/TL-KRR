import numpy as np
from abc import ABC
from scipy.optimize import minimize

class LearningKernelAlignment(ABC):

    def __init__(self):

        r"""
        Python implementation of the 'alignf' to find 
        the optimal convex combination of linear kernels.
        
        Reference:
        Cortes, C., Mohri, M., & Rostamizadeh, A. (2012). 
        Algorithms for Learning Kernels Based on Centered Alignment. 
        Journal of Machine Learning Research, 13, 795-828.
        """
        ...

    def compute_Ma(self, vec, y):

        r"""
        Compute the matrix M which stores pairwise alignment scores, 
        and a vector a which stores alignment scores between individual layers and the target 

        Parameters
        ---------
        vec : (n_layers) list
            list of feature vectors 
        y   : (n_samples, n_classes) numpy array
            onehot-encoded target vectors

        Returns
        --------
        M : (n_layers, n_layers) numpy array
            M_ij refers to the alignment score between two sets of feature vectors from layer i and j
        a : (n_layers,) numpy array
            a_i refers to the alignment score between feature vectors from layer i and target y 

        """

        a = np.array([((vec[iii].T @ y) ** 2.).sum() for iii in range(len(vec))])

        M = np.zeros((len(vec), len(vec)))
        for iii in range(M.shape[0]):
            for jjj in range(iii, M.shape[0]):
                temp = vec[iii].T @ vec[jjj]
                M[iii, jjj] = (temp ** 2.).sum()
            M[iii, iii] /= 2

        M = M + M.T
        return M, a


    def gradientdescent(self, M, a, init_mu=None):

        r"""
        Solve the optimisation problem presented in Proposition 9 in Cortes et al., JMLR2012
        LaTex: \min_{v \geq 0} v^\top M v - 2 v^\top a 

        Parameters
        ---------
        M : (n_layers, n_layers) numpy array
            M_ij refers to the alignment score between two sets of feature vectors from layer i and j
        a : (n_layers,) numpy array
            a_i refers to the alignment score between feature vectors from layer i and target y
        init_mu (Optional) : (n_layers,) numpy array 
            initialisation vector for solving the optimisation problem

        Returns
        --------
        mu : (n_layers, ) numpy array
            L2 normalised vector that gives the optimal convex combination of layers 

        """

        if init_mu == None:
            init_mu = np.ones_like(a)

        f = lambda v: v.reshape(1, -1) @ M @ v.reshape(-1, 1) - 2 * v.reshape(1, -1) @ a
        bnds = tuple([(0, None) for i in range(len(a))])
        res = minimize(f, init_mu, bounds=bnds)

        mu = res.x.clip(0.)
        mu = mu / np.linalg.norm(mu)

        return mu


    def compute_alignment(self, feat_vecs, onehot_targets):

        r"""
        Compute the optimal convex combination of layers given feature vectors and targets

        Parameters
        ---------
        feat_vecs      : (n_layers) list
            list of low-rank feature vectors from individual layers
        onehot_targets : (n_layers, n_classes) numpy array
            onehot-encoded target vectors

        Returns
        --------
        None

        """

        M, a = self.compute_Ma(feat_vecs, onehot_targets)
        self.mu = self.gradientdescent(M, a)
