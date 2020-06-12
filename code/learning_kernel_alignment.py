import numpy as np
from abc import ABC
from scipy.optimize import minimize

class LearningKernelAlignment(ABC):

    def __init__(self):

        r"""
        Initialise variables
        """
        ...

    def compute_Ma(self, vec, y):

        a = [((vec[iii].T @ y) ** 2.).sum() for iii in range(len(vec))]

        M = np.zeros((len(vec), len(vec)))
        for iii in range(M.shape[0]):
            for jjj in range(iii, M.shape[0]):
                temp = vec[iii].T @ vec[jjj]
                M[iii, jjj] = (temp ** 2.).sum()
            M[iii, iii] /= 2

        M = M + M.T
        return M, a


    def gradientdescent(self, M, a, init_mu=None):
        if init_mu == None:
            init_mu = np.ones_like(a)

        f = lambda v: v.reshape(1, -1) @ M @ v.reshape(-1, 1) - 2 * v.reshape(1, -1) @ a
        bnds = tuple([(0, None) for i in range(len(a))])
        res = minimize(f, init_mu, bounds=bnds)

        mu = res.x.clip(0.)
        mu = mu / np.linalg.norm(mu)

        return mu


    def compute_alignment(self, feat_vecs, onehot_targets):
        M, a = self.compute_Ma(feat_vecs, onehot_targets)
        self.mu = self.gradientdescent(M, a)
