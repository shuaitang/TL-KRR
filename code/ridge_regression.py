import numpy as np
from abc import ABC
from collections import OrderedDict
import scipy, scipy.linalg
from sklearn.metrics import r2_score

class RidgeRegression(ABC):

    def __init__(self, style='c', alpha=1.0):
        self.alpha = alpha
        self.style = style

    def biasing(self, X):
        n, d = X.shape
        X = np.concatenate([X, np.ones((n, 1))], axis=-1)
        return X

    def fit(self, X, y):

        X, y = np.float32(X), np.float32(y)
        self.mean_vec = X.mean(axis=0, keepdims=True)
        X = self.preprocessing(X)

        if self.style == 'c':
            cov = X.T @ X
            XTy = X.T @ y
            d = cov.shape[0]
            self.base = scipy.linalg.solve(cov + self.alpha / d * np.trace(cov) * np.identity(d), XTy, assume_a="pos")
        elif self.style == 'k':
            kernel = X @ X.T
            d = kernel.shape[0]
            self.base = X.T @ scipy.linalg.solve(kernel + self.alpha / d * np.trace(kernel) * np.identity(d), y, assume_a="pos")


    def preprocessing(self, X):
        X = np.float32(X)
        X -= self.mean_vec
        X = self.biasing(X)
        return X


    def predict(self, X):
        X = self.preprocessing(X)
        output = X @ self.base

        return output


    def score(self, X, y):

        predicted_targets = self.predict(X)
        r2 = r2_score(y, predicted_targets)
        return r2


    def get_params(self, *args, **kwargs):
        return OrderedDict({"alpha": self.alpha, "style": self.style})


    def set_params(self, **params):

        self.alpha = params["alpha"] if "alpha" in params else ...
        self.style = params["style"] if "style" in params else ...

        return self
