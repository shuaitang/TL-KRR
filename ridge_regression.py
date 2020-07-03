import numpy as np
from abc import ABC
from collections import OrderedDict
import scipy, scipy.linalg
from sklearn.metrics import r2_score

class RidgeRegression(ABC):

    def __init__(self, style='c', alpha=1.0):

        r"""
        Python implementation of Ridge Regression.
        Quarantine gave me the strength and time to implement it by myself. ;)

        Parameters
        ---------
        style : str
            It indicates whether to solve the problem in its primal or dual view.
            'c' means primal and 'k' means dual. 
        alpha : float
            Strength of L2 regularisation in Ridge Regression 

        """

        self.alpha = alpha 
        self.style = style 
       

    def biasing(self, X):

        r"""
        Adding an extra dimension with 1 to each of the input feature vector

        Parameters
        ---------
        X : (n_samples, d_sample) numpy array
            input feature vectors in a matrix
        
        Returns
        --------
        X : (n_samples, d_sample+1) numpy array        
            input feature vectors in a matrix with an extra column filled with 1s

        """

        n, d = X.shape
        X = np.concatenate([X, np.ones((n, 1))], axis=-1)
        return X

    def fit(self, X, y):

        r"""
        Fit a Ridge Regression model in primal or dual view

        Parameters
        ---------
        X : (n_samples, d_sample) numpy array
            input feature vectors in a matrix
        y:  (n_samples, d_target) numpy array
            target vectors in a matrix
        
        Returns
        --------
        None

        """

        X, y = np.float32(X), np.float32(y)

        # Compute the mean vector on the training set 
        # and use it to zero-centre test set later
        self.mean_vec = X.mean(axis=0, keepdims=True)

        # Preprocessing
        X = self.preprocessing(X)


        if self.style == 'c':
            # Solve the optimisation problem in the primal view
            cov = X.T @ X
            XTy = X.T @ y
            d = cov.shape[0]
            self.base = scipy.linalg.solve(cov + self.alpha / d * np.trace(cov) * np.identity(d), XTy, assume_a="pos")
        elif self.style == 'k':
            # Solve the optimisation problem in the dual view
            kernel = X @ X.T
            d = kernel.shape[0]
            self.base = X.T @ scipy.linalg.solve(kernel + self.alpha / d * np.trace(kernel) * np.identity(d), y, assume_a="pos")


    def preprocessing(self, X):
        
        r"""
        Zero-centre the input feature vectors, 
        and an extra column to it as the bias

        Parameters
        ---------
        X : (n_samples, d_sample) numpy array
            input feature vectors in a matrix
        
        Returns
        --------
        None

        """

        X = np.float32(X)
        X -= self.mean_vec
        X = self.biasing(X)
        return X


    def predict(self, X):

        r"""
        Make predictions

        Parameters
        ---------
        X : (n_samples, d_sample) numpy array
            input feature vectors in a matrix
        
        Returns
        --------
        output : (n_samples, d_target) numpy array
            predicted targets

        """

        X = self.preprocessing(X)
        output = X @ self.base

        return output


    def score(self, X, y):

        r"""
        R^2 measure for the goodness-of-fit of a Ridge Regression model

        Parameters
        ---------
        X : (n_samples, d_sample) numpy array
            input feature vectors in a matrix
        y : (n_samples, d_target) numpy array
            target vectors in a matrix
        
        Returns
        --------
        r2 : float
            R^2 score

        """

        predicted_targets = self.predict(X)
        r2 = r2_score(y, predicted_targets)
        return r2


    def get_params(self, *args, **kwargs):
        return OrderedDict({"alpha": self.alpha, "style": self.style})


    def set_params(self, **params):

        self.alpha = params["alpha"] if "alpha" in params else ...
        self.style = params["style"] if "style" in params else ...

        return self
