from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.covariance as sk_cov
from tqdm import tqdm
import numpy as np


def is_pos_def(x):
    """Check if a matrix is positive definite

    Parameters
    ----------
    x : float
        matrice

    Returns
    -------
    bool
        SPD ou non
    """
    return np.all(np.linalg.eigvals(x) > 0)


class Covariance_MultiScale(BaseEstimator, TransformerMixin):
    """Classe for the calculation of covariance matrix from a dataset of variable size"""

    def __init__(self, methode="LedoitWolf", **param) -> None:
        super().__init__()
        self.methode = methode
        self._select_method_cov(**param)  # sk_cov.LedoitWolf
        self.covariance_ = np.array([])
        self.spd = []
        self.spd_check = True

    def _select_method_cov(self, **param):
        """Select the method for the calculation of covariance matrix and its parameters"""
        meth = getattr(sk_cov, self.methode)
        if type(meth) == type:
            if param:
                try:
                    self.meth_cov = meth(**param)
                except:
                    raise TypeError("Unknown parameters")
            else:
                self.meth_cov = meth()
        else:
            raise TypeError("Not a class from sklearn.covariance")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cov_tens = []
        self._size_data = len(X)
        for i in tqdm(range(self._size_data)):
            matrix = X[i]
            if len(matrix.shape) > 2:
                matrix = matrix.reshape(
                    matrix.shape[0] * matrix.shape[1], matrix.shape[2]
                )
            fit_covar = self.meth_cov.fit(matrix)
            self.spd.append(is_pos_def(fit_covar.covariance_))
            cov_tens.append(fit_covar.covariance_)
        self.covariance_ = np.array(cov_tens)
        return self.covariance_


class Vector_CovarianceEuclid(BaseEstimator, TransformerMixin):
    """Transform a tensor or a list of covariance matrix into a vector array.
    Each vector is the extraction of the triangular matrix of its associated matrix"""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        vec = []
        self._size_data = len(X)
        for i in tqdm(range(self._size_data)):
            matrix = X[i]
            vec.append(matrix[np.triu_indices(n=matrix.shape[0])])
        self.vectorize_ = np.array(vec)
        return self.vectorize_


class Vector_Euclid(BaseEstimator, TransformerMixin):
    """Transform a tensor or a list of matrix into a vector array"""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        vec = []
        self._size_data = len(X)
        for i in tqdm(range(self._size_data)):
            matrix = X[i]
            vec.append(matrix.flatten())
        self.vectorize_ = np.array(vec)
        return self.vectorize_
