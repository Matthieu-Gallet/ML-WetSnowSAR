from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import numpy as np


def optimal_bins(data):
    """Compute the optimal number of bins for a histogram
    using the Freedman-Diaconis rule

    Parameters
    ----------
    data : np.array
        data to compute the optimal number of bins

    Returns
    -------
    int
        optimal number of bins
    """
    n = len(data)
    min_x = np.min(data)
    max_x = np.max(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    hbins = int(np.ceil((1 / (2 * (q3 - q1))) * (n ** (1 / 3)) * (max_x - min_x)))
    return hbins


def moments_features(data, type_stats="mean"):
    """Compute the moments of a list of array or a matrix on the first axis

    Parameters
    ----------
    data : np.array
        data to compute the moments
    type_stats : str, optional
        type of moments to compute. If "std" compute in addition
        the standard deviation, if "skew" compute in addition
        the skewness, if "kurtosis" compute in addition
        the kurtosis, by default "mean"

    Returns
    -------
    np.array
        vector of moments
    """

    vec = []
    for j in range(data.shape[1]):
        vec.append(data[:, j].mean())
        if type_stats == "std":
            vec.append(data[:, j].std())
        elif type_stats == "skew":
            vec.append(data[:, j].std())
            vec.append(skew(data[:, j]))
        elif type_stats == "kurtosis":
            vec.append(data[:, j].std())
            vec.append(skew(data[:, j]))
            vec.append(kurtosis(data[:, j]))
    return np.array(vec).flatten()


def calcul_hist(data, n_bands, bin):
    """Compute the histogram of a list of array or a matrix on the first axis

    Parameters
    ----------
    data : np.array
        data to compute the histogram
    n_bands : int
        number of bands
    bin : int
        number of bins

    Returns
    -------
    np.array
        vector of histogram of each band between 0 and 1
    """

    vec = []
    for j in range(n_bands):
        hst = np.histogram(data[:, j], bins=bin, range=(0, 1))[0]
        vec.append(hst)
    return np.array(vec).flatten()


class Stats_SAR(BaseEstimator, TransformerMixin):
    """Tranform a tensor or a list of matrix into a vector array of the
    selected moments of each band.
    """

    def __init__(self, type_stats="mean") -> None:
        super().__init__()
        self.type_stats = type_stats

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.vectorize_ = []
        self._size_data = X.shape
        self.n_bands = X.shape[-1]
        X_t = X.reshape(
            self._size_data[0], self._size_data[1] * self._size_data[2], self.n_bands
        )
        self.vectorize_ = Parallel(n_jobs=-2)(
            delayed(moments_features)(x, self.type_stats) for x in tqdm(X_t)
        )
        return np.array(self.vectorize_)


class Hist_SAR(BaseEstimator, TransformerMixin):
    """Tranform a tensor or a list of matrix into a vector array of the
    histogram of each band.
    """

    def __init__(self, nbins=16) -> None:
        super().__init__()
        self.nbins = nbins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.vectorize_ = []
        self._size_data = X.shape
        self.n_bands = X.shape[-1]
        X_t = X.reshape(
            self._size_data[0], self._size_data[1] * self._size_data[2], self.n_bands
        )
        if self.nbins == -1:
            self.nbins = optimal_bins(X_t)

        self.vectorize_ = Parallel(n_jobs=-2)(
            delayed(calcul_hist)(x, self.n_bands, self.nbins) for x in tqdm(X_t)
        )
        return np.array(self.vectorize_)
