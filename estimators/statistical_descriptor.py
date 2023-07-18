import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from tqdm import tqdm
import numpy as np

from utils.metrics import BAROC, FRCROC, BF1ROC


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


class Hist_SAR(BaseEstimator, ClassifierMixin):
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


class Nagler_WS(BaseEstimator, ClassifierMixin):
    """Compute the wet snow prediction using an learning adaptive threshold of type Nagler
    This is done by simply computing the mean of the ratio VV with a snow free reference
    """

    def __init__(self, bands=6, threshold_type="FPR", FPR_rate=0.05) -> None:
        super().__init__()
        self.bands = bands
        self.threshold_type = threshold_type
        self.FPR_rate = FPR_rate
        self.success = False

    def fit(self, X, y=None):
        self.y_train = y
        self.ypred_train = 1 - X.mean(axis=(1, 2))[:, self.bands]
        if self.threshold_type == "FPR":
            _, self.threshold = FRCROC(self.y_train, self.ypred_train, self.FPR_rate)
            self.success = True
        elif self.threshold_type == "BAROC":
            _, self.threshold = BAROC(self.y_train, self.ypred_train)
            self.success = True
        elif self.threshold_type == "BF1ROC":
            _, self.threshold = BF1ROC(self.y_train, self.ypred_train)
            self.success = True
        else:
            raise ValueError("Threshold type not implemented")

    def predict_proba(self, X):
        self.X_test = X
        self.y_proba = 1 - self.X_test.mean(axis=(1, 2))[:, self.bands]
        return np.vstack([1 - self.y_proba, self.y_proba]).T

    def predict(self, X):
        if self.success:
            self.predict_proba(X)
            self.y_pred = np.where(self.y_proba > self.threshold, 1, 0)
        else:
            raise ValueError("Model not fitted")
        return self.y_pred

    def score(self, X, y=None):
        if self.success:
            y = y.ravel()
            y = self.label_encoder.transform(y)
            if self.y_pred is None:
                self.predict(X)
            self.accuracy_ = accuracy_score(y, self.y_pred)
            self.f1_score_ = f1_score(y, self.y_pred)
            return self.accuracy_
        else:
            raise ValueError("Model not fitted")
