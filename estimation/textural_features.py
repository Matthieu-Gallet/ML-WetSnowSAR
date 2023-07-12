from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import multiscale_basic_features, graycomatrix
from utils import scale_image  # > import utils
from tqdm import tqdm
import numpy as np


class Multi_Features_GLCM(BaseEstimator, TransformerMixin):
    def __init__(self, gray_param={}) -> None:
        super().__init__()
        self.gray_param = gray_param

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.vectorize_ = []
        if len(X.shape) == 3:
            n_canal = 0
        elif len(X.shape) == 4:
            n_canal = X.shape[-1]
        else:
            raise ValueError("X must be a 3D or 4D array")
        for i in tqdm(range(X.shape[0])):
            vec = []
            for j in range(n_canal):
                img = X[i, :, :, j]
                if np.max(img) > 1:
                    img = scale_image(img)
                img_proc = np.uint8((self.gray_param["levels"] - 1) * img)
                gcm = graycomatrix(img_proc, **self.gray_param)
                vec.append(
                    gcm.reshape(gcm.shape[0], gcm.shape[1], gcm.shape[2] * gcm.shape[3])
                )
            vec = np.concatenate(vec, axis=-1)
            self.vectorize_.append(vec)
        return np.array(self.vectorize_)


class Multi_Features_Basic(BaseEstimator, TransformerMixin):
    def __init__(self, features_param={}) -> None:
        super().__init__()
        self.features_param = features_param

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.vectorize_ = []
        if len(X.shape) == 3:
            n_canal = 0
        elif len(X.shape) == 4:
            n_canal = X.shape[-1]
        else:
            raise ValueError("X must be a 3D or 4D array")
        for i in tqdm(range(X.shape[0])):
            vec = []
            for j in range(n_canal):
                msf = multiscale_basic_features(X[i, :, :, j], **self.features_param)
                vec.append(msf.reshape(msf.shape[0] * msf.shape[1], msf.shape[2]))
            vec = np.concatenate(vec, axis=-1)
            self.vectorize_.append(vec)
        return np.array(self.vectorize_)
