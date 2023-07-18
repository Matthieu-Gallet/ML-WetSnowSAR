import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import os

from files_management import load_h5


def random_shuffle(X, y, rng=-1):
    """Shuffle randomly the dataset

    Parameters
    ----------
    X : numpy array
        dataset of images

    y : numpy array
        dataset of labels

    rng : int, optional
        Random seed, by default -1, must be a np.random.default_rng() object

    Returns
    -------
    numpy array
        shuffled dataset of images

    numpy array
        shuffled dataset of labels

    """
    if rng == -1:
        rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y


def balance_dataset(X, Y, shuffle=False):
    """Balance the dataset by taking the minimum number of samples per class (under-sampling)

    Parameters
    ----------
    X : numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands)

    Y : numpy array
        dataset of labels in string, shape (n_samples,)

    shuffle : bool, optional
        Shuffle the dataset, by default False

    Returns
    -------
    numpy array
        balanced dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        balanced dataset of labels in string, shape (n_samples,)
    """
    if shuffle:
        X, Y = random_shuffle(X, Y)
    cat, counts = np.unique(Y, return_counts=True)
    min_count = np.min(counts)
    X_bal = []
    Y_bal = []
    for category in cat:
        idx = np.where(Y == category)[0]
        idx = idx[:min_count]
        X_bal.append(X[idx])
        Y_bal.append(Y[idx])
    X_bal = np.concatenate(X_bal)
    Y_bal = np.concatenate(Y_bal)
    return X_bal, Y_bal


class BFold:
    """Balanced Fold cross-validator, only valid for binary classification.

    Split dataset into k consecutive folds (without shuffling by default),
    ensuring that each fold has the same number of samples from each class.
    It allows to have B sub-datasets balanced, and have a complete view of the
    data.

    Parameters
    ----------
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, default=None
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.

    Attributes
    ----------
    n_splits : int
        Returns the number of splitting iterations in the cross-validator.

    """

    def __init__(self, shuffle=False, random_state=42):
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_state)

    def get_n_splits(self, X, y, groups=None):
        argmin_minor = np.argmin(np.unique(y, return_counts=True)[1])
        minority_numb = np.unique(y, return_counts=True)[1][argmin_minor]
        majority_numb = np.max(np.unique(y, return_counts=True)[1])
        self.n_splits = int(majority_numb / minority_numb)
        return self.n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training set.

        Parameters
        ----------
        X : numpy array
            dataset of images

        y : numpy array
            dataset of labels        test : numpy array
            The testing set indices for that split.
        ------
        train : numpy array
            The training set indices for that split.
        """
        argmin_minor = np.argmin(np.unique(y, return_counts=True)[1])
        minority_numb = np.unique(y, return_counts=True)[1][argmin_minor]
        majority_numb = np.max(np.unique(y, return_counts=True)[1])
        minority_class = np.unique(y, return_counts=True)[0][argmin_minor]
        ratio = majority_numb / minority_numb
        if majority_numb % minority_numb == 0:
            self.n_splits = int(ratio)
        else:
            self.n_splits = int(ratio) + 1

        idx_minority = np.where(y == minority_class)[0]
        idx_majority = np.where(y != minority_class)[0]

        if self.shuffle:
            self.rng.shuffle(idx_majority)

        for i in range(self.n_splits):
            start = i * minority_numb
            end = (i + 1) * minority_numb
            if i == self.n_splits - 1:
                end = len(idx_majority)
                miss = minority_numb - (end - start)
                idx_majority_balanced = np.concatenate(
                    [idx_majority[start:end], idx_majority[:miss]]
                )
            else:
                idx_majority_balanced = idx_majority[start:end]

            idx_train = np.concatenate([idx_minority, idx_majority_balanced])
            self.rng.shuffle(idx_train)
            yield idx_train


def parser_pipeline(dict_parameter, idx):
    """Parse a dictionary to create a pipeline of estimators
    The dictionary must have the following structure::
    {
        "import": [
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.decomposition import PCA",
            "from sklearn.svm import SVC",
        ],
        "pipeline": [
            [
                ["StandardScaler", {"with_mean": False, "with_std": False}],
                ["PCA", {"n_components": 0.95}],
                ["SVC", {"kernel": "rbf", "C": 10, "gamma": 0.01}],
            ]
        ],
    }

    Parameters
    ----------
    dict_parameter : dict
        Dictionary containing the pipeline

    idx : int
        Index of the pipeline to use in case of multiple pipelines
        analysis

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline of estimators
    """

    for import_lib in dict_parameter["import"]:
        exec(import_lib)
    pipe = dict_parameter["pipeline"][idx]

    step = []
    for i in range(len(pipe)):
        name_methode = pipe[i][0]
        estim = locals()[name_methode]()

        if len(pipe[i]) > 1:
            [
                [
                    setattr(estim, param, pipe[i][g][param])
                    for param in pipe[i][g].keys()
                ]
                for g in range(1, len(pipe[i]))
            ]
        step.append((name_methode, estim))
    return Pipeline(step, verbose=True, memory=".cache")


def load_train(i_path, bands_max, balanced, shffle=True, encode=True):
    """Load a hdf5 file containing the training dataset

    Parameters
    ----------
    i_path : str
        Path to the hdf5 file with name "data_train.h5"

    bands_max : list
        List of bands to keep in the dataset

    balanced : bool
        If True, the dataset is balanced

    shffle : bool, optional
        If True, the dataset is shuffled (rng 42), by default True

    encode : bool, optional
        If True, the labels are encoded, by default True

    Returns
    -------
    numpy array
        Dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        Dataset of labels in string, shape (n_samples,)

    sklearn.preprocessing.LabelEncoder
        Encoder used to encode the labels
    """

    X_train, Y_train = load_h5(os.path.join(i_path, "data_train.h5"))
    if bands_max != -1:
        X_train = X_train[:, :, :, bands_max]
    if balanced:
        X_train, Y_train = balance_dataset(X_train, Y_train, shuffle=shffle)
    if encode:
        encoder = LabelEncoder()
        encoder.fit(Y_train)
        Y_train = encoder.transform(Y_train)
    else:
        encoder = None
    return X_train, Y_train, encoder


def load_test(i_path, bands_max, balanced, shffle=True, encoder=None):
    """Load a hdf5 file containing the testing dataset

    Parameters
    ----------
    i_path : str
        Path to the hdf5 file with name "data_test.h5"

    bands_max : list
        List of bands to keep in the dataset

    balanced : bool
        If True, the dataset is balanced

    shffle : bool, optional
        If True, the dataset is shuffled (rng 42), by default True

    encoder : sklearn.preprocessing.LabelEncoder, optional
        Encoder used to encode the labels, by default None
        The encoder must be fitted before ie using the `load_train` function

    Returns
    -------
    numpy array
        Dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        Dataset of labels in string, shape (n_samples,)
    """

    X_test, Y_test = load_h5(os.path.join(i_path, "data_test.h5"))
    X_test = X_test[:, :, :, bands_max]
    if balanced:
        X_test, Y_test = balance_dataset(X_test, Y_test, shuffle=shffle)
    if encoder is not None:
        Y_test = encoder.transform(Y_test)
    return X_test, Y_test
