import numpy as np
from sklearn.metrics import roc_curve


def BAROC(y_true, y_est):
    """Estimate the best threshold given the best accuracy based on the ROC curve

    Parameters
    ----------
    y_true : np.array
        True labels

    y_est : np.array
        Estimated labels

    Returns
    -------
    float
        Best accuracy

    float
        Best threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_est)
    num_pos_class = y_true[y_true > 0].sum()
    num_neg_class = len(y_true) - num_pos_class
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)
    best_threshold = thresholds[np.argmax(acc)]
    return np.amax(acc), best_threshold


def FRCROC(y_true, y_est, rate=0.05):
    """Estimate the best threshold given a fixed FPR (False Positive Rate) based on the ROC curve

    Parameters
    ----------
    y_true : np.array
        True labels

    y_est : np.array
        Estimated labels

    rate : float, optional
        Fixed FPR, by default 0.05

    Returns
    -------
    float
        accuracy at the fixed FPR

    float
        threshold at the fixed FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_est)
    num_pos_class = y_true[y_true > 0].sum()
    num_neg_class = len(y_true) - num_pos_class
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)
    threshold_fpc = thresholds[fpr >= rate][0]
    accu_fpc = acc[fpr >= rate][0]
    return accu_fpc, threshold_fpc
