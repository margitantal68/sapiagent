import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# positive - normal
# negative - outlier
def compute_AUC_EER(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=1)
    # plot_ROC(fpr, tpr)
    roc_auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)
    return roc_auc, EER, fpr, tpr
