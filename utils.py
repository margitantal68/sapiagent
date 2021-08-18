import numpy as np
import pandas as pd
from sklearn import metrics


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
