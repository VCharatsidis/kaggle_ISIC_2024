from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np


def score(targets, preds , min_tpr=0.80):

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(targets) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(preds)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    #     # Equivalent code that uses sklearn's roc_auc_score
    #     v_gt = abs(np.asarray(solution.values)-1)
    #     v_pred = np.array([1.0 - x for x in submission.values])
    #     max_fpr = abs(1-min_tpr)
    #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return (partial_auc)