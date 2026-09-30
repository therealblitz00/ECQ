import numpy as np
import pandas as pd


def exact_match_ratio(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).all(axis=1).mean())


def hamming_loss(y_true, y_pred):
    return float((np.asarray(y_true) != np.asarray(y_pred)).mean())


def row_errors(y_true, y_pred):
    return (np.asarray(y_true) != np.asarray(y_pred)).sum(axis=1)


def evaluate(y_true, y_pred):
    err = row_errors(y_true, y_pred)
    return {
        'emr': float((err == 0).mean()),
        'hamming_loss': float(err.mean() / np.asarray(y_true).shape[1]),
        'share_dist1': float((err == 1).mean()),
        'share_dist2': float((err == 2).mean()),
        'share_dist3plus': float((err >= 3).mean()),
    }


def multilabel_report(y_true, y_pred, proba=None):
    """EMR, F1 (micro/macro/samples), precision/recall, AUCs and the cell-level confusion matrix.

    Macro F1 is averaged over labels that have at least one positive in y_true.
    """
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    active = y_true.sum(axis=0) > 0
    per_label_f1 = f1_score(y_true[:, active], y_pred[:, active], average=None, zero_division=0)
    summary = {
        'EMR': exact_match_ratio(y_true, y_pred),
        'bit accuracy': 1 - hamming_loss(y_true, y_pred),
        'Hamming loss': hamming_loss(y_true, y_pred),
        'F1 micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'F1 samples': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'F1 macro': float(per_label_f1.mean()),
        'median per-label F1': float(np.median(per_label_f1)),
        'precision micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
    }
    if proba is not None:
        summary['ROC-AUC micro'] = roc_auc_score(y_true.ravel(), np.asarray(proba).ravel())
        summary['PR-AUC micro'] = average_precision_score(y_true.ravel(), np.asarray(proba).ravel())
    t, p = y_true == 1, y_pred == 1
    confusion = pd.DataFrame([[int((t & p).sum()), int((t & ~p).sum())],
                              [int((~t & p).sum()), int((~t & ~p).sum())]],
                             index=['actual 1', 'actual 0'], columns=['predicted 1', 'predicted 0'])
    return pd.Series(summary), confusion, per_label_f1


def top_offending_columns(y_true, y_pred, cols, top_n=20):
    """Columns responsible for rows that are exactly one bit away from a perfect match."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    diff = y_true != y_pred
    near = diff.sum(axis=1) == 1
    counts = diff[near].sum(axis=0)
    fn = (diff & (y_true == 1))[near].sum(axis=0)
    out = pd.DataFrame({
        'bil_column': cols,
        'n_rows_sole_error': counts,
        'missed_positive': fn,
        'false_positive': counts - fn,
        'col_error_rate_all_rows': diff.mean(axis=0),
    })
    out = out[out['n_rows_sole_error'] > 0].sort_values('n_rows_sole_error', ascending=False)
    out['share_of_dist1_rows'] = out['n_rows_sole_error'] / max(int(near.sum()), 1)
    return out.head(top_n).reset_index(drop=True)
