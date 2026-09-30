"""Reusable multi-label model wrappers."""
import numpy as np
import lightgbm as lgb
from joblib import Parallel, delayed

LGB_DEFAULT = dict(n_estimators=100, num_leaves=15, learning_rate=0.1,
                   min_child_samples=5, verbose=-1)


class BinaryRelevanceLGBM:
    """One LightGBM per label, trained in parallel threads (LightGBM releases the GIL)."""

    def __init__(self, params=None, n_jobs=8, threads_per_model=2):
        self.params = {**LGB_DEFAULT, **(params or {})}
        self.n_jobs, self.tpm = n_jobs, threads_per_model

    def _fit_one(self, X, y, w):
        if y.min() == y.max():
            return float(y[0])
        m = lgb.LGBMClassifier(n_jobs=self.tpm, **self.params)
        m.fit(X, y, sample_weight=w)
        return m

    def fit(self, X, Y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        self.models_ = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._fit_one)(X, Y[:, j], sample_weight) for j in range(Y.shape[1]))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)

        def one(m):
            return np.full(len(X), m, np.float32) if isinstance(m, float) else m.predict_proba(X)[:, 1]
        cols = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(one)(m) for m in self.models_)
        return np.column_stack(cols).astype(np.float32)


def cooccurrence_parents(Y, order, n_parents=30):
    """For each label, the upstream labels (earlier in `order`) it co-occurs with most (Jaccard)."""
    Yf = Y.astype(np.float32)
    inter = Yf.T @ Yf
    n = np.diag(inter)
    jac = inter / np.maximum(n[:, None] + n[None, :] - inter, 1)
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order))
    parents = []
    for j in range(Y.shape[1]):
        up = order[:pos[j]]
        parents.append(up[np.argsort(-jac[j, up])[:n_parents]] if len(up) else up)
    return parents


class CoOccurrenceChainLGBM:
    """Classifier chain where each label sees X plus its most co-occurring upstream labels.

    Training uses true upstream labels (teacher forcing), so all fits are independent and run
    in parallel; prediction walks the chain in order, feeding predicted upstream labels forward.
    """

    def __init__(self, params=None, n_parents=30, n_jobs=8, threads_per_model=2):
        self.params = {**LGB_DEFAULT, **(params or {})}
        self.n_parents, self.n_jobs, self.tpm = n_parents, n_jobs, threads_per_model

    def _fit_one(self, X, Y, j):
        y = Y[:, j]
        if y.min() == y.max():
            return float(y[0])
        Xj = np.hstack([X, Y[:, self.parents_[j]].astype(np.float32)])
        return lgb.LGBMClassifier(n_jobs=self.tpm, **self.params).fit(Xj, y)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        self.order_ = np.argsort(-Y.mean(0), kind='stable')
        self.parents_ = cooccurrence_parents(Y, self.order_, self.n_parents)
        self.models_ = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self._fit_one)(X, Y, j) for j in range(Y.shape[1]))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        P = np.zeros((len(X), len(self.models_)), np.float32)
        B = np.zeros_like(P)
        for j in self.order_:
            m = self.models_[j]
            if isinstance(m, float):
                P[:, j] = m
            else:
                P[:, j] = m.predict_proba(np.hstack([X, B[:, self.parents_[j]]]))[:, 1]
            B[:, j] = P[:, j] >= 0.5
        return P
