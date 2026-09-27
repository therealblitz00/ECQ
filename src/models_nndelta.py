"""Neighbour + pack-delta predictor (untried model family, built from a Sprint 2 finding).

Finding: for ~70 % of validation rows the nearest training configuration differs by
ONE CRM pack, and copying its billing is wrong by exactly one BIL bit ~59 % of the time.
So: copy the nearest configuration's billing, then apply the billing change that
this pack is known to cause.

fit   : find every pair of training configurations that differ by exactly one pack j
        (A and A+j). For each pack j, count how often each BIL bit switches on or off
        when j is added. A bit is in on[j] / off[j] if it switches in >= min_rate of
        the pairs and there are >= min_pairs pairs.
predict: take the nearest training configurations (Hamming, all ties at the minimum
        distance, at most max_nb), transform each neighbour's billing by the deltas of
        the packs that are added / removed, and average -> probability.
Packs without enough pairs contribute no delta (the neighbour's bit is kept).
"""
from __future__ import annotations

import numpy as np


class NeighbourDelta:
    def __init__(self, min_pairs=2, min_rate=0.5, max_nb=5, chunk=512):
        self.min_pairs, self.min_rate, self.max_nb, self.chunk = min_pairs, min_rate, max_nb, chunk

    def fit(self, X, Y):
        X = np.ascontiguousarray(X, np.uint8); Y = np.ascontiguousarray(Y, np.uint8)
        self.X_, self.Y_ = X, Y
        n, m = X.shape
        L = Y.shape[1]
        keys = np.packbits(X, axis=1)
        index = {k.tobytes(): i for i, k in enumerate(keys)}
        par, chi, pk = [], [], []
        for c in range(n):                       # child c = parent + pack j
            kb = keys[c].copy()
            for j in np.flatnonzero(X[c]):
                kb[j >> 3] ^= 0x80 >> (j & 7)
                p = index.get(kb.tobytes())
                kb[j >> 3] ^= 0x80 >> (j & 7)
                if p is not None:
                    par.append(p); chi.append(c); pk.append(j)
        par, chi, pk = map(np.asarray, (par, chi, pk))
        on = np.zeros((m, L), np.float32); off = np.zeros((m, L), np.float32)
        cnt = np.bincount(pk, minlength=m).astype(np.float32) if len(pk) else np.zeros(m, np.float32)
        if len(pk):
            Yp, Yc = Y[par].astype(np.float32), Y[chi].astype(np.float32)
            np.add.at(on, pk, Yc * (1 - Yp))
            np.add.at(off, pk, Yp * (1 - Yc))
        ok = cnt >= self.min_pairs
        with np.errstate(invalid="ignore", divide="ignore"):
            self.on_ = ((on / cnt[:, None]) >= self.min_rate) & ok[:, None]
            self.off_ = ((off / cnt[:, None]) >= self.min_rate) & ok[:, None]
        self.n_pairs_, self.packs_with_delta_ = len(pk), int(ok.sum())
        return self

    def _neighbours(self, Xq):
        A = self.X_.astype(np.float32)
        na = A.sum(1)
        for s in range(0, len(Xq), self.chunk):
            Q = Xq[s:s + self.chunk].astype(np.float32)
            D = Q.sum(1, keepdims=True) + na[None, :] - 2 * Q @ A.T
            k = min(self.max_nb, D.shape[1])
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            dd = np.take_along_axis(D, idx, 1)
            o = np.argsort(dd, 1)
            yield s, np.take_along_axis(idx, o, 1), np.take_along_axis(dd, o, 1)

    def predict_proba(self, X):
        X = np.asarray(X, np.uint8)
        out = np.zeros((len(X), self.Y_.shape[1]), np.float32)
        self.dist_ = np.zeros(len(X), np.int32)
        for s, idx, dd in self._neighbours(X):
            for r in range(len(idx)):
                x = X[s + r].astype(bool)
                sel = idx[r][dd[r] <= dd[r, 0] + 0.5]          # all ties at the minimum
                acc = np.zeros(self.Y_.shape[1], np.float32)
                for i in sel:
                    y = self.Y_[i].astype(bool).copy()
                    nb = self.X_[i].astype(bool)
                    for j in np.flatnonzero(x & ~nb):           # pack added
                        y = (y | self.on_[j]) & ~self.off_[j]
                    for j in np.flatnonzero(nb & ~x):           # pack removed
                        y = (y & ~self.on_[j]) | self.off_[j]
                    acc += y
                out[s + r] = acc / len(sel)
                self.dist_[s + r] = int(round(dd[r, 0]))
        return out
