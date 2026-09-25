"""Exp 2 - Label Powerset: ceilings, and LP as a decoder over observed training BIL configs.

A multi-class model over ~54k distinct training configurations is not trainable, and the
top-K configurations cover <1% of validation rows (validation CRM configs are unseen). So
LP is evaluated as a constrained decoder: given per-label probabilities from a BR model,
pick the observed training configuration with the highest log-likelihood.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import data, experiment as E, validation as V


def lp_decode(P, C, chunk=1024, eps=1e-6):
    """Best observed config per row: argmax_c sum_j c_j log p_j + (1-c_j) log(1-p_j)."""
    lp, lq = np.log(np.clip(P, eps, 1)), np.log(np.clip(1 - P, eps, 1))
    Cf = C.astype(np.float32)
    best, best_ll = np.empty(len(P), np.int64), np.empty(len(P), np.float32)
    for s in range(0, len(P), chunk):
        a, b = lp[s:s + chunk], lq[s:s + chunk]
        ll = a @ Cf.T + b.sum(1, keepdims=True) - b @ Cf.T
        best[s:s + chunk] = ll.argmax(1)
        best_ll[s:s + chunk] = ll.max(1)
    return best, best_ll


def main(src='E3a2'):
    d = E.setup()
    tr, va, Yc = d['tr'], d['va'], d['Y_cons']

    ktr = pd.Series(V.row_keys(Yc[tr]))
    vc = ktr.value_counts()
    kva = V.row_keys(Yc[va])
    seen = set(vc.index)
    ceil_all = float(np.mean([k in seen for k in kva]))
    ceil_top = {K: float(np.mean([k in set(vc.index[:K]) for k in kva])) for K in (100, 1000, 5000)}
    print('LP EMR ceiling (val config seen in train):', ceil_all, ' top-K:', ceil_top)

    P = np.load(os.path.join(data.DATA_DIR, 'cache', f'proba_{src}.npy')).astype(np.float32)
    first = pd.Series(np.arange(len(tr))).groupby(ktr.to_numpy()).first().to_numpy()
    C = Yc[tr][first]
    with E.Timer() as t:
        best, best_ll = lp_decode(P, C)
    snapped = C[best]
    E.log('E2-lp', 'raw+consensus', 'Label Powerset decoder (all seen configs)',
          {'probs_from': src, 'n_configs': len(C)}, E.score(d, snapped), t.s,
          notes=f'ceiling {ceil_all:.2%}; top-1000 ceiling {ceil_top[1000]:.2%}')

    free = (P >= 0.5).astype(np.uint8)
    eps = 1e-6
    free_ll = (np.log(np.clip(np.where(free == 1, P, 1 - P), eps, 1))).sum(1)
    for delta in (0.5, 1.0, 2.0, 4.0):
        use = (free_ll - best_ll) <= delta
        pred = np.where(use[:, None], snapped, free)
        E.log(f'E2-hyb{delta}', 'raw+consensus', 'BR + LP snap when within delta log-lik',
              {'probs_from': src, 'delta': delta}, E.score(d, pred), t.s,
              notes=f'snapped {use.mean():.1%} of rows')


if __name__ == '__main__':
    main(*sys.argv[1:])
