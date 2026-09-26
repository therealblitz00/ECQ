"""Exp 5 - hybrid post-processing on saved validation probabilities.

Combines BR (E3a2) and chain (E4-cc30) probabilities, the 42 Section 11 rules, a global
threshold, and k-NN routing for rows that contain a rare CRM pack.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E
from scripts.exp1_lookup_knn import knn_hamming


def main():
    d = E.setup()
    tr, va = d['tr'], d['va']
    Xv = d['X_train'][va]
    load = lambda k: np.load(os.path.join(data.DATA_DIR, 'cache', f'proba_{k}.npy')).astype(np.float32)
    P_br, P_cc = load('E3a2'), load('E4-cc30')
    rules = lambda Y: data.apply_additive_rules(Y, Xv, d['crm_cols'], d['bil_cols'])

    for name, P in [('br', P_br), ('cc', P_cc), ('avg', (P_br + P_cc) / 2)]:
        for thr in (0.5, 0.6):
            pred = rules((P >= thr).astype(np.uint8))
            E.log(f'E5-{name}-t{thr}', 'raw+consensus', f'{name} probs + rules + threshold',
                  {'thr': thr}, E.score(d, pred), 0)

    # k-NN routing for rows containing a CRM pack seen in < R training rows
    ftr = d['X_train'][tr].sum(0)
    Xu, Yu, cnt = E.dedup_configs(d['X_train'][tr], d['Y_cons'][tr])
    idx, dist = knn_hamming(Xu, Xv, k=15)
    w = (1.0 / (1.0 + dist)) * np.log1p(cnt[idx])
    knn = ((Yu[idx] * w[:, :, None]).sum(1) / w.sum(1, keepdims=True) >= 0.5).astype(np.uint8)
    base = rules(((P_br + P_cc) / 2 >= 0.6).astype(np.uint8))
    for R in (100, 200, 400):
        rare_row = (Xv[:, (ftr > 0) & (ftr < R)].sum(1) > 0)
        pred = np.where(rare_row[:, None], knn, base)
        E.log(f'E5-knnroute{R}', 'raw+consensus', 'avg+rules+t0.6, k-NN on rare-pack rows',
              {'R': R}, E.score(d, pred), 0, notes=f'{rare_row.mean():.1%} rows routed to k-NN')


if __name__ == '__main__':
    main()
