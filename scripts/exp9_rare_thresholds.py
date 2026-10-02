"""Exp 9 - lower decision threshold for rare BIL columns only (Step 4.2).

The remaining errors are mostly missed billing items in rows with rare packs. A rare BIL
column = active in < rare_freq of the (cleaned) training-fold rows. Its threshold t is
lowered; every other column keeps 0.5. t is CHOSEN on the seed-42 split and only CHECKED on
the independent seed-7 split, so the reported seed-7 gain is not fitted to its own rows.

Usage:  exp9_rare_thresholds.py <probs id, e.g. E8-mcs5-l1>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E, metrics as M

DROP_K = 4
GRID = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
RARE_FREQS = [0.002, 0.005]


def evaluate(seed, probs_id):
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    prev = d['Y_cons'][keep].mean(0)
    sfx = '' if seed == 42 else f'-s{seed}'
    P = np.load(os.path.join(data.DATA_DIR, 'cache', f'proba_{probs_id}{sfx}.npy')).astype(np.float32)
    Xv, cl = d['X_train'][va], d['clean_like'][va]
    out = {}
    for rf in RARE_FREQS:
        rare_col = prev < rf
        for t in GRID:
            thr = np.where(rare_col, t, 0.5)
            pred = data.apply_additive_rules((P >= thr).astype(np.uint8), Xv, d['crm_cols'], d['bil_cols'])
            out[(rf, t)] = M.exact_match_ratio(d['Y_cons'][va][cl], pred[cl])
    return d, P, out, prev


def main(probs_id='E8-mcs5-l1'):
    d42, _, res42, _ = evaluate(42, probs_id)
    _, _, res7, _ = evaluate(7, probs_id)
    print(f'clean-like EMR, probabilities from {probs_id}')
    print(f'{"rare_freq":>9} {"t":>5} {"seed 42":>9} {"seed 7":>9}')
    for k in res42:
        print(f'{k[0]:>9.3f} {k[1]:>5.2f} {res42[k]:>9.4%} {res7[k]:>9.4%}')
    best = max(res42, key=res42.get)
    base = (best[0], 0.5)
    print(f'\nchosen on seed 42: rare_freq={best[0]}, t={best[1]} -> seed 42 {res42[best]:.4%} '
          f'(vs {res42[base]:.4%} at t=0.5) | CHECK on seed 7: {res7[best]:.4%} (vs {res7[base]:.4%})')


if __name__ == '__main__':
    main(*sys.argv[1:])
