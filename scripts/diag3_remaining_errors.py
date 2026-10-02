"""Diagnostic 3 - what is left wrong on clean-like validation rows (the population that
tracks the test set)? Read-only.

Run:  .venv/Scripts/python scripts/diag3_remaining_errors.py E7-drop-k3 > data/cache/diag3.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import data, experiment as E, metrics as M


def section(title):
    print('\n' + '=' * 100 + f'\n{title}\n' + '=' * 100)


def flip_rate(d):
    """Per BIL column: share of repeated CRM configs (where it appears) in which it is not constant."""
    Yr, g = d['Y_raw'], d['groups']
    rep = np.bincount(g)[g] >= 2
    df = pd.DataFrame(Yr[rep])
    df['g'] = g[rep]
    s = df.groupby('g').sum().to_numpy()
    n = df.groupby('g').size().to_numpy()[:, None]
    return ((s > 0) & (s < n)).sum(0) / np.maximum((s > 0).sum(0), 1)


def main(exp_id='E7-drop-k3'):
    d = E.setup(seed=42)
    va = d['va']
    X, Y = d['X_train'][va], d['Y_cons'][va]
    bil = np.array(d['bil_cols'])
    P = np.load(os.path.join(data.DATA_DIR, 'cache', f'proba_{exp_id}.npy')).astype(np.float32)
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), X, d['crm_cols'], d['bil_cols'])
    cl = d['clean_like'][va]
    X, Y, P, pred, rare = X[cl], Y[cl], P[cl], pred[cl], d['rare_count'][va][cl]
    diff = pred != Y
    err = diff.sum(1)

    section(f'Remaining errors of {exp_id} on clean-like validation rows ({cl.sum():,} rows)')
    print(f'EMR {np.mean(err == 0):.4%} | wrong rows {np.mean(err > 0):.2%}')
    print('wrong bits per row:', pd.Series(np.minimum(err, 6)).value_counts().sort_index()
          .rename(index={6: '6+'}).to_dict())
    fp, fn = int((diff & (pred == 1)).sum()), int((diff & (pred == 0)).sum())
    print(f'wrong cells: {fp + fn:,} = {fp:,} false positives + {fn:,} false negatives')

    section('Which kind of row / column the errors sit in')
    catm = np.array([not c.endswith('_PACK') for c in bil])
    flips = flip_rate(d) >= 0.5
    wrong = err > 0
    groups = {
        'row has 1-2 rare CRM packs (legitimate rare products?)': wrong & (rare > 0),
        'row has no rare CRM pack': wrong & (rare == 0),
        'only categorical columns wrong': wrong & ~diff[:, ~catm].any(1),
        'only "flipping" columns wrong (inconsistent for identical CRM)': wrong & ~diff[:, ~flips].any(1),
    }
    print(f'rows with 1-2 rare CRM packs: {np.mean(rare > 0):.2%} of clean-like rows, EMR {np.mean(err[rare > 0] == 0):.2%} '
          f'| rows with none: EMR {np.mean(err[rare == 0] == 0):.2%}')
    for k, m in groups.items():
        print(f'  {k:66s} {m.sum():5,} rows = {m.sum() / max(wrong.sum(), 1):6.1%} of wrong rows')

    section('How confident are the wrong bits? (room for threshold tuning)')
    pw = P[diff]
    print(f'wrong cells with probability in [0.2, 0.8]: {np.mean((pw > .2) & (pw < .8)):.1%}; '
          f'in [0.35, 0.65]: {np.mean((pw > .35) & (pw < .65)):.1%}')
    one = err == 1
    pone = P[one][diff[one]]
    print(f'1-bit rows: {one.sum():,}; probability of their wrong bit: median {np.median(pone):.3f}, '
          f'share within 0.15 of the threshold {np.mean(np.abs(pone - .5) < .15):.1%}')
    print(f'upper bound if every 1-bit row were fixed: EMR {np.mean(err <= 1):.4%}')

    section('Columns that alone break an otherwise perfect clean-like row')
    top = M.top_offending_columns(Y, pred, list(bil), top_n=15)
    top['flips'] = top['bil_column'].map(dict(zip(bil, flips)))
    print(top.round(4).to_string(index=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
