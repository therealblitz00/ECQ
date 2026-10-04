"""Exp 10 - Step 5: catch light corruption, repair instead of drop, capacity check.

All runs start from the Step 4 champion: k=4 corrupted-row filter, per-label LightGBM with
min_child_samples=5, reg_lambda=1, threshold 0.5, 42 additive rules.

  cl      confident learning: 3-fold out-of-fold predictions on the cleaned training
          configurations; drop configurations whose own labels contradict a very confident
          out-of-fold prediction (>= m such cells), retrain.
  repair  add back the rows dropped for having >= 4 rare packs (categoricals valid), with
          their rare CRM and rare BIL packs stripped.
  cap     more capacity: 200 trees / 31 leaves.

Usage:  exp10_step5.py <seed> <mode> [<mode> ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.model_selection import KFold

from src import data, experiment as E, validation as V
from src.models import BinaryRelevanceLGBM

DROP_K = 4
BASE = dict(min_child_samples=5, reg_lambda=1.0)
CACHE = os.path.join(data.DATA_DIR, 'cache')
LO, HI = 0.02, 0.98


def run(d, exp_id, Xu, Yu, params, notes, extra=None):
    va = d['va']
    Xv = d['X_train'][va]
    with E.Timer() as t:
        P = BinaryRelevanceLGBM(params).fit(Xu, Yu).predict_proba(Xv)
    np.save(os.path.join(CACHE, f'proba_{exp_id}.npy'), P.astype(np.float16))
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), Xv, d['crm_cols'], d['bil_cols'])
    E.log(exp_id, 'raw+consensus (corrupted rows dropped, k=4)', notes,
          {**params, 'drop_k': DROP_K, 'split_seed': d['seed'], **(extra or {})},
          E.score(d, pred), t.s, notes=f'{len(Xu):,} unique train configs')


def confident_learning(d, Xu, Yu, sfx):
    oof = np.zeros(Yu.shape, np.float32)
    for k, (a, b) in enumerate(KFold(3, shuffle=True, random_state=0).split(Xu)):
        oof[b] = BinaryRelevanceLGBM(BASE).fit(Xu[a], Yu[a]).predict_proba(Xu[b])
        print(f'  out-of-fold fold {k + 1}/3 done', flush=True)
    miss = (Yu == 1) & (oof < LO)      # item present, model sure it should not be
    extra = (Yu == 0) & (oof > HI)     # item absent, model sure it should be
    n = (miss | extra).sum(1)
    print(f'  confident disagreements per config: {np.bincount(np.minimum(n, 5)).tolist()} '
          f'(cells: {int(miss.sum())} unexpected items, {int(extra.sum())} missing items)')
    for m in (1, 2):
        keep = n < m
        run(d, f'E10-cl-m{m}{sfx}', Xu[keep], Yu[keep], BASE,
            'Step 4 champion + confident-learning cleaning',
            {'cl_min_disagreements': m, 'cl_dropped_configs': int((~keep).sum()),
             'cl_dropped_share': round(float((~keep).mean()), 4)})


def repair(d, Xu, Yu, sfx):
    tr, X, Y = d['tr'], d['X_train'], d['Y_cons']
    rows = tr[d['test_like'][tr] & (d['rare_count'][tr] >= DROP_K)]
    cp = np.array([c.endswith('_PACK') for c in d['crm_cols']])
    bp = np.array([c.endswith('_PACK') for c in d['bil_cols']])
    rare_c = cp & (X.mean(0) < V.RARE_FREQ)
    rare_b = bp & (Y.mean(0) < V.RARE_FREQ)
    Xr, Yr = X[rows].copy(), Y[rows].copy()
    Xr[:, rare_c] = 0
    Yr[:, rare_b] = 0
    Xa, Ya, _ = E.dedup_configs(np.vstack([Xu, Xr]), np.vstack([Yu, Yr]))
    print(f'  repaired {len(rows):,} rows -> {len(Xa) - len(Xu):,} new unique configs')
    run(d, f'E10-repair{sfx}', Xa, Ya, BASE, 'Step 4 champion + repaired corrupted rows',
        {'repaired_rows': int(len(rows))})


def main(seed, *modes):
    seed = int(seed)
    d = E.setup(seed=seed)
    d['seed'] = seed
    tr = d['tr']
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    Xu, Yu, _ = E.dedup_configs(d['X_train'][keep], d['Y_cons'][keep])
    sfx = '' if seed == 42 else f'-s{seed}'
    if 'cl' in modes:
        confident_learning(d, Xu, Yu, sfx)
    if 'repair' in modes:
        repair(d, Xu, Yu, sfx)
    if 'cap' in modes:
        run(d, f'E10-cap{sfx}', Xu, Yu, {**BASE, 'n_estimators': 200, 'num_leaves': 31},
            'Step 4 champion with more capacity')


if __name__ == '__main__':
    main(*sys.argv[1:])
