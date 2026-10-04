"""Exp 8 - re-tune LightGBM regularisation on the cleaned training data (k=4 filter).

Sprint 2 chose min_child_samples=20 while corrupted rows were still in training and
validation. With them removed, a smaller leaf size may let rare packs get their own rules.

Usage:  exp8_retune.py <seed> <mcs>:<lambda> [<mcs>:<lambda> ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from src import data, experiment as E
from src.models import BinaryRelevanceLGBM

DROP_K = 4


def main(seed, *grid):
    seed = int(seed)
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    X, Xv = data.features(d, 'raw'), d['X_train'][va]
    rules = lambda Y: data.apply_additive_rules(Y, Xv, d['crm_cols'], d['bil_cols'])
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    Xu, Yu, _ = E.dedup_configs(X[keep], d['Y_cons'][keep])
    sfx = '' if seed == 42 else f'-s{seed}'
    for spec in grid:
        mcs, lam = spec.split(':')
        params = dict(min_child_samples=int(mcs), reg_lambda=float(lam))
        with E.Timer() as t:
            P = BinaryRelevanceLGBM(params).fit(Xu, Yu).predict_proba(Xv)
        exp_id = f'E8-mcs{mcs}-l{lam}{sfx}'
        np.save(os.path.join(data.DATA_DIR, 'cache', f'proba_{exp_id}.npy'), P.astype(np.float16))
        E.log(exp_id, 'raw+consensus (corrupted rows dropped, k=4)', 'BR LightGBM re-tuned on clean data',
              {**params, 'drop_k': DROP_K, 'split_seed': seed},
              E.score(d, rules((P >= .5).astype(np.uint8))), t.s, notes=f'{len(Xu):,} unique train configs')


if __name__ == '__main__':
    main(*sys.argv[1:])
