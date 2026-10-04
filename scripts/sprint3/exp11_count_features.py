"""Exp 11 - Step 7: basket-size count features on top of the Sprint 3 champion.

A teammate's MCA test (+0.31 EMR, Sprint 2 set-up) had a component correlated +0.61 with the
number of packs per customer. This tests the simple version of that signal directly:
  npk       + number of CRM packs the customer has
  npk_rare  + number of CRM packs and number of rare CRM packs (< 0.2% of train rows)
Champion otherwise unchanged (k=4 corrupted-row filter, min_child_samples=5, reg_lambda=1,
threshold 0.5, 42 additive rules on the raw CRM columns).

Usage:  exp11_count_features.py <seed> npk npk_rare
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from src import data, experiment as E
from src.models import BinaryRelevanceLGBM

CHAMPION = dict(min_child_samples=5, reg_lambda=1.0)
DROP_K = 4


def main(seed, *variants):
    seed = int(seed)
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    X = d['X_train']
    pk = np.array([c.endswith('_PACK') for c in d['crm_cols']])
    extra = {'npk': [X[:, pk].sum(1)], 'npk_rare': [X[:, pk].sum(1), d['rare_count']]}
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    sfx = '' if seed == 42 else f'-s{seed}'
    for v in variants:
        Xa = np.hstack([X.astype(np.float32)] + [c.astype(np.float32)[:, None] for c in extra[v]])
        Xu, Yu, _ = E.dedup_configs(Xa[keep], d['Y_cons'][keep])
        with E.Timer() as t:
            P = BinaryRelevanceLGBM(CHAMPION).fit(Xu, Yu).predict_proba(Xa[va])
        exp_id = f'E11-{v}{sfx}'
        np.save(os.path.join(data.DATA_DIR, 'cache', f'proba_{exp_id}.npy'), P.astype(np.float16))
        pred = data.apply_additive_rules((P >= .5).astype(np.uint8), X[va], d['crm_cols'], d['bil_cols'])
        E.log(exp_id, 'raw+consensus (corrupted rows dropped, k=4)', f'Champion + count features ({v})',
              {**CHAMPION, 'drop_k': DROP_K, 'split_seed': seed, 'extra_features': v},
              E.score(d, pred), t.s, notes=f'{Xu.shape[1]} features, {len(Xu):,} unique train configs')


if __name__ == '__main__':
    main(*sys.argv[1:])
