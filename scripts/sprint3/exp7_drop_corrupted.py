"""Exp 7 - retrain the champion without suspected-corrupted training rows.

Suspected corrupted = not test-like (invalid categorical one-hot / value '0') OR carrying
>= k rare CRM packs (pack frequency < 0.2% of train rows). k in {2, 3, 4} is a sensitivity
check. Detection uses CRM features only. Every run is scored on all / test-like /
clean-like validation rows (clean-like is always defined with the fixed k=3 so the
metric does not move with the experiment).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from src import data, experiment as E
from src.models import BinaryRelevanceLGBM

CHAMPION = dict(min_child_samples=20, reg_lambda=1.0)
CACHE = os.path.join(data.DATA_DIR, 'cache')


def main(seed=42, ks='2,3,4'):
    seed = int(seed)
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    X, Xv = data.features(d, 'raw'), d['X_train'][va]
    rules = lambda Y: data.apply_additive_rules(Y, Xv, d['crm_cols'], d['bil_cols'])
    sfx = '' if seed == 42 else f'-s{seed}'

    for ref, label in [('E5-reg-mcs20', 'Sprint 2 champion'), ('E6-clean', 'E6-clean (non-test-like dropped)')]:
        path = os.path.join(CACHE, f'proba_{ref}{sfx}.npy')
        if os.path.exists(path):
            P = np.load(path).astype(np.float32)
            E.log(f'E7-rescore-{ref}{sfx}', 'raw+consensus', f'{label}, re-scored', {'split_seed': seed},
                  E.score(d, rules((P >= .5).astype(np.uint8))), 0, notes='adds clean-like metric')

    for k in [int(x) for x in ks.split(',')]:
        keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < k)]
        Xu, Yu, _ = E.dedup_configs(X[keep], d['Y_cons'][keep])
        with E.Timer() as t:
            P = BinaryRelevanceLGBM(CHAMPION).fit(Xu, Yu).predict_proba(Xv)
        exp_id = f'E7-drop-k{k}{sfx}'
        np.save(os.path.join(CACHE, f'proba_{exp_id}.npy'), P.astype(np.float16))
        E.log(exp_id, 'raw+consensus (corrupted rows dropped)',
              f'Champion trained without suspected-corrupted rows (>= {k} rare packs or not test-like)',
              {**CHAMPION, 'k': k, 'split_seed': seed, 'dropped_train_rows': int(len(tr) - len(keep)),
               'dropped_share': round(1 - len(keep) / len(tr), 4)},
              E.score(d, rules((P >= .5).astype(np.uint8))), t.s, notes=f'{len(Xu):,} unique train configs')


if __name__ == '__main__':
    main(*sys.argv[1:])
