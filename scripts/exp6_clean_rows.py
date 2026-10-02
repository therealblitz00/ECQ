"""Exp 6 - test-like validation metric, and the champion retrained without non-test-like rows.

Non-test-like = CRM categorical blocks that are not a valid one-hot (2+ or 0 statuses/types,
no business line) or use the value '0'. ~1.5% of train rows, ~0.001% of test rows.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E
from src.models import BinaryRelevanceLGBM

CHAMPION = dict(min_child_samples=20, reg_lambda=1.0)


def main(seed=42):
    seed = int(seed)
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    X, Xv = data.features(d, 'raw'), d['X_train'][va]
    rules = lambda Y: data.apply_additive_rules(Y, Xv, d['crm_cols'], d['bil_cols'])
    sfx = '' if seed == 42 else f'-s{seed}'

    base = os.path.join(data.DATA_DIR, 'cache', f'proba_E5-reg-mcs20{sfx}.npy')
    if os.path.exists(base):
        P = np.load(base).astype(np.float32)
        E.log(f'E6-base{sfx}', 'raw+consensus', 'Sprint 2 champion, re-scored',
              {**CHAMPION, 'split_seed': seed}, E.score(d, rules((P >= .5).astype(np.uint8))), 0,
              notes='same predictions as E5-reg-mcs20; adds test-like metric')

    keep = tr[d['test_like'][tr]]
    Xu, Yu, _ = E.dedup_configs(X[keep], d['Y_cons'][keep])
    with E.Timer() as t:
        P = BinaryRelevanceLGBM(CHAMPION).fit(Xu, Yu).predict_proba(Xv)
    np.save(os.path.join(data.DATA_DIR, 'cache', f'proba_E6-clean{sfx}.npy'), P.astype(np.float16))
    E.log(f'E6-clean{sfx}', 'raw+consensus (test-like rows)', 'Champion trained without non-test-like rows',
          {**CHAMPION, 'split_seed': seed, 'dropped_train_rows': int(len(tr) - len(keep))},
          E.score(d, rules((P >= .5).astype(np.uint8))), t.s,
          notes=f'{len(Xu):,} unique train configs')


if __name__ == '__main__':
    main(*sys.argv[1:])
