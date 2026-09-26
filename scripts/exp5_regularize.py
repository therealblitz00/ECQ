"""Exp 5b - variance reduction for per-label LightGBM: regularized params and seed bagging."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E
from src.models import BinaryRelevanceLGBM

CONFIGS = {
    'E5-reg-mcs20': dict(min_child_samples=20, reg_lambda=1.0),
    'E5-reg-mcs50': dict(min_child_samples=50, reg_lambda=1.0),
    'E5-bag3': dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
}


def main(which, seed=42):
    d = E.setup(seed=int(seed))
    tr, va = d['tr'], d['va']
    X = data.features(d, 'raw')
    Xu, Yu, _ = E.dedup_configs(X[tr], d['Y_cons'][tr])
    rules = lambda Y: data.apply_additive_rules(Y, X[va], d['crm_cols'], d['bil_cols'])
    for exp_id in which:
        params = CONFIGS[exp_id]
        n_models = 3 if exp_id.startswith('E5-bag') else 1
        with E.Timer() as t:
            P = np.mean([BinaryRelevanceLGBM({**params, 'random_state': s}).fit(Xu, Yu).predict_proba(X[va])
                         for s in range(n_models)], axis=0)
        tag = f'{exp_id}-s{seed}' if int(seed) != 42 else exp_id
        np.save(os.path.join(data.DATA_DIR, 'cache', f'proba_{tag}.npy'), P.astype(np.float16))
        for thr in (0.5, 0.6):
            E.log(f'{tag}-t{thr}', 'raw+consensus', 'BR LightGBM (variance reduction) + rules',
                  {**params, 'n_models': n_models, 'thr': thr, 'split_seed': int(seed)},
                  E.score(d, rules((P >= thr).astype(np.uint8))), t.s)


if __name__ == '__main__':
    args = sys.argv[1:]
    seed = args.pop(0).split('=')[1] if args and args[0].startswith('seed=') else 42
    main(args or list(CONFIGS), seed)
