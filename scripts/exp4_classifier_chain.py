"""Exp 4 - co-occurrence classifier chain (LightGBM), unique configs + consensus labels."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E
from src.models import CoOccurrenceChainLGBM


def main(n_parents=30, mcs=5):
    n_parents, mcs = int(n_parents), int(mcs)
    d = E.setup()
    tr, va = d['tr'], d['va']
    X = data.features(d, 'raw')
    Xu, Yu, _ = E.dedup_configs(X[tr], d['Y_cons'][tr])
    params = {'min_child_samples': mcs, 'reg_lambda': 1.0} if mcs != 5 else {}
    with E.Timer() as t:
        model = CoOccurrenceChainLGBM(params, n_parents=n_parents).fit(Xu, Yu)
    P = model.predict_proba(X[va])
    exp_id = f'E4-cc{n_parents}' + (f'-mcs{mcs}' if mcs != 5 else '')
    np.save(os.path.join(data.DATA_DIR, 'cache', f'proba_{exp_id}.npy'), P.astype(np.float16))
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), X[va], d['crm_cols'], d['bil_cols'])
    E.log(exp_id, 'raw+consensus', 'Co-occurrence Classifier Chain LightGBM + rules',
          {'n_parents': n_parents, 'order': 'prevalence desc', 'min_child_samples': mcs},
          E.score(d, pred), t.s, notes='teacher-forced training, sequential prediction')


if __name__ == '__main__':
    main(*sys.argv[1:])
