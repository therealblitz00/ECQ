"""Exp 3 - per-label LightGBM (binary relevance), Section 10 vs Section 11 pipelines."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E, validation as V
from src.models import BinaryRelevanceLGBM

OUT = os.path.join(data.DATA_DIR, 'cache')


def run(d, exp_id, pipe, Xtr, Ytr, Xva, w=None, rules=False, notes=''):
    with E.Timer() as t:
        model = BinaryRelevanceLGBM().fit(Xtr, Ytr, sample_weight=w)
    P = model.predict_proba(Xva)
    np.save(os.path.join(OUT, f'proba_{exp_id}.npy'), P.astype(np.float16))
    pred = (P >= 0.5).astype(np.uint8)
    res = E.score(d, pred)
    E.log(exp_id, pipe, 'Binary Relevance LightGBM', {'n_estimators': 100, 'num_leaves': 15},
          res, t.s, notes=notes)
    if rules:
        pr = data.apply_additive_rules(pred, d['X_train'][d['va']], d['crm_cols'], d['bil_cols'])
        E.log(exp_id + '+rules', pipe, 'Binary Relevance LightGBM + 42 additive rules',
              {'n_estimators': 100, 'num_leaves': 15}, E.score(d, pr), t.s,
              notes='42 Section 11 rule columns overwritten')
    return P


def main(which):
    d = E.setup()
    tr, va = d['tr'], d['va']

    if 'E3a' in which:
        # Superseded: multiplicity weights (max 8114) made rare labels fire on ~5% of rows.
        X = data.features(d, 'raw')
        Xu, Yu, cnt = E.dedup_configs(X[tr], d['Y_cons'][tr])
        run(d, 'E3a', 'raw+consensus', Xu, Yu, X[va], w=cnt.astype(np.float32),
            notes='unique configs, consensus labels, weight=multiplicity (BUG: over-weights)')

    if 'E3a2' in which:
        X = data.features(d, 'raw')
        Xu, Yu, _ = E.dedup_configs(X[tr], d['Y_cons'][tr])
        run(d, 'E3a2', 'raw+consensus', Xu, Yu, X[va], rules=True,
            notes='unique configs, consensus labels, unweighted')

    # Feature-set ablations on identical (deduplicated, unweighted, consensus) training rows
    for exp_id, pipe in [('E3d', 's10'), ('E3e', 's11')]:
        if exp_id in which:
            X = data.features(d, pipe)
            Xu, Yu, _ = E.dedup_configs(data.features(d, 'raw')[tr], d['Y_cons'][tr])
            codes, _ = V.factorize_rows(data.features(d, 'raw')[tr])
            _, first = np.unique(codes, return_index=True)
            run(d, exp_id, f'{pipe}+consensus', X[tr][first], Yu, X[va], rules=pipe == 's11',
                notes=f'{pipe} features, unique configs, consensus labels, unweighted')

    if 'E3b' in which:
        # Section 10 pipeline as defined: 734 CRM feats, majority-relabeled row labels
        X = data.features(d, 's10')
        run(d, 'E3b', 's10', X[tr], data.train_target(d, 's10')[tr], X[va],
            notes='all rows, S10 relabeled targets')

    if 'E3c' in which:
        # Section 11 pipeline: 742 CRM + 45 SVD, raw labels; with and without the 42 hard rules
        X = data.features(d, 's11')
        run(d, 'E3c', 's11', X[tr], data.train_target(d, 's11')[tr], X[va], rules=True,
            notes='all rows, raw targets, 742 CRM + 45 SVD')


if __name__ == '__main__':
    main(sys.argv[1:] or ['E3a2', 'E3b', 'E3c'])
