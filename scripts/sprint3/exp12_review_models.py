"""Exp 12 - models requested by the review (Step 9).

  kfold          5-fold GroupKFold (groups = exact CRM configuration) of the final model:
                 mean +/- std EMR on all / test-like / clean-like held-out rows.
  lr <seed>      binary-relevance logistic regression (liblinear, C=1) on the same cleaned
                 training data as the final model: the linear baseline for the
                 "billing is additive per pack" claim.

Final model = per-label LightGBM, raw CRM, unique configs + consensus labels, rows not
test-like or with >= 4 rare packs dropped, min_child_samples=5, reg_lambda=1, threshold 0.5,
no additive rules.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from src import data, experiment as E, metrics as M, validation as V
from src.models import BinaryRelevanceLGBM

FINAL = dict(min_child_samples=5, reg_lambda=1.0)
DROP_K = 4
CACHE = os.path.join(data.DATA_DIR, 'cache')


def clean_train(d, idx):
    keep = idx[d['test_like'][idx] & (d['rare_count'][idx] < DROP_K)]
    return E.dedup_configs(d['X_train'][keep], d['Y_cons'][keep])[:2]


def kfold(n_splits=5):
    d = E.setup(seed=42)
    rows = []
    thr_rows = []
    for f, (tr, va) in enumerate(V.grouped_kfold(d['groups'], n_splits)):
        d['tr'], d['va'] = tr, va
        Xu, Yu = clean_train(d, tr)
        with E.Timer() as t:
            P = BinaryRelevanceLGBM(FINAL).fit(Xu, Yu).predict_proba(d['X_train'][va])
        np.save(os.path.join(CACHE, f'proba_E12-kfold-f{f}.npy'), P.astype(np.float16))
        res = E.score(d, (P >= .5).astype(np.uint8))
        E.log(f'E12-kfold-f{f}', 'raw+consensus (corrupted rows dropped, k=4)',
              'Final model, 5-fold GroupKFold', {**FINAL, 'fold': f, 'n_splits': n_splits},
              res, t.s, notes=f'{len(va):,} held-out rows')
        rows.append(res)
        # Same fit, rare-column threshold 0.45 (selected on seed 42 in Step 4.2, re-tested here).
        keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
        rare = d['Y_cons'][keep].mean(0) < 0.002
        res_t = E.score(d, (P >= np.where(rare, 0.45, 0.5)).astype(np.uint8))
        E.log(f'E12-kfold-thr-f{f}', 'raw+consensus (corrupted rows dropped, k=4)',
              'Final model + rare-column threshold 0.45, 5-fold GroupKFold',
              {**FINAL, 'fold': f, 'n_splits': n_splits, 'rare_thr': 0.45}, res_t, 0, notes='same fit as E12-kfold')
        thr_rows.append(res_t)
    for label, rr in (('final', rows), ('+ rare thr 0.45', thr_rows)):
        for k in ('emr', 'emr_testlike', 'emr_cleanlike'):
            v = np.array([r[k] for r in rr])
            print(f'{label:16s} {k:14s} mean {v.mean():.4%}  std {v.std(ddof=1):.4%}  folds {np.round(v * 100, 2).tolist()}')
    diff = np.array([b['emr_cleanlike'] - a['emr_cleanlike'] for a, b in zip(rows, thr_rows)])
    print(f'threshold gain per fold (clean-like, points): {np.round(diff * 100, 3).tolist()} | mean {diff.mean() * 100:.3f}')


def _fit_lr(X, y):
    if y.min() == y.max():
        return float(y[0])
    return LogisticRegression(C=1.0, solver='liblinear', max_iter=200).fit(X, y)


def lr(seed):
    seed = int(seed)
    d = E.setup(seed=seed)
    Xu, Yu = clean_train(d, d['tr'])
    Xf = Xu.astype(np.float64)
    with E.Timer() as t:
        models = Parallel(n_jobs=8)(delayed(_fit_lr)(Xf, Yu[:, j]) for j in range(Yu.shape[1]))
    Xv = d['X_train'][d['va']].astype(np.float64)
    P = np.column_stack([np.full(len(Xv), m) if isinstance(m, float) else m.predict_proba(Xv)[:, 1] for m in models])
    sfx = '' if seed == 42 else f'-s{seed}'
    np.save(os.path.join(CACHE, f'proba_E12-lr{sfx}.npy'), P.astype(np.float16))
    E.log(f'E12-lr{sfx}', 'raw+consensus (corrupted rows dropped, k=4)',
          'Binary relevance logistic regression (liblinear, C=1)', {'C': 1.0, 'split_seed': seed},
          E.score(d, (P >= .5).astype(np.uint8)), t.s, notes=f'{len(Xu):,} unique train configs')


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'kfold': kfold, 'lr': lr}[mode](*args)
