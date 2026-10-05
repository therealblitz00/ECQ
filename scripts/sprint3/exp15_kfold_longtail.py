"""Exp 15 - 5-fold confirmation of the Exp 14 long-tail specialist (Step 13).

Per fold (GroupKFold by CRM configuration over all training rows, as in Step 9):
  1. Stage 1 = champion (per-label LightGBM, mcs=5, lambda=1, threshold 0.5) on the cleaned
     training part (k=4 filter), unique configurations + consensus labels.
  2. Problem labels = items whose row-weighted 5-fold OOF F1 (inside the training part) < 0.5.
  3. Stage 2 = per problem label, LightGBM on CRM with scale_pos_weight = sqrt(neg/pos) capped
     at 10 (`crm-cw`).
  4. Merge = add-only, Stage-2 p >= 0.5, only on rows with >= 1 rare CRM pack (`rare` gate).
Also scored: the ablation 'Stage-1 probabilities re-thresholded at 0.3 under the same gate'.
Metrics on the held-out fold's clean-like rows, against consensus targets.

Usage: exp15_kfold_longtail.py [folds, e.g. 0,1,2,3,4]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sklearn.model_selection import KFold

from src import experiment as E, validation as V
from scripts.sprint3 import exp13_macro_f1 as X13

DROP_K = 4
OUT = os.path.join(os.path.dirname(E.RESULTS), 'e15_kfold.jsonl')


def run_fold(d, k, tr, va):
    X, Yc = d['X_train'], d['Y_cons']
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    Xu, Yu, wu = E.dedup_configs(X[keep], Yc[keep])
    wu = wu.astype(np.float32)
    ones = np.ones(Yu.shape[1])
    with E.Timer() as t:
        P1 = X13.fit_predict(Xu, Yu, X[va], ones)
        oof = np.zeros(Yu.shape, np.float32)
        for a, b in KFold(5, shuffle=True, random_state=0).split(Xu):
            oof[b] = X13.fit_predict(Xu[a], Yu[a], Xu[b], ones)
        f1_oof = X13.f1_from(*X13.counts(Yu, oof >= 0.5, wu))
        labels = np.flatnonzero((Yu.sum(0) > 0) & (np.nan_to_num(f1_oof, nan=1.0) < 0.5))
        P2 = X13.fit_predict(Xu, Yu[:, labels], X[va], X13.spw_for('cw10', Yu[:, labels]))

    cl = d['clean_like'][va]
    Y = Yc[va][cl]
    rare_crm = np.array([c.endswith('_PACK') for c in d['crm_cols']]) & (X.mean(0) < V.RARE_FREQ)
    gate = (X[va][:, rare_crm].sum(1) > 0)[:, None]
    base = (P1 >= 0.5).astype(np.uint8)

    def add(Pl, tau):
        pr = base.copy()
        sub = pr[:, labels]
        pr[:, labels] = np.where((sub == 0) & (Pl >= tau) & gate, 1, sub)
        return pr[cl]

    rows = {'champion': base[cl], 'specialist (crm-cw, rare gate, 0.5)': add(P2, 0.5),
            'ablation: Stage-1 at 0.3, rare gate': add(P1[:, labels], 0.3)}
    ok0 = (rows['champion'] == Y).all(1)
    res = {}
    for name, pr in rows.items():
        ok = (pr == Y).all(1)
        tp, fp, fn = X13.counts(Y, pr)
        res[name] = dict(emr=float(ok.mean()), macro_f1=X13.macro_f1(Y, pr), fp=int(fp.sum()), fn=int(fn.sum()),
                         recovered=int((ok & ~ok0).sum()), broken=int((~ok & ok0).sum()))
    rec = dict(fold=k, n_problem=int(len(labels)), n_clean_rows=int(cl.sum()), seconds=round(t.s), results=res)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def main(folds='0,1,2,3,4'):
    d = E.setup(seed=42)
    splits = V.grouped_kfold(d['groups'], 5)
    for k in map(int, folds.split(',')):
        run_fold(d, k, *splits[k])


if __name__ == '__main__':
    main(*sys.argv[1:])
