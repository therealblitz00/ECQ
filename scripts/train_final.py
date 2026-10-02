"""Fit the frozen champion on all of train.csv and write the submission + test diagnostic.

Champion (selected on grouped validation, confirmed on a second split seed):
per-label LightGBM, raw 745 CRM features, unique CRM configurations with consensus labels,
min_child_samples=20 + reg_lambda=1, threshold 0.5, then the 42 Section 11 additive rules.

Usage:  train_final.py              Sprint 2 champion  -> submission_sprint2.csv
        train_final.py <k> [<mcs>]  Sprint 3: drop suspected-corrupted training rows (not
                                    test-like, or >= k rare CRM packs), optionally with
                                    min_child_samples=<mcs> -> submission_sprint3_k<k>[_mcs<mcs>].csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import data, experiment as E, metrics as M, validation as V
from src.models import BinaryRelevanceLGBM

CHAMPION_PARAMS = dict(min_child_samples=20, reg_lambda=1.0)


def main(drop_k=None, mcs=None):
    drop_k = int(drop_k) if drop_k else None
    params = {**CHAMPION_PARAMS, **({'min_child_samples': int(mcs)} if mcs else {})}
    tag = (f'sprint3_k{drop_k}' if drop_k else 'sprint2') + (f'_mcs{mcs}' if mcs else '')
    submission = os.path.join(data.DERIVED_DIR, f'submission_{tag}.csv')
    d = data.load()
    groups, _ = V.factorize_rows(d['X_train'])
    Y_cons, _ = V.consensus_targets(groups, d['Y_raw'])
    keep = np.ones(len(Y_cons), bool)
    if drop_k:
        keep = V.test_like_mask(d['X_train'], d['crm_cols']) & \
               (V.rare_pack_count(d['X_train'], d['crm_cols']) < drop_k)
        print(f'dropping {(~keep).sum():,} suspected-corrupted training rows ({(~keep).mean():.2%})')
    Xu, Yu, _ = E.dedup_configs(d['X_train'][keep], Y_cons[keep])
    print(f'training on {len(Xu):,} unique CRM configurations x {Yu.shape[1]} labels')

    with E.Timer() as t:
        model = BinaryRelevanceLGBM(params).fit(Xu, Yu)
    P = model.predict_proba(d['X_test'])
    pred = data.apply_additive_rules((P >= 0.5).astype(np.uint8), d['X_test'], d['crm_cols'], d['bil_cols'])
    print(f'fit+predict {t.s:.0f}s')

    bill = [''.join(r) for r in pred.astype(str)]
    sub = pd.DataFrame({'MSISDN': d['msisdn_test'], 'Bill_Conf': bill})
    assert len(sub) == 97_100 and sub.notna().all().all()
    assert sub['Bill_Conf'].str.len().eq(731).all() and sub['Bill_Conf'].str.fullmatch('[01]+').all()
    assert sub['MSISDN'].is_unique
    sub.to_csv(submission, index=False)
    print(f'wrote {submission}: {len(sub):,} rows, Bill_Conf length 731, no NaNs')

    # External diagnostic only - model selection was frozen before this was computed.
    sol = pd.read_csv(os.path.join(data.DATA_DIR, 'solution.csv'), dtype=str)
    m = sub.merge(sol, on='MSISDN', suffixes=('', '_true'))
    assert len(m) == len(sub), 'MSISDN mismatch with solution.csv'
    Yt = np.array([np.frombuffer(s.encode(), np.uint8) - 48 for s in m['Bill_Conf_true']])
    Yp = np.array([np.frombuffer(s.encode(), np.uint8) - 48 for s in m['Bill_Conf']])
    res = M.evaluate(Yt, Yp)
    res['emr_vs_raw'] = res['emr']
    E.log('EXT-solution' + (f'-drop-k{drop_k}' if drop_k else '') + (f'-mcs{mcs}' if mcs else ''), 'raw+consensus',
          'FINAL model on test (solution.csv diagnostic)',
          {**params, 'thr': 0.5, 'rules': True, 'drop_k': drop_k,
           'trained_on': 'all train' + (' minus suspected-corrupted rows' if drop_k else '')}, res, t.s,
          notes='external diagnostic only, not used for any selection')
    name = f'proba_test_{tag}.npy'
    np.save(os.path.join(data.DATA_DIR, 'cache', name), P.astype(np.float16))


if __name__ == '__main__':
    main(*sys.argv[1:])
