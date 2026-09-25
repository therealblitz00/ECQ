"""Fit the frozen champion on all of train.csv and write data/derived/submission_sprint2.csv.

Champion (selected on grouped validation, confirmed on a second split seed):
per-label LightGBM, raw 745 CRM features, unique CRM configurations with consensus labels,
min_child_samples=20 + reg_lambda=1, threshold 0.5, then the 42 Section 11 additive rules.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import data, experiment as E, metrics as M, validation as V
from src.models import BinaryRelevanceLGBM

CHAMPION_PARAMS = dict(min_child_samples=20, reg_lambda=1.0)
SUBMISSION = os.path.join(data.DERIVED_DIR, 'submission_sprint2.csv')


def main():
    d = data.load()
    groups, _ = V.factorize_rows(d['X_train'])
    Y_cons, _ = V.consensus_targets(groups, d['Y_raw'])
    Xu, Yu, _ = E.dedup_configs(d['X_train'], Y_cons)
    print(f'training on {len(Xu):,} unique CRM configurations x {Yu.shape[1]} labels')

    with E.Timer() as t:
        model = BinaryRelevanceLGBM(CHAMPION_PARAMS).fit(Xu, Yu)
    P = model.predict_proba(d['X_test'])
    pred = data.apply_additive_rules((P >= 0.5).astype(np.uint8), d['X_test'], d['crm_cols'], d['bil_cols'])
    print(f'fit+predict {t.s:.0f}s')

    bill = [''.join(r) for r in pred.astype(str)]
    sub = pd.DataFrame({'MSISDN': d['msisdn_test'], 'Bill_Conf': bill})
    assert len(sub) == 97_100 and sub.notna().all().all()
    assert sub['Bill_Conf'].str.len().eq(731).all() and sub['Bill_Conf'].str.fullmatch('[01]+').all()
    assert sub['MSISDN'].is_unique
    sub.to_csv(SUBMISSION, index=False)
    print(f'wrote {SUBMISSION}: {len(sub):,} rows, Bill_Conf length 731, no NaNs')

    # External diagnostic only - model selection was frozen before this was computed.
    sol = pd.read_csv(os.path.join(data.DATA_DIR, 'solution.csv'), dtype=str)
    m = sub.merge(sol, on='MSISDN', suffixes=('', '_true'))
    assert len(m) == len(sub), 'MSISDN mismatch with solution.csv'
    Yt = np.array([np.frombuffer(s.encode(), np.uint8) - 48 for s in m['Bill_Conf_true']])
    Yp = np.array([np.frombuffer(s.encode(), np.uint8) - 48 for s in m['Bill_Conf']])
    res = M.evaluate(Yt, Yp)
    res['emr_vs_raw'] = res['emr']
    E.log('EXT-solution', 'raw+consensus', 'FINAL champion on test (solution.csv diagnostic)',
          {**CHAMPION_PARAMS, 'thr': 0.5, 'rules': True, 'trained_on': 'all train'}, res, t.s,
          notes='external diagnostic only, not used for any selection')
    np.save(os.path.join(data.DATA_DIR, 'cache', 'proba_test_final.npy'), P.astype(np.float16))


if __name__ == '__main__':
    main()
