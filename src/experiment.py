"""Shared setup + result logging so every experiment uses the same split and targets."""
import json
import os
import time

import numpy as np

from src import data, metrics, validation

RESULTS = os.path.join(data.DATA_DIR, 'cache', 'results.jsonl')


def setup(seed=42, val_size=0.2):
    d = data.load()
    groups, _ = validation.factorize_rows(d['X_train'])
    Y_cons, is_noisy = validation.consensus_targets(groups, d['Y_raw'])
    tr, va = validation.grouped_holdout(groups, val_size=val_size, seed=seed)
    test_like = validation.test_like_mask(d['X_train'], d['crm_cols'])
    rare_count = validation.rare_pack_count(d['X_train'], d['crm_cols'])
    clean_like = test_like & (rare_count < validation.HEAVY_K)
    d.update(groups=groups, Y_cons=Y_cons, is_noisy=is_noisy, tr=tr, va=va,
             test_like=test_like, rare_count=rare_count, clean_like=clean_like)
    return d


def score(d, Y_pred_val):
    """Metrics on the validation fold against consensus targets: all rows, and the test-like
    rows only (the population the test set actually contains)."""
    va = d['va']
    out = metrics.evaluate(d['Y_cons'][va], Y_pred_val)
    out['emr_vs_raw'] = metrics.exact_match_ratio(d['Y_raw'][va], Y_pred_val)
    for key, mask in [('emr_testlike', d['test_like'][va]), ('emr_cleanlike', d['clean_like'][va])]:
        out[key] = metrics.exact_match_ratio(d['Y_cons'][va][mask], np.asarray(Y_pred_val)[mask])
    return out


def log(exp_id, pipeline, family, params, res, train_time, notes=''):
    rec = dict(exp_id=exp_id, pipeline=pipeline, family=family, params=params,
               train_time_s=round(train_time, 1), notes=notes, ts=time.strftime('%Y-%m-%d %H:%M'), **res)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    tl = ''.join(f" {k.upper()}={res[k]:.4%}" for k in ('emr_testlike', 'emr_cleanlike') if k in res)
    print(f"[{exp_id}] {pipeline} {family} EMR={res['emr']:.4%}{tl} HL={res['hamming_loss']:.6f} "
          f"d1={res['share_dist1']:.3%} t={train_time:.0f}s {notes}")
    return rec


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0


def dedup_configs(X, Y):
    """Collapse rows to unique X configs (Y assumed consistent per config, e.g. consensus)."""
    codes, _ = validation.factorize_rows(X)
    _, first = np.unique(codes, return_index=True)
    return X[first], Y[first], np.bincount(codes)[codes[first]]
