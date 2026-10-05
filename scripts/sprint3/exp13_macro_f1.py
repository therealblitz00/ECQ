"""Exp 13 - raise clean-like macro F1 without lowering clean-like EMR (Step 12).

Baseline = final model (per-label LightGBM, min_child_samples=5, reg_lambda=1, threshold 0.5,
no additive rules) on the cleaned training fold (k=4 filter).

Thresholds are never tuned on validation: each fit also produces 5-fold out-of-fold (OOF)
probabilities on the training fold's unique configurations (folds = configurations, so
grouped), weighted by how many training rows each configuration has.

Success rule (set before any candidate was run): on BOTH seeds, the paired configuration
bootstrap 95% CI of the macro-F1 difference is above 0, and the clean-like EMR difference
vs the baseline is >= 0.

Usage:
  exp13_macro_f1.py fit  <seed> <variant>     variant: base | cw<cap>  (class weights, capped)
  exp13_macro_f1.py diag <seed>               Step 1 diagnostics of the baseline
  exp13_macro_f1.py eval <seed> <candidate>   candidate: see CANDIDATES
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.model_selection import KFold

from src import data, experiment as E

CACHE = os.path.join(data.DATA_DIR, 'cache')
DROP_K = 4
BASE = dict(n_estimators=100, num_leaves=15, learning_rate=0.1, min_child_samples=5,
            reg_lambda=1.0, verbose=-1)
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
B = 1000


# ---------------------------------------------------------------- data / fitting
def setup(seed):
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    d['keep'] = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    d['Xu'], d['Yu'], d['wu'] = E.dedup_configs(d['X_train'][d['keep']], d['Y_cons'][d['keep']])
    d['cl'] = d['clean_like'][va]
    d['Yv'] = d['Y_cons'][va][d['cl']]
    d['gv'] = d['groups'][va][d['cl']]
    d['sfx'] = '' if seed == 42 else f'-s{seed}'
    return d


def spw_for(variant, Y):
    """Per-label scale_pos_weight. base: 1. cw<cap>: sqrt(neg/pos) clipped to [1, cap]."""
    if variant == 'base':
        return np.ones(Y.shape[1])
    cap = float(variant[2:])
    pos = Y.sum(0).astype(float)
    return np.clip(np.sqrt((len(Y) - pos) / np.maximum(pos, 1)), 1, cap)


def fit_predict(Xtr, Ytr, Xte, spw):
    Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)

    def one(j):
        y = Ytr[:, j]
        if y.min() == y.max():
            return np.full(len(Xte), float(y[0]), np.float32)
        m = lgb.LGBMClassifier(n_jobs=2, scale_pos_weight=float(spw[j]), **BASE).fit(Xtr, y)
        return m.predict_proba(Xte)[:, 1].astype(np.float32)
    cols = Parallel(n_jobs=8, backend='threading')(delayed(one)(j) for j in range(Ytr.shape[1]))
    return np.column_stack(cols)


def fit(seed, variant):
    seed = int(seed)
    d = setup(seed)
    Xu, Yu = d['Xu'], d['Yu']
    with E.Timer() as t:
        P = fit_predict(Xu, Yu, d['X_train'][d['va']], spw_for(variant, Yu))
        print('  main fit done', flush=True)
        oof = np.zeros(Yu.shape, np.float32)
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xu)):
            oof[b] = fit_predict(Xu[a], Yu[a], Xu[b], spw_for(variant, Yu[a]))
            print(f'  inner fold {k + 1}/5 done', flush=True)
    tag = f'E13-{variant}{d["sfx"]}'
    np.save(os.path.join(CACHE, f'proba_{tag}.npy'), P.astype(np.float16))
    np.save(os.path.join(CACHE, f'oof_{tag}.npy'), oof.astype(np.float16))
    pred = (P >= 0.5).astype(np.uint8)
    res = E.score(d, pred)
    res['f1_macro_cleanlike'] = macro_f1(d['Yv'], pred[d['cl']])
    params = {**BASE, 'drop_k': DROP_K, 'split_seed': seed, 'threshold': 0.5,
              'scale_pos_weight': 'none' if variant == 'base' else f'sqrt(neg/pos) clipped to [1, {variant[2:]}]'}
    E.log(tag, 'raw+consensus (corrupted rows dropped, k=4)', 'BR LightGBM, macro-F1 study', params, res, t.s,
          notes=f'{len(Xu):,} unique train configs; time includes 5 inner OOF fits')
    print(f'clean-like macro F1 {res["f1_macro_cleanlike"]:.4f}')


def load(d, variant):
    tag = f'E13-{variant}{d["sfx"]}'
    P = np.load(os.path.join(CACHE, f'proba_{tag}.npy')).astype(np.float32)
    oof = np.load(os.path.join(CACHE, f'oof_{tag}.npy')).astype(np.float32)
    return P, oof


# ---------------------------------------------------------------- metrics
def counts(Y, Pr, w=None):
    Y, Pr = Y.astype(bool), Pr.astype(bool)
    w = np.ones(len(Y), np.float32) if w is None else np.asarray(w, np.float32)
    tp = w @ (Y & Pr)
    fp = w @ (~Y & Pr)
    fn = w @ (Y & ~Pr)
    return tp, fp, fn


def f1_from(tp, fp, fn):
    return np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), np.nan)


def macro_f1(Y, Pr):
    """Mean F1 over labels with at least one positive in Y (as in the notebook)."""
    tp, fp, fn = counts(Y, Pr)
    act = (tp + fn) > 0
    return float(f1_from(tp, fp, fn)[act].mean())


# ---------------------------------------------------------------- thresholds (OOF only)
def best_thresholds(oof, Y, w, allowed=GRID, min_gain=0.0, min_pos=0, objective='f1'):
    """Per label, the threshold in `allowed` maximising row-weighted OOF F1. 0.5 is kept unless
    the gain over 0.5 is >= min_gain and the label has >= min_pos positive configurations.
    Ties go to the threshold closest to 0.5.
    objective='errors': minimise row-weighted wrong bits (FP + FN) instead, which is the
    EMR-aligned criterion; min_gain is then the required reduction in weighted wrong rows."""
    if objective == 'errors':
        return error_thresholds(oof, Y, w, allowed, min_gain)
    thr = np.full(Y.shape[1], 0.5, np.float32)
    f05 = f1_from(*counts(Y, oof >= 0.5, w))
    npos = Y.sum(0)
    for j in range(Y.shape[1]):
        if npos[j] == 0 or npos[j] < min_pos:
            continue
        y = Y[:, j].astype(bool)
        best_t, best_f = 0.5, f05[j]
        for t in allowed:
            p = oof[:, j] >= t
            tp, fp, fn = w[y & p].sum(), w[~y & p].sum(), w[y & ~p].sum()
            f = 2 * tp / max(2 * tp + fp + fn, 1e-9)
            if f > best_f + 1e-9 or (abs(f - best_f) <= 1e-9 and abs(t - 0.5) < abs(best_t - 0.5)):
                best_t, best_f = t, f
        if best_f - f05[j] >= min_gain:
            thr[j] = best_t
    return thr


def error_thresholds(oof, Y, w, allowed, min_gain):
    thr = np.full(Y.shape[1], 0.5, np.float32)
    for j in range(Y.shape[1]):
        y = Y[:, j].astype(bool)
        if not y.any():
            continue
        err05 = w[y != (oof[:, j] >= 0.5)].sum()
        errs = [(w[y != (oof[:, j] >= t)].sum(), abs(t - 0.5), t) for t in allowed]
        e, _, t = min(errs)
        if err05 - e >= max(min_gain, 1e-9):
            thr[j] = t
    return thr


CANDIDATES = {
    # name: (variant whose probabilities are used, threshold rule; None = 0.5 everywhere)
    'thr-free': ('base', dict()),
    'thr-low': ('base', dict(allowed=GRID[GRID <= 0.5])),
    'thr-net': ('base', dict(allowed=GRID[GRID <= 0.5], objective='errors', min_gain=2.0)),
}


# ---------------------------------------------------------------- paired bootstrap over configurations
def boot_weights(groups, seed=0):
    codes, _ = pd.factorize(groups)
    n_cfg = codes.max() + 1
    G = sparse.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))),
                          shape=(n_cfg, len(codes)))
    rng = np.random.default_rng(seed)
    W = np.zeros((B, n_cfg), np.float32)
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n_cfg, n_cfg), minlength=n_cfg)
    return G, W


def boot_stats(G, W, Y, Pr):
    Yb, Pb = Y.astype(bool), Pr.astype(bool)
    tp = W @ (G @ (Yb & Pb).astype(np.float32))
    fp = W @ (G @ (~Yb & Pb).astype(np.float32))
    fn = W @ (G @ (Yb & ~Pb).astype(np.float32))
    act = (tp + fn) > 0
    f1 = np.where(act, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9), 0).sum(1) / act.sum(1)
    ok = G @ (Yb == Pb).all(1).astype(np.float32)
    n = np.asarray(G.sum(1)).ravel()
    emr = (W @ ok) / (W @ n)
    return f1, emr


def eval_candidate(seed, name):
    seed = int(seed)
    d = setup(seed)
    variant, rule = CANDIDATES[name]
    cl, Y = d['cl'], d['Yv']
    P0, _ = load(d, 'base')
    P, oof = load(d, variant)
    thr = np.full(Y.shape[1], 0.5, np.float32) if rule is None else \
        best_thresholds(oof, d['Yu'], d['wu'].astype(np.float32), **rule)
    pred0 = (P0[cl] >= 0.5).astype(np.uint8)
    full = (P >= thr).astype(np.uint8)
    pred = full[cl]
    G, W = boot_weights(d['gv'])
    f1a, emra = boot_stats(G, W, Y, pred0)
    f1b, emrb = boot_stats(G, W, Y, pred)
    df, de = f1b - f1a, (emrb - emra) * 100
    f1_0, f1_1 = macro_f1(Y, pred0), macro_f1(Y, pred)
    e0, e1 = float((pred0 == Y).all(1).mean()), float((pred == Y).all(1).mean())
    out = dict(seed=seed, candidate=name,
               f1_base=round(f1_0, 4), f1_cand=round(f1_1, 4), f1_delta=round(f1_1 - f1_0, 4),
               f1_ci=np.percentile(df, [2.5, 97.5]).round(4).tolist(),
               emr_base=round(e0 * 100, 3), emr_cand=round(e1 * 100, 3), emr_delta_pt=round((e1 - e0) * 100, 3),
               emr_ci_pt=np.percentile(de, [2.5, 97.5]).round(3).tolist(),
               rows_net=int((pred == Y).all(1).sum() - (pred0 == Y).all(1).sum()),
               thr_lowered=int((thr < 0.5).sum()), thr_raised=int((thr > 0.5).sum()),
               fn=int(((Y == 1) & (pred == 0)).sum()), fp=int(((Y == 0) & (pred == 1)).sum()))
    out['success'] = bool(out['f1_ci'][0] > 0 and out['emr_delta_pt'] >= 0)
    res = E.score(d, full)
    res['f1_macro_cleanlike'] = f1_1
    res.update({k: v for k, v in out.items() if k not in ('seed', 'candidate')})
    rule_log = 'none' if rule is None else {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rule.items()}
    E.log(f'E13-{name}{d["sfx"]}', 'raw+consensus (corrupted rows dropped, k=4)', 'BR LightGBM, macro-F1 study',
          {'probs': variant, 'threshold_rule': rule_log, 'split_seed': seed,
           'tuned_on': '5-fold OOF of the training fold (row-weighted)'}, res, 0)
    np.save(os.path.join(CACHE, f'thr_E13-{name}{d["sfx"]}.npy'), thr)
    print(json.dumps(out))


# ---------------------------------------------------------------- Step 1 diagnostics
def diag(seed):
    seed = int(seed)
    d = setup(seed)
    P, oof = load(d, 'base')
    P = P[d['cl']]
    Y, pred = d['Yv'], (P >= 0.5).astype(np.uint8)
    tp, fp, fn = counts(Y, pred)
    act = (tp + fn) > 0
    f1 = f1_from(tp, fp, fn)
    n_act = act.sum()
    print(f'seed {seed}: clean-like EMR {(pred == Y).all(1).mean():.4%} | macro F1 {np.nanmean(f1[act]):.4f} '
          f'over {n_act} items with a validation positive | micro F1 '
          f'{2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()):.4f}')
    print(f'total FN {int(fn.sum()):,} vs FP {int(fp.sum()):,}')

    pos_tr = d['Yu'].sum(0)
    sup = pd.cut(tp + fn, [0, 2, 5, 20, 100, 1e9], labels=['1-2', '3-5', '6-20', '21-100', '>100'])
    trs = pd.cut(pos_tr, [-1, 5, 20, 50, 200, 1e9], labels=['<=5', '6-20', '21-50', '51-200', '>200'])
    loss = np.where(act, 1 - np.nan_to_num(f1), 0)
    zero = act & (tp == 0)
    for name, bins in (('validation positives (rows)', sup), ('training positive configurations', trs)):
        t = pd.DataFrame({'bin': bins, 'f1': f1, 'loss': loss, 'zero': zero})[act]
        g = t.groupby('bin', observed=True).agg(items=('f1', 'size'), mean_F1=('f1', 'mean'),
                                                items_F1_0=('zero', 'sum'), loss=('loss', 'sum'))
        g['share_of_macro_loss'] = g['loss'] / g['loss'].sum()
        print(f'\nby {name}:')
        print(g.drop(columns='loss').round(3).to_string())

    print(f'\nitems with F1 = 0: {zero.sum()} ({zero.sum() / n_act:.1%} of scored items); never predicted '
          f'positive: {(zero & (fp == 0)).sum()}, only false alarms: {(zero & (fp > 0)).sum()}')
    print(f'macro-F1 lost to F1=0 items: {zero.sum() / n_act:.4f} of {loss.sum() / n_act:.4f} total')

    w = d['wu'].astype(np.float32)
    a = d['Yu'].sum(0) > 0
    f_oof05 = f1_from(*counts(d['Yu'], oof >= 0.5, w))
    thr_oof = best_thresholds(oof, d['Yu'], w)
    f_oofbest = f1_from(*counts(d['Yu'], oof >= thr_oof, w))
    print(f'\nOOF (training fold, row-weighted): macro F1 at 0.5 {np.nanmean(f_oof05[a]):.4f} -> best per-item '
          f'threshold {np.nanmean(f_oofbest[a]):.4f}; {(thr_oof < 0.5).sum()} items want a lower threshold, '
          f'{(thr_oof > 0.5).sum()} a higher one')
    orc = np.zeros(Y.shape[1])
    for j in np.flatnonzero(act):
        orc[j] = max(f1_from(*counts(Y[:, j:j + 1], P[:, j:j + 1] >= t))[0] for t in GRID)
    print(f'validation-oracle per-item threshold (upper bound, diagnostic only, never used to select): '
          f'macro F1 {orc[act].mean():.4f}')
    sep = np.zeros(Y.shape[1], bool)
    for j in np.flatnonzero(zero):
        pos, neg = P[Y[:, j] == 1, j], P[Y[:, j] == 0, j]
        sep[j] = pos.min() > np.quantile(neg, 0.999)
    print(f'F1=0 items whose positives all score above the 99.9th percentile of negatives '
          f'(ranked right, only the threshold is wrong): {sep.sum()} of {zero.sum()}')
    pmax = np.array([P[Y[:, j] == 1, j].max() for j in np.flatnonzero(zero)])
    print('F1=0 items, highest probability given to any true positive:',
          pd.cut(pmax, [-1e-9, 0.05, 0.2, 0.35, 0.5], labels=['<0.05', '0.05-0.2', '0.2-0.35', '0.35-0.5'])
          .value_counts().sort_index().to_dict())

    # Are the F1=0 items' positives the residue of the random rare-pack injection?
    rare_v = d['rare_count'][d['va']][d['cl']] > 0
    pos_rows = Y[:, zero].any(1)
    print(f'\nclean-like validation rows with >= 1 rare CRM pack: {rare_v.mean():.1%} of all rows, but '
          f'{rare_v[pos_rows].mean():.1%} of rows billed for an F1=0 item')
    Xu, Yu = d['Xu'].astype(np.float32), d['Yu'].astype(np.float32)
    cond = (Xu.T @ Yu) / np.maximum(Xu.sum(0)[:, None], 1)       # P(item | CRM column), training fold
    best = cond.max(0)
    for nm, m in (('F1=0 items', zero), ('items with F1 >= 0.5', act & (np.nan_to_num(f1) >= 0.5))):
        print(f'{nm}: median over items of the best P(item | single CRM column) in training = '
              f'{np.median(best[m]):.2f}')


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    {'fit': fit, 'diag': diag, 'eval': eval_candidate}[cmd](*args)
