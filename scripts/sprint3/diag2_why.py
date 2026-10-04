"""Diagnostic 2 - why do non-test-like rows exist, and why are certain pack columns missed?

(A) Merge hypothesis: is a non-test-like training row the OR of two normal (test-like) rows,
    and is its billing the OR of their billings?
(B) Anatomy of the most-missed BIL pack columns on test-like validation rows.

Read-only. Run:  .venv/Scripts/python scripts/sprint3/diag2_why.py > data/cache/diag2.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src import data, experiment as E, metrics as M

TOP_MISSED = ['BIL_3940_PACK', 'BIL_4289_PACK', 'BIL_4297_PACK', 'BIL_3803_PACK', 'BIL_3812_PACK']


def section(title):
    print('\n' + '=' * 100 + f'\n{title}\n' + '=' * 100)


def unique_rows(A, Y):
    keys = pd.Series([r.tobytes() for r in np.packbits(A, axis=1)])
    first = ~keys.duplicated().to_numpy()
    return A[first], Y[first]


def merge_test(d, n_sample=600, seed=0):
    section('(A) Are non-test-like rows the OR of two normal rows?')
    X, Y, ok = d['X_train'], d['Y_cons'], d['test_like']
    crm = np.array(d['crm_cols'])
    status = np.array([c.startswith('CRM_SUBSCRIBER_STATUS_ORIG_') for c in crm])
    multi = (X[:, status].sum(1) >= 2) & ~ok
    print(f'non-test-like train rows: {(~ok).sum():,} | of which 2+ statuses: {multi.sum():,}')

    Nx, Ny = unique_rows(X[ok], Y[ok])
    Nf = Nx.astype(np.float32)
    nsz = Nf.sum(1)
    rng = np.random.default_rng(seed)
    rows = rng.choice(np.flatnonzero(multi), size=min(n_sample, multi.sum()), replace=False)

    found, bil_or_match, bil_or_dist, own_dist = 0, 0, [], []
    for r in rows:
        x = X[r].astype(np.float32)
        sub = np.flatnonzero((Nf @ x) == nsz)            # normal configs that are subsets of x
        if len(sub) < 2:
            continue
        S = Nf[sub]
        inter = S @ S.T
        union = nsz[sub][:, None] + nsz[sub][None, :] - inter
        hit = np.argwhere(np.triu(union == x.sum(), 1))  # pairs whose OR equals x exactly
        if not len(hit):
            continue
        found += 1
        best = None
        for i, j in hit[:200]:
            ybar = Ny[sub[i]] | Ny[sub[j]]
            dist = int((ybar != Y[r]).sum())
            best = dist if best is None else min(best, dist)
        bil_or_dist.append(best)
        bil_or_match += best == 0
    print(f'sampled {len(rows)} multi-status rows: {found} ({found / len(rows):.1%}) are exactly the OR of '
          f'two normal training configurations')
    if found:
        bd = np.array(bil_or_dist)
        print(f'  of those, BIL == OR of the two normal BILs exactly: {bil_or_match} ({bil_or_match / found:.1%}); '
              f'median Hamming distance {np.median(bd):.0f}, 90th pct {np.percentile(bd, 90):.0f}')

    # Reference: how many CRM packs / statuses do multi rows carry vs. 2x a normal row?
    pk = np.array([c.endswith('_PACK') for c in crm])
    print(f'mean CRM packs: 2+-status rows {X[multi][:, pk].sum(1).mean():.1f} | normal rows '
          f'{X[ok][:, pk].sum(1).mean():.1f} (2x = {2 * X[ok][:, pk].sum(1).mean():.1f})')
    print('number of statuses on 2+-status rows:', pd.Series(X[multi][:, status].sum(1)).value_counts().to_dict())
    names = np.array([c.split('_ORIG_')[1] for c in crm[status]])
    combos = pd.Series(['+'.join(names[X[i, status] == 1]) for i in np.flatnonzero(multi)]).value_counts()
    print('most common status combinations (a real status history would not be this uniform):')
    print(combos.head(10).to_string())


def missed_columns(d):
    section('(B) Anatomy of the most-missed BIL pack columns (test-like validation rows)')
    tr, va = d['tr'], d['va']
    X, Y = d['X_train'], d['Y_cons']
    crm, bil = np.array(d['crm_cols']), list(d['bil_cols'])
    P = np.load(os.path.join(data.DATA_DIR, 'cache', 'proba_E5-reg-mcs20.npy')).astype(np.float32)
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), X[va], d['crm_cols'], d['bil_cols'])
    tl = d['test_like'][va]
    err = M.row_errors(Y[va], pred)

    trk = tr[d['test_like'][tr]]
    Xu, Yu = unique_rows(X[trk], Y[trk])
    for name in TOP_MISSED:
        j = bil.index(name)
        yt, yv, pv, pr = Yu[:, j], Y[va][:, j], P[:, j], pred[:, j]
        miss = tl & (yv == 1) & (pr == 0)
        hit = tl & (yv == 1) & (pr == 1)
        sole = miss & (err == 1)
        print(f'\n--- {name}: train prevalence {yt.mean():.4%} ({int(yt.sum())} unique configs) | '
              f'val test-like positives {int((tl & (yv == 1)).sum())}: hit {int(hit.sum())}, missed {int(miss.sum())} '
              f'({int(sole.sum())} of them the only error in the row)')
        if miss.any():
            print(f'    champion prob on missed positives: median {np.median(pv[miss]):.3f}, '
                  f'max {pv[miss].max():.3f}')
        # Single CRM drivers in train
        pos = Xu[yt == 1].mean(0)
        neg = Xu[yt == 0].mean(0)
        sup = Xu.sum(0)
        cond = np.divide(Xu[yt == 1].sum(0), sup, out=np.zeros(len(sup)), where=sup > 0)
        top = np.argsort(-(pos - neg))[:5]
        print('    CRM packs most enriched in train positives: ' + ', '.join(
            f'{crm[i]} (in {pos[i]:.0%} of pos, {neg[i]:.1%} of neg, P(b|c)={cond[i]:.2f}, n={int(sup[i])})'
            for i in top))
        if miss.any():
            mv = X[va][miss].mean(0)
            hv = X[va][hit].mean(0) if hit.any() else np.zeros_like(mv)
            dif = np.argsort(-np.abs(mv - hv))[:5]
            print('    CRM packs that differ most between missed and hit val positives: ' + ', '.join(
                f'{crm[i]} (missed {mv[i]:.0%} vs hit {hv[i]:.0%}, train support {int(sup[i])})' for i in dif))
        # Does a tiny tree recover it? (depth 3, min 5 samples per leaf)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(Xu, yt)
        tp = tree.predict(X[va])
        print(f'    depth-3 tree on test-like val: recall {np.mean(tp[tl & (yv == 1)] == 1):.3f} '
              f'(champion {np.mean(pr[tl & (yv == 1)] == 1):.3f}) | false positives {int(((tp == 1) & (yv == 0) & tl).sum())} '
              f'(champion {int(((pr == 1) & (yv == 0) & tl).sum())})')
        rules = export_text(tree, feature_names=list(crm), max_depth=3)
        print('    ' + rules.replace('\n', '\n    ').rstrip())


def label_consistency(d):
    section('(C) Which BIL columns flip between rows with IDENTICAL CRM configurations?')
    Yr, g, bil = d['Y_raw'], d['groups'], list(d['bil_cols'])
    rep = np.bincount(g)[g] >= 2
    df = pd.DataFrame(Yr[rep])
    df['g'] = g[rep]
    s = df.groupby('g').sum().to_numpy()
    n = df.groupby('g').size().to_numpy()[:, None]
    mixed, anypos = (s > 0) & (s < n), s > 0
    rate = pd.Series(mixed.sum(0) / np.maximum(anypos.sum(0), 1), index=bil)
    for b in TOP_MISSED + ['BIL_3921_PACK', 'BIL_3762_PACK']:
        j = bil.index(b)
        print(f'  {b}: inconsistent in {rate[b]:.1%} of the {int(anypos[:, j].sum())} repeated configs where it appears')
    print(f'columns inconsistent in >=50% of their configs: {(rate >= .5).sum()} | median over all columns: {rate.median():.1%}')
    va, Y = d['va'], d['Y_cons']
    P = np.load(os.path.join(data.DATA_DIR, 'cache', 'proba_E5-reg-mcs20.npy')).astype(np.float32)
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), d['X_train'][va], d['crm_cols'], d['bil_cols'])
    diff = pred != Y[va]
    w = diff.any(1) & d['test_like'][va]
    only = w & ~diff[:, ~(rate >= .5).to_numpy()].any(1)
    print(f'test-like val rows wrong: {w.mean():.2%} | wrong ONLY in such inconsistent columns: {only.mean():.2%}')


def injection_test(d, rare_freq=0.002, heavy_k=3):
    section('(D) Were random rare packs injected into some training rows?')
    X, Xte, Y, ok, va = d['X_train'], d['X_test'], d['Y_cons'], d['test_like'], d['va']
    cp = np.array([c.endswith('_PACK') for c in d['crm_cols']])
    bp = np.array([c.endswith('_PACK') for c in d['bil_cols']])
    rare = X[:, cp].mean(0) < rare_freq
    rtr, rte = X[:, cp][:, rare].sum(1), Xte[:, cp][:, rare].sum(1)
    print(f'{rare.sum()} CRM packs with train frequency < {rare_freq:.1%}')
    print('  #rare CRM packs per row, train:', pd.Series(np.minimum(rtr, 12)).value_counts().sort_index().to_dict())
    print('  #rare CRM packs per row, test: ', pd.Series(np.minimum(rte, 12)).value_counts().sort_index().to_dict())
    rb = Y[:, bp].mean(0) < rare_freq
    rbt = Y[:, bp][:, rb].sum(1)
    print('  rare CRM packs (rows) vs rare BIL packs (columns) per train row:')
    print(pd.crosstab(np.minimum(rtr, 6), np.minimum(rbt, 6)).to_string())
    print(f'  non-test-like rows carry {rtr[~ok].mean():.2f} rare CRM / {rbt[~ok].mean():.2f} rare BIL packs; '
          f'test-like rows {rtr[ok].mean():.3f} / {rbt[ok].mean():.3f}')
    heavy = rtr >= heavy_k
    cnt = X[heavy][:, cp][:, rare].sum(0)
    rho = pd.Series(cnt).corr(pd.Series(Xte[:, cp].mean(0)[rare]), method='spearman')
    print(f'rows with >={heavy_k} rare CRM packs: train {heavy.mean():.2%} | test {np.mean(rte >= heavy_k):.2%}')
    print(f'  how evenly those rows use the {rare.sum()} rare packs: coefficient of variation {cnt.std() / cnt.mean():.2f} '
          f'(uniform random ~ small); Spearman with natural test frequency {rho:.2f}')
    P = np.load(os.path.join(data.DATA_DIR, 'cache', 'proba_E5-reg-mcs20.npy')).astype(np.float32)
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), X[va], d['crm_cols'], d['bil_cols'])
    err, hv = M.row_errors(Y[va], pred), heavy[va]
    print(f'champion val EMR: rows with >={heavy_k} rare packs ({hv.mean():.2%}) {np.mean(err[hv] == 0):.2%} | '
          f'other rows {np.mean(err[~hv] == 0):.2%} | other & test-like {np.mean(err[~hv & ok[va]] == 0):.2%}')
    print(f'share of 6+-bit val rows that are (>={heavy_k} rare packs OR not test-like): '
          f'{np.mean((hv | ~ok[va])[err >= 6]):.1%}')


def main():
    d = E.setup(seed=42)
    merge_test(d)
    missed_columns(d)
    label_consistency(d)
    injection_test(d)


if __name__ == '__main__':
    main()
