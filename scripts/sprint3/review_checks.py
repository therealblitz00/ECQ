"""Step 9 - checks requested by the mock jury review. Read-only except for logging the two
cheap baselines (rules floor, k-NN on the clean-like metric).

Run:  .venv/Scripts/python scripts/sprint3/review_checks.py > data/cache/review_checks.txt

Final model throughout = Step 4 champion WITHOUT additive rules (probabilities E8-mcs5-l1).
Uncertainty is computed at the level of CRM configurations (a configuration's rows share
both prediction and consensus target, so they are not independent).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src import data, experiment as E, metrics as M, validation as V
from scripts.sprint2.exp1_lookup_knn import knn_hamming

CACHE = os.path.join(data.DATA_DIR, 'cache')
DROP_K = 4
B = 2000


def section(t):
    print('\n' + '=' * 100 + f'\n{t}\n' + '=' * 100)


def load_p(name):
    return np.load(os.path.join(CACHE, f'proba_{name}.npy')).astype(np.float32)


def rules(d, pred):
    return data.apply_additive_rules(pred, d['X_train'][d['va']], d['crm_cols'], d['bil_cols'])


def cluster_boot(groups, values, rng, b=B):
    """Bootstrap the mean of `values` resampling whole groups. Returns (mean, se, lo, hi, p_le_0)."""
    codes, uniq = pd.factorize(groups)
    s = np.bincount(codes, weights=values)
    n = np.bincount(codes)
    idx = rng.integers(0, len(s), size=(b, len(s)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    m = s.sum() / n.sum()
    return m, boot.std(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), np.mean(boot <= 0)


def setup(seed):
    d = E.setup(seed=seed)
    tr, va = d['tr'], d['va']
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    d['keep'] = keep
    d['Yv'] = d['Y_cons'][va]
    d['cl'] = d['clean_like'][va]
    d['gv'] = d['groups'][va]
    return d


def correct(d, pred, mask):
    return (pred[mask] == d['Yv'][mask]).all(1).astype(float)


# ----------------------------------------------------------------------------------------------
def uncertainty_and_paired():
    section('Uncertainty: configuration-level SE and paired bootstrap for the key decisions (clean-like rows)')
    rng = np.random.default_rng(0)
    out = []
    for seed in (42, 7):
        d = setup(seed)
        s = '' if seed == 42 else f'-s{seed}'
        cl = d['cl']
        thr = lambda P: (P >= .5).astype(np.uint8)
        final = thr(load_p(f'E8-mcs5-l1{s}'))
        m, se, lo, hi, _ = cluster_boot(d['gv'][cl], correct(d, final, cl), rng)
        n_rows, n_cfg = cl.sum(), len(np.unique(d['gv'][cl]))
        naive = np.sqrt(m * (1 - m) / n_rows)
        print(f'seed {seed}: final model clean-like EMR {m:.4%} | config-level SE {se * 100:.3f} pt '
              f'(naive row SE {naive * 100:.3f}) | 95% CI [{lo:.4%}, {hi:.4%}] | {n_rows:,} rows in {n_cfg:,} configs')

        prev = d['Y_cons'][d['keep']].mean(0)
        rare_bil = prev < 0.002
        P8 = load_p(f'E8-mcs5-l1{s}')
        comps = [
            ('corruption filter k=4 vs none (mcs 20)', thr(load_p(f'E5-reg-mcs20{s}')), thr(load_p(f'E7-drop-k4{s}'))),
            ('min_child_samples 5 vs 20 (after filter)', thr(load_p(f'E7-drop-k4{s}')), final),
            ('Sprint 1 additive rules on vs off', final, rules(d, final)),
            ('min_child_samples 2 vs 5', final, thr(load_p(f'E8-mcs2-l1{s}'))),
            ('more capacity (200 trees/31 leaves) vs final', final, thr(load_p(f'E10-cap{s}'))),
            ('rare-column threshold 0.45 vs 0.5', final, (P8 >= np.where(rare_bil, 0.45, 0.5)).astype(np.uint8)),
            ('basket-size count feature vs final', final, thr(load_p(f'E11-npk{s}'))),
        ]
        for name, a, b in comps:
            delta = correct(d, b, cl) - correct(d, a, cl)
            dm, dse, dlo, dhi, p = cluster_boot(d['gv'][cl], delta, rng)
            out.append(dict(decision=name, seed=seed, delta_pt=dm * 100, ci_lo=dlo * 100, ci_hi=dhi * 100,
                            p_one_sided=p, rows_net=int(delta.sum())))
    t = pd.DataFrame(out)
    print('\npaired difference (B - A) in clean-like EMR, points; CI = 95% configuration bootstrap; '
          'p = share of bootstrap draws <= 0')
    print(t.round(3).to_string(index=False))
    return t


# ----------------------------------------------------------------------------------------------
def baselines():
    section('Baselines on the same metric (all / test-like / clean-like validation rows)')
    for seed in (42, 7):
        d = setup(seed)
        s = '' if seed == 42 else f'-s{seed}'
        va = d['va']
        Xu, Yu, _ = E.dedup_configs(d['X_train'][d['keep']], d['Y_cons'][d['keep']])
        # Rules floor: each BIL column = OR of CRM columns that bill it >= 90% of the time (support >= 5)
        Xf, Yf = Xu.astype(np.float32), Yu.astype(np.float32)
        inter = Xf.T @ Yf
        sup = Xf.sum(0)[:, None]
        cond = inter / np.maximum(sup, 1)
        trig = (cond >= 0.9) & (sup >= 5)
        Xv = d['X_train'][va].astype(np.float32)
        pred = ((Xv @ trig.astype(np.float32)) > 0).astype(np.uint8)
        E.log(f'E12-floor{s}', 'raw+consensus (corrupted rows dropped, k=4)',
              'Rules floor: OR of CRM columns with P(BIL|CRM) >= 0.9', {'split_seed': seed, 'min_support': 5},
              E.score(d, pred), 0, notes=f'{int(trig.any(0).sum())} BIL columns have at least one trigger')
        # k-NN (k=15), exactly as Sprint 2 E1, now also scored on the clean-like metric
        Xa, Ya, cnt = E.dedup_configs(d['X_train'][d['tr']], d['Y_cons'][d['tr']])
        idx, dist = knn_hamming(Xa, d['X_train'][va], k=15)
        w = (1.0 / (1.0 + dist)) * np.log1p(cnt[idx])
        pk = ((Ya[idx] * w[:, :, None]).sum(1) / w.sum(1, keepdims=True) >= 0.5).astype(np.uint8)
        E.log(f'E12-knn15{s}', 'raw+consensus', 'Hamming k-NN (k=15), re-scored on all metrics',
              {'k': 15, 'split_seed': seed}, E.score(d, pk), 0, notes='Sprint 2 E1 set-up')


# ----------------------------------------------------------------------------------------------
def harshness_segments_coverage():
    d = setup(42)
    va, cl, Yv = d['va'], d['cl'], d['Yv']
    P = load_p('E8-mcs5-l1')
    pred = (P >= .5).astype(np.uint8)

    section('EMR harshness (clean-like validation rows, final model)')
    err = (pred[cl] != Yv[cl]).sum(1)
    e = (pred[cl] != Yv[cl]).mean()
    print(f'bit error rate {e:.2e} -> expected wrong bits per row {e * 731:.4f}')
    print(f'EMR if bit errors were independent: (1 - e)^731 = {(1 - e) ** 731:.4%} | actual EMR {np.mean(err == 0):.4%}')
    print(f'rows with 2+ wrong bits: {np.mean(err >= 2):.2%} of rows but {err[err >= 2].sum() / err.sum():.1%} of wrong bits '
          f'-> errors cluster in few rows')

    section('Errors by customer segment (clean-like validation rows, final model)')
    X = d['X_train'][va][cl]
    crm = np.array(d['crm_cols'])
    pk = np.array([c.endswith('_PACK') for c in crm])
    rows = []

    def seg(name, labels):
        for lab in pd.unique(labels):
            m = labels == lab
            rows.append(dict(segment=name, value=lab, rows=int(m.sum()), share=m.mean(),
                             EMR=np.mean(err[m] == 0), share_of_wrong_rows=(err[m] > 0).sum() / max((err > 0).sum(), 1)))
    for g in ('SUBSCRIBER_STATUS', 'SUBSCRIBER_TYPE'):
        cols = [i for i, c in enumerate(crm) if c.startswith(f'CRM_{g}_ORIG_')]
        seg(g.lower(), np.array([c.split('_ORIG_')[1] for c in crm[cols]])[X[:, cols].argmax(1)])
    npk = X[:, pk].sum(1)
    seg('CRM packs per customer', pd.cut(npk, [0, 10, 15, 20, 1000], labels=['<= 10', '11-15', '16-20', '21+']).astype(str))
    seg('rare CRM packs', np.where(d['rare_count'][va][cl] > 0, '1-2', '0'))
    t = pd.DataFrame(rows).sort_values(['segment', 'rows'], ascending=[True, False])
    print(t.round(4).to_string(index=False))

    section('Diffuse billing items: how much of the business do they touch? (clean-like validation rows)')
    drv = pd.read_csv(os.path.join(data.DERIVED_DIR, 'sprint3_shap_drivers.csv'))
    diffuse = (drv.top1_share_of_shap <= 0.3).to_numpy()
    yt = Yv[cl]
    tp = ((yt == 1) & (pred[cl] == 1)).sum(0)
    fp = ((yt == 0) & (pred[cl] == 1)).sum(0)
    fn = ((yt == 1) & (pred[cl] == 0)).sum(0)
    f1 = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    weak = f1 < 0.5
    pos = yt.sum()
    print(f'diffuse columns (SHAP top-1 share <= 0.3): {diffuse.sum()} of 731 columns ({diffuse.mean():.0%}), '
          f'but only {yt[:, diffuse].sum() / pos:.2%} of billed items')
    print(f'columns with validation F1 < 0.5: {np.nansum(weak)} columns, {yt[:, weak].sum() / pos:.2%} of billed items')
    has = yt[:, weak].any(1)
    print(f'customers with at least one billed item in a weak (F1 < 0.5) column: {has.mean():.2%} | '
          f'their EMR {np.mean(err[has] == 0):.2%} vs {np.mean(err[~has] == 0):.2%} for the rest')
    print('note: "diffuse" (no dominant SHAP driver) also covers common bundle items explained by several '
          'correlated packs, so the F1-based measure is the one that describes what the model cannot verify')

    section('Does the grouped split mirror test? Nearest training configuration (Hamming distance)')
    Xtr_u, _, _ = E.dedup_configs(d['X_train'][d['tr']], d['Y_cons'][d['tr']])
    _, dv = knn_hamming(Xtr_u, d['X_train'][va], k=1)
    Xall_u, _, _ = E.dedup_configs(d['X_train'], d['Y_cons'])
    _, dt = knn_hamming(Xall_u, d['X_test'], k=1)
    band = lambda x: pd.Series(pd.cut(x[:, 0], [-1, 0, 1, 2, 3, 1000], labels=['0', '1', '2', '3', '4+'])).value_counts(normalize=True).sort_index()
    print(pd.DataFrame({'validation (vs train fold)': band(dv), 'test (vs all of train)': band(dt)}).round(4).to_string())


# ----------------------------------------------------------------------------------------------
def error_detection():
    section('Error detection: flag billing items that disagree with a confident prediction (seed-42 validation fold)')
    d = setup(42)
    va = d['va']
    P = load_p('E8-mcs5-l1')
    Yobs = d['Y_raw'][va]                  # what the billing system actually contains
    err_cell = Yobs != d['Y_cons'][va]     # known provisioning errors (only identifiable in repeated configs)
    err_row = d['is_noisy'][va]
    cl = d['cl']
    rows = []
    for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
        flag = np.abs(P - Yobs) > tau
        frow = flag.any(1)
        for pop, m in (('all rows', np.ones(len(va), bool)), ('clean-like rows', cl)):
            fr, er = frow[m], err_row[m]
            fc, ec = flag[m], err_cell[m]
            rows.append(dict(tau=tau, population=pop, alerts_per_10k=10_000 * fr.mean(),
                             row_precision=(fr & er).sum() / max(fr.sum(), 1), row_recall=(fr & er).sum() / max(er.sum(), 1),
                             cell_precision=(fc & ec).sum() / max(fc.sum(), 1), cell_recall=(fc & ec).sum() / max(ec.sum(), 1)))
    print(f'known erroneous rows in the fold: {err_row.sum()} ({err_row.mean():.2%}); in clean-like rows: {err_row[cl].sum()}')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f'\nsuspected-corrupted rows (not clean-like) in the fold: {(~cl).sum():,} ({(~cl).mean():.1%}); '
          f'flagged at tau=0.9: {(np.abs(P - Yobs) > 0.9).any(1)[~cl].mean():.1%} '
          f'(they are flagged by the CRM-side filter anyway: impossible categories / bursts of rare packs)')

    print('\nReview queue (clean-like customers): flag if any item disagrees (|p - y| > 0.5); order flagged customers by '
          '(1) rare-product customers last, (2) fewer disagreeing items first. Both rules come from earlier findings.')
    for seed in (42, 7):
        ds = setup(seed)
        s = '' if seed == 42 else f'-s{seed}'
        c = ds['cl']
        D = np.abs(load_p(f'E8-mcs5-l1{s}')[c] - ds['Y_raw'][ds['va']][c])
        e = ds['is_noisy'][ds['va']][c]
        flag = (D > 0.5).any(1)
        key = (~flag) * 1000 + (ds['rare_count'][ds['va']][c] > 0) * 100 + (D > 0.5).sum(1)
        order = np.lexsort((np.arange(len(e)), key))
        out = []
        for share in (0.01, 0.02, flag.mean()):
            k = int(share * len(e))
            out.append(f'top {share:.1%}: caught {e[order[:k]].sum() / e.sum():.0%}, precision {e[order[:k]].mean():.0%}, '
                       f'lift {e[order[:k]].mean() / e.mean():.0f}x')
        print(f'  seed {seed} ({e.sum()} known errors, base rate {e.mean():.2%}): ' + ' | '.join(out))


# ----------------------------------------------------------------------------------------------
def queue(D, rare):
    """Review queue: flagged customers first; rare-product customers last; fewer disagreeing items first."""
    dis = D > 0.5
    flag = dis.any(1)
    key = (~flag) * 1000 + rare * 100 + dis.sum(1)
    return np.lexsort((np.arange(len(D)), key)), flag


def ratio_boot(groups, num, den, rng, b=B):
    codes, _ = pd.factorize(groups)
    sn, sd = np.bincount(codes, weights=num), np.bincount(codes, weights=den)
    idx = rng.integers(0, len(sn), size=(b, len(sn)))
    boot = sn[idx].sum(1) / np.maximum(sd[idx].sum(1), 1)
    return np.percentile(boot, [2.5, 97.5])


def detection_extended(reps=5):
    section('Error detection, extended: recall CIs, synthetic injected errors, illustrative costs')
    rng = np.random.default_rng(0)
    for seed in (42, 7):
        d = setup(seed)
        s = '' if seed == 42 else f'-s{seed}'
        va, cl = d['va'], d['cl']
        P = load_p(f'E8-mcs5-l1{s}')[cl]
        Yobs = d['Y_raw'][va][cl]
        e = d['is_noisy'][va][cl]
        g = d['gv'][cl]
        rare = d['rare_count'][va][cl] > 0
        D = np.abs(P - Yobs)
        order, flag = queue(D, rare)
        n = len(e)
        print(f'\n--- seed {seed}: {n:,} clean-like customers, {e.sum()} known errors')
        for share in (0.01, 0.02):
            top = np.zeros(n, bool)
            top[order[:int(share * n)]] = True
            lo, hi = ratio_boot(g, (e & top).astype(float), e.astype(float), rng)
            print(f'  known errors caught in top {share:.0%}: {(e & top).sum() / e.sum():.0%} '
                  f'(95% configuration-bootstrap CI {lo:.0%}-{hi:.0%}); precision {e[top].mean():.0%}')

        # Synthetic injection on customers that are not known errors (their observed billing = consensus).
        clean = np.flatnonzero(~e)
        prev = d['Y_cons'][d['keep']].mean(0)
        val_count = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
        unique_cfg = val_count == 1
        print(f'  synthetic test: {unique_cfg[clean].mean():.0%} of these customers have a configuration seen once in '
              f'the fold (like every test customer)')
        rows = []
        for kind in ('missing item', 'extra item'):
            for k in (1, 2, 3):
                rec2, rec_all, rec_u, rec_r, prec2 = [], [], [], [], []
                for _ in range(reps):
                    S = rng.choice(clean, size=max(1, int(0.0065 * n)), replace=False)
                    Ymod = Yobs.copy()
                    for i in S:
                        pool = np.flatnonzero(Ymod[i] == (1 if kind == 'missing item' else 0))
                        if kind == 'missing item':
                            pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
                        else:
                            w = prev[pool] / prev[pool].sum()
                            pick = rng.choice(pool, size=k, replace=False, p=w)
                        Ymod[i, pick] = 1 - Ymod[i, pick]
                    Dm = D.copy()
                    Dm[S] = np.abs(P[S] - Ymod[S])
                    o, f = queue(Dm, rare)
                    inj = np.zeros(n, bool)
                    inj[S] = True
                    top = np.zeros(n, bool)
                    top[o[:int(0.02 * n)]] = True
                    rec2.append(top[S].mean())
                    rec_all.append(f[S].mean())
                    rec_u.append(top[S][unique_cfg[S]].mean() if unique_cfg[S].any() else np.nan)
                    rec_r.append(top[S][~unique_cfg[S]].mean() if (~unique_cfg[S]).any() else np.nan)
                    prec2.append(inj[top].mean())
                rows.append(dict(error=kind, n_bits=k, flagged=np.mean(rec_all), caught_top2pct=np.mean(rec2),
                                 caught_unique_cfg=np.nanmean(rec_u), caught_repeated_cfg=np.nanmean(rec_r),
                                 precision_top2pct=np.mean(prec2)))
        print(pd.DataFrame(rows).round(3).to_string(index=False))

        # Illustrative operating point from assumed costs (known-error curve).
        caught = np.cumsum(e[order]) / e.sum()
        errors_per_10k = 10_000 * e.mean()
        c_alert = 2.5  # assumed: 5 analyst-minutes at EUR 30/h
        shares = np.linspace(0.0025, 0.06, 232)
        for c_miss in (20, 50, 200):
            cost = [sh * 10_000 * c_alert + (1 - caught[int(sh * n) - 1]) * errors_per_10k * c_miss for sh in shares]
            j = int(np.argmin(cost))
            print(f'  assumed cost of a missed error EUR {c_miss:>3}: cheapest review share {shares[j]:.1%} '
                  f'(catches {caught[int(shares[j] * n) - 1]:.0%}; EUR {cost[j]:.0f} per 10,000 customers)')


# ----------------------------------------------------------------------------------------------
def test_ci():
    section('Test EMR with 95% confidence intervals (final model, no rules; solution.csv used for scoring only)')
    d = data.load()
    P = load_p('test_sprint3_k4_mcs5')
    pred = (P >= .5).astype(np.uint8)
    sol = pd.read_csv(os.path.join(data.DATA_DIR, 'solution.csv'), dtype=str).set_index('MSISDN').loc[d['msisdn_test'], 'Bill_Conf']
    Yt = np.array([np.frombuffer(s.encode(), np.uint8) - 48 for s in sol])
    c = (pred == Yt).all(1).astype(float)
    m = c.mean()
    se = np.sqrt(m * (1 - m) / len(c))
    g, _ = V.factorize_rows(d['X_test'])
    cm, cse, lo, hi, _ = cluster_boot(g, c, np.random.default_rng(1))
    print(f'test EMR {m:.4%} | binomial 95% CI [{m - 1.96 * se:.4%}, {m + 1.96 * se:.4%}] | '
          f'configuration bootstrap 95% CI [{lo:.4%}, {hi:.4%}] ({len(np.unique(g)):,} test configurations)')
    e = (pred != Yt).mean()
    print(f'test bit error rate {e:.2e}: EMR if independent {(1 - e) ** 731:.4%} vs actual {m:.4%}')
    print(f'gap to the ~98% benchmark: {(0.98 - m) * len(c):.0f} rows')


if __name__ == '__main__':
    parts = sys.argv[1:] or ['paired', 'baselines', 'harsh', 'detect', 'testci']
    if 'paired' in parts:
        uncertainty_and_paired()
    if 'baselines' in parts:
        baselines()
    if 'harsh' in parts:
        harshness_segments_coverage()
    if 'detect' in parts:
        error_detection()
    if 'detect2' in parts:
        detection_extended()
    if 'testci' in parts:
        test_ci()
