"""Diagnostic 1 - audit column encoding, categorical blocks, pack-id structure and where the
champion's validation errors live. Read-only: trains nothing, writes nothing.

Run from the repo root:  .venv/Scripts/python scripts/sprint3/diag1_audit.py > data/cache/diag1.txt
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import data, experiment as E, metrics as M

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

CAT = re.compile(r'^(CRM|BIL)_(.+?)_ORIG_(.+)$')
PACK = re.compile(r'^(CRM|BIL)_(\d+)_PACK$')


def taxonomy(cols):
    rows = []
    for i, c in enumerate(cols):
        m = CAT.match(c)
        if m:
            rows.append(dict(i=i, col=c, kind='cat', group=m.group(2), value=m.group(3), pack_id=-1))
            continue
        m = PACK.match(c)
        if m:
            rows.append(dict(i=i, col=c, kind='pack', group='PACK', value='', pack_id=int(m.group(2))))
            continue
        rows.append(dict(i=i, col=c, kind='other', group='?', value='', pack_id=-1))
    return pd.DataFrame(rows)


def section(title):
    print('\n' + '=' * 100 + f'\n{title}\n' + '=' * 100)


def group_cols(t, g):
    return t[(t.kind == 'cat') & (t.group == g)]


def group_labels(A, t):
    """Value name per row for one categorical group; '<none>' / '<multi>' when not one-hot."""
    sub = A[:, t.i.to_numpy()]
    s = sub.sum(1)
    lab = np.array(t.value.tolist(), dtype=object)[sub.argmax(1)]
    return np.where(s == 0, '<none>', np.where(s > 1, '<multi>', lab))


def main():
    d = E.setup(seed=42)
    tr, va = d['tr'], d['va']
    Xtr, Xte, Yraw, Y = d['X_train'], d['X_test'], d['Y_raw'], d['Y_cons']
    tc, tb = taxonomy(d['crm_cols']), taxonomy(d['bil_cols'])
    groups = sorted(set(tc[tc.kind == 'cat'].group) | set(tb[tb.kind == 'cat'].group))

    # ------------------------------------------------------------------------------------------
    section('D1  Column taxonomy and one-hot validity')
    for name, t in [('CRM', tc), ('BIL', tb)]:
        p = t[t.kind == 'pack'].pack_id
        print(f"{name}: {t['kind'].value_counts().to_dict()} | pack ids {p.min()}..{p.max()} "
              f"({len(p)} packs, {p.max() - p.min() + 1 - len(p)} ids unused in that range)")
        other = t[t.kind == 'other'].col.tolist()
        if other:
            print(f'  unparsed columns: {other}')

    sources = [('CRM train', tc, Xtr), ('CRM test', tc, Xte), ('BIL train', tb, Yraw)]
    for g in groups:
        vals = sorted(set(group_cols(tc, g).value) | set(group_cols(tb, g).value))
        rows = []
        for v in vals:
            r = {'value': v}
            for nm, t, A in sources:
                hit = group_cols(t, g)
                hit = hit[hit.value == v]
                r[nm] = A[:, hit.i.iloc[0]].mean() if len(hit) else np.nan
            rows.append(r)
        print(f'\n[{g}] share of rows per value (NaN = column does not exist on that side)')
        print(pd.DataFrame(rows).round(5).to_string(index=False))
        for nm, t, A in sources:
            idx = group_cols(t, g).i.to_numpy()
            if len(idx) == 0:
                continue
            s = A[:, idx].sum(1)
            print(f'  {nm:9s}: {len(idx):2d} cols | rows with 0 active {np.mean(s == 0):8.4%} | '
                  f'exactly 1 {np.mean(s == 1):8.4%} | 2+ {np.mean(s >= 2):8.4%}')

    # ------------------------------------------------------------------------------------------
    section('D2  Is each BIL categorical block a copy of the CRM block? (train, consensus labels)')
    for g in groups:
        ci, bi = group_cols(tc, g), group_cols(tb, g)
        if len(ci) == 0 or len(bi) == 0:
            print(f'[{g}] only on one side (CRM {len(ci)} cols, BIL {len(bi)} cols)')
            continue
        cl, bl = group_labels(Xtr, ci), group_labels(Y, bi)
        print(f'\n[{g}] CRM value == BIL value on {np.mean(cl == bl):.4%} of train rows')
        print(pd.crosstab(pd.Series(cl, name='CRM \\ BIL'), pd.Series(bl, name='BIL')).to_string())

    # ------------------------------------------------------------------------------------------
    section('D3  Champion validation predictions: where the errors live')
    P = np.load(os.path.join(data.DATA_DIR, 'cache', 'proba_E5-reg-mcs20.npy')).astype(np.float32)
    Xv, Yv = Xtr[va], Y[va]
    pred = data.apply_additive_rules((P >= 0.5).astype(np.uint8), Xv, d['crm_cols'], d['bil_cols'])
    print(f'reproduced champion validation EMR: {M.exact_match_ratio(Yv, pred):.4%}')
    diff = pred != Yv
    catm, packm = (tb.kind == 'cat').to_numpy(), (tb.kind == 'pack').to_numpy()
    wrong = diff.any(1)
    only_cat = wrong & ~diff[:, packm].any(1)
    only_pack = wrong & ~diff[:, catm].any(1)
    print(f'rows wrong {wrong.mean():.4%} | only categorical cols wrong {only_cat.mean():.4%} | '
          f'only pack cols wrong {only_pack.mean():.4%} | both {np.mean(wrong & ~only_cat & ~only_pack):.4%}')
    print(f'wrong cells: categorical {int(diff[:, catm].sum()):,} | pack {int(diff[:, packm].sum()):,}')
    for g in groups:
        idx = group_cols(tb, g).i.to_numpy()
        if len(idx) == 0:
            continue
        s = pred[:, idx].sum(1)
        print(f'[{g}] predicted rows with 0 values {np.mean(s == 0):.4%} | 2+ values {np.mean(s >= 2):.4%} '
              f'| group wrong in {diff[:, idx].any(1).mean():.4%} of rows')

    print('\nWhat-if checks (diagnostic only, not a model change):')
    fix = pred.copy()
    for g in groups:
        idx = group_cols(tb, g).i.to_numpy()
        if len(idx) >= 2:
            fix[:, idx] = 0
            fix[np.arange(len(fix)), idx[P[:, idx].argmax(1)]] = 1
    print(f'  force exactly one value per BIL group (argmax prob): EMR {M.exact_match_ratio(Yv, fix):.4%}')
    cp = pred.copy()
    for g in groups:
        ci, bi = group_cols(tc, g), group_cols(tb, g)
        for _, r in bi.iterrows():
            hit = ci[ci.value == r.value]
            if len(hit):
                cp[:, r.i] = Xv[:, hit.i.iloc[0]]
    print(f'  copy CRM categorical values into BIL:               EMR {M.exact_match_ratio(Yv, cp):.4%}')
    pk = pred.copy()
    pk[:, catm] = Yv[:, catm]
    print(f'  ceiling if every categorical column were perfect:   EMR {M.exact_match_ratio(Yv, pk):.4%}')
    ct = pred.copy()
    ct[:, packm] = Yv[:, packm]
    print(f'  ceiling if every pack column were perfect:          EMR {M.exact_match_ratio(Yv, ct):.4%}')

    # ------------------------------------------------------------------------------------------
    section('D4  Columns that alone break an otherwise perfect validation row')
    top = M.top_offending_columns(Yv, pred, d['bil_cols'], top_n=20)
    top['kind'] = top['bil_column'].map(dict(zip(tb.col, tb.kind)))
    print(top.round(4).to_string(index=False))

    # ------------------------------------------------------------------------------------------
    section('D5  Hidden structure in the numeric pack ids (train fold)')
    cpk, bpk = tc[tc.kind == 'pack'], tb[tb.kind == 'pack']
    Xp = Xtr[tr][:, cpk.i.to_numpy()].astype(np.float32)
    Yp = Y[tr][:, bpk.i.to_numpy()].astype(np.float32)
    inter = Xp.T @ Yp
    nx, ny = Xp.sum(0), Yp.sum(0)
    score = np.minimum(inter / np.maximum(nx[:, None], 1), inter / np.maximum(ny[None, :], 1))
    pd_ = pd.DataFrame({'crm_id': cpk.pack_id.to_numpy(), 'n_rows': nx.astype(int),
                        'best_bil_id': bpk.pack_id.to_numpy()[score.argmax(1)],
                        'score': score.max(1).round(3),
                        'second_score': np.sort(score, 1)[:, -2].round(3)}).sort_values('crm_id')
    seen = pd_[pd_.n_rows >= 20].reset_index(drop=True)
    bands = pd.cut(seen.score, [-0.01, 0.2, 0.5, 0.8, 0.95, 1.0]).value_counts().sort_index()
    print(f'CRM packs with >=20 train rows: {len(seen)} | best-partner score bands:')
    print(bands.to_string())
    print(f'BIL packs that are best partner of more than one CRM pack: '
          f'{int((seen.best_bil_id.value_counts() > 1).sum())}')
    same = (seen.best_bil_id.to_numpy()[1:] == seen.best_bil_id.to_numpy()[:-1]) & \
           (np.diff(seen.crm_id.to_numpy()) <= 2)
    print(f'consecutive CRM ids (gap <= 2) sharing the same best BIL partner: {int(same.sum())} of {len(seen) - 1}')
    strong = seen[seen.score >= 0.8]
    print(f'Spearman(CRM id, best BIL id) over {len(strong)} strong pairs: '
          f'{spearmanr(strong.crm_id, strong.best_bil_id).statistic:+.3f}')
    print('\nfirst 40 CRM packs by id:')
    print(seen.head(40).to_string(index=False))
    bil_best = score.max(0)
    print(f'\nBIL packs (>=20 rows) with no CRM partner scoring >= 0.5: '
          f'{int(((ny >= 20) & (bil_best < 0.5)).sum())} of {int((ny >= 20).sum())}')

    # ------------------------------------------------------------------------------------------
    section('D6  Pipeline sanity checks')
    Xf = data.features(d, 'raw')
    print(f'champion feature matrix: {Xf.shape[1]} columns, dtype {Xf.dtype} '
          f'(raw CRM only, no PCA/SVD: {Xf.shape[1] == len(d["crm_cols"])})')
    on_tr, on_te = Xtr.any(0), Xte.any(0)
    crm = np.array(d['crm_cols'])
    print(f'CRM columns never active in train but active in test: {crm[~on_tr & on_te].tolist()} '
          f'(test rows affected: {int(Xte[:, ~on_tr & on_te].any(1).sum())})')
    print(f'CRM columns active in train but never in test: {int((on_tr & ~on_te).sum())}')
    print(f'BIL columns never active in train (consensus): {np.array(d["bil_cols"])[~Y.any(0)].tolist()}')
    keys = pd.Series([c.tobytes() for c in np.packbits(np.ascontiguousarray(Xtr.T), axis=1)])
    dup = keys.duplicated(keep=False)
    print(f'CRM columns that are exact duplicates of another column (train): {crm[dup.to_numpy()].tolist()}')

    cpm = (tc.kind == 'pack').to_numpy()
    n_crm = Xtr[:, cpm].sum(1).astype(int)
    n_bil = Y[:, packm].sum(1).astype(int)
    gap = n_bil - n_crm
    print('\n#BIL packs - #CRM packs per train row:', pd.Series(gap).value_counts().head(8).to_dict())
    m2 = gap == -2
    lift = pd.DataFrame({'crm_pack': crm[cpm], 'rate_in_gap-2_rows': Xtr[m2][:, cpm].mean(0),
                         'rate_in_gap0_rows': Xtr[gap == 0][:, cpm].mean(0)})
    lift['diff'] = lift['rate_in_gap-2_rows'] - lift['rate_in_gap0_rows']
    print('CRM packs most over-represented in rows billing 2 fewer packs than listed:')
    print(lift.sort_values('diff', ascending=False).head(10).round(4).to_string(index=False))

    # ------------------------------------------------------------------------------------------
    section('D7  Test-like rows: exactly one status/type/business line and no "0" value')

    def test_like(A):
        ok = np.ones(len(A), bool)
        for g in groups:
            h = group_cols(tc, g)
            ok &= A[:, h.i.to_numpy()].sum(1) == 1
            zero = h[h.value == '0']
            if len(zero):
                ok &= A[:, zero.i.iloc[0]] == 0
        return ok

    ok_tr, ok_te = test_like(Xtr), test_like(Xte)
    print(f'test-like rows: train {ok_tr.mean():.4%} | test {ok_te.mean():.4%}')
    print(f'mean #CRM packs: non-test-like train rows {Xtr[~ok_tr][:, cpm].sum(1).mean():.2f} | '
          f'test-like train rows {Xtr[ok_tr][:, cpm].sum(1).mean():.2f} | test {Xte[:, cpm].sum(1).mean():.2f}')
    err = M.row_errors(Yv, pred)
    a = ok_tr[va]
    print(f'champion val EMR: test-like rows {np.mean(err[a] == 0):.4%} ({a.sum():,}) | '
          f'other rows {np.mean(err[~a] == 0):.4%} ({(~a).sum():,})')
    print(f'share of 6+-bit val rows that are not test-like: {np.mean(~a[err >= 6]):.2%}')
    freq = Xtr[tr].sum(0)
    has_rare = Xv[:, (freq > 0) & (freq < 200)].any(1)
    print(f'val rows with a CRM pack seen in <200 train rows: {has_rare.mean():.2%} | '
          f'median #CRM packs there {np.median(Xv[has_rare][:, cpm].sum(1)):.0f} vs test {np.median(Xte[:, cpm].sum(1)):.0f}')


if __name__ == '__main__':
    main()
