"""Diag 4 - "one CRM pack too many, same billing": customers whose CRM carries a pack that is
normally billed, but whose BIL lacks every item that pack is billed as.

Professor's hint (2026-10-07): employees are suspected of adding CRM packs for some customers
without the matching billing (a more expensive pack, billing unchanged). Model-free:

A. Billing map: on the rows kept for training (test-like, < 4 rare packs), CRM pack p is
   billed as BIL item b if P(b | p) >= CONF and lift P(b | p) / P(b) >= LIFT.
B. Flag (row, p) when p is active and all of p's billing items are 0 in the row's BIL.
   Per-pack flag rate = flagged rows / rows with p, in kept and in dropped rows.
C. Confirmation by lookup: the CRM configuration without p exists in train (kept rows) and
   its consensus BIL equals this row's BIL exactly -> "same billing as if p were absent".

Uses train.csv only (no test labels, no solution.csv).
D. Uniformity of flags across packs; E. reverse direction (billed items with no CRM pack that
   is billed as them) and pack counts per row; F. one line per customer.

Outputs: data/derived/unbilled_crm_pack_rates.csv (per pack), unbilled_crm_suspects.csv
(per customer x pack), unbilled_crm_customers.csv (per customer).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src import data, validation

CONF, LIFT, MIN_N = 0.8, 3.0, 30


def main():
    d = data.load()
    X, Y = d['X_train'], d['Y_raw']
    crm, bil = np.array(d['crm_cols']), np.array(d['bil_cols'])
    msisdn = d['msisdn_train']
    test_like = validation.test_like_mask(X, d['crm_cols'])
    rare = validation.rare_pack_count(X, d['crm_cols'])
    kept = test_like & (rare < 4)
    print(f'rows: {len(X):,}  kept {kept.sum():,}  dropped {(~kept).sum():,}')

    # A. billing map on kept rows
    Xk, Yk = X[kept].astype(np.float32), Y[kept].astype(np.float32)
    n_p = Xk.sum(0)
    co = Xk.T @ Yk                                    # rows with p and b
    conf = co / np.maximum(n_p, 1)[:, None]
    base = Yk.mean(0)
    lift = conf / np.maximum(base, 1e-9)[None, :]
    is_pack = np.char.endswith(crm, '_PACK')
    partner = (conf >= CONF) & (lift >= LIFT) & (n_p >= MIN_N)[:, None] & is_pack[:, None]
    billable = np.where(partner.any(1))[0]
    print(f'CRM packs with >= {MIN_N} kept rows: {(is_pack & (n_p >= MIN_N)).sum()}; '
          f'billable (>= 1 partner item): {len(billable)}')

    # B. flags: p active, all partner items 0
    rows, packs, missing = [], [], []
    stats = []
    for p in billable:
        items = np.where(partner[p])[0]
        has = X[:, p] == 1
        viol = has & (Y[:, items].sum(1) == 0)
        idx = np.where(viol)[0]
        rows.append(idx)
        packs.append(np.full(len(idx), p))
        missing.append(np.full(len(idx), ';'.join(bil[items])))
        stats.append(dict(crm_pack=crm[p], billed_as=';'.join(bil[items]),
                          conf_min=round(float(conf[p, items].min()), 4),
                          n_kept=int((has & kept).sum()), flagged_kept=int((viol & kept).sum()),
                          n_dropped=int((has & ~kept).sum()), flagged_dropped=int((viol & ~kept).sum())))
    rows, packs, missing = np.concatenate(rows), np.concatenate(packs), np.concatenate(missing)
    st = pd.DataFrame(stats)
    st['rate_kept'] = st.flagged_kept / st.n_kept.clip(lower=1)
    st['rate_dropped'] = st.flagged_dropped / st.n_dropped.clip(lower=1)

    # C. lookup confirmation: config without p, consensus BIL of kept rows
    codes, _ = validation.factorize_rows(X)
    Ycons, _ = validation.consensus_targets(codes, Y)
    keys = validation.row_keys(X)
    cons_by_key = {}
    for i in np.where(kept)[0]:
        cons_by_key.setdefault(keys[i], i)
    Xr = X[rows].copy()
    Xr[np.arange(len(rows)), packs] = 0
    red_keys = validation.row_keys(Xr)
    found = np.array([k in cons_by_key for k in red_keys])
    same = np.array([k in cons_by_key and (Ycons[cons_by_key[k]] == Y[r]).all()
                     for k, r in zip(red_keys, rows)])

    sus = pd.DataFrame(dict(row=rows, MSISDN=msisdn[rows], crm_pack=crm[packs], missing_bil=missing,
                            kept_for_training=kept[rows], n_rare_packs=rare[rows],
                            reduced_config_in_train=found, same_billing_as_without_pack=same))
    sus['n_flags_in_row'] = sus.groupby('row')['row'].transform('size')
    conf_rate = sus.groupby('crm_pack').agg(lookup_found=('reduced_config_in_train', 'sum'),
                                            lookup_same=('same_billing_as_without_pack', 'sum'))
    st = st.merge(conf_rate, left_on='crm_pack', right_index=True, how='left').fillna(0)
    st = st.sort_values('flagged_kept', ascending=False)

    out = data.DERIVED_DIR
    st.to_csv(os.path.join(out, 'unbilled_crm_pack_rates.csv'), index=False)
    sus.sort_values(['kept_for_training', 'n_flags_in_row'], ascending=[False, True]) \
        .to_csv(os.path.join(out, 'unbilled_crm_suspects.csv'), index=False)

    pd.set_option('display.width', 220, 'display.max_columns', 20, 'display.max_colwidth', 40)
    print('\n== flags ==')
    print(f'(row, pack) flags: {len(sus):,}; distinct rows: {sus.row.nunique():,} '
          f'(kept {sus[sus.kept_for_training].row.nunique():,}, dropped {sus[~sus.kept_for_training].row.nunique():,})')
    print(f'reduced config found in train: {found.sum():,}; same billing as without pack: {same.sum():,}')
    k = st[st.n_kept >= MIN_N]
    tot_rate = k.flagged_kept.sum() / k.n_kept.sum()
    print(f'pooled flag rate on kept rows: {tot_rate:.3%}; median per-pack rate {k.rate_kept.median():.3%}')
    print(f'top-10 packs hold {k.flagged_kept.head(10).sum() / max(k.flagged_kept.sum(), 1):.1%} of kept flags '
          f'({len(k)} billable packs)')
    print('\n== top 25 packs by flagged kept rows ==')
    print(k.head(25)[['crm_pack', 'billed_as', 'conf_min', 'n_kept', 'flagged_kept', 'rate_kept',
                      'n_dropped', 'rate_dropped', 'lookup_found', 'lookup_same']].to_string(index=False))
    print('\n== top 15 packs by kept flag rate (n_kept >= 200) ==')
    print(k[k.n_kept >= 200].sort_values('rate_kept', ascending=False).head(15)[
        ['crm_pack', 'billed_as', 'n_kept', 'flagged_kept', 'rate_kept', 'rate_dropped', 'lookup_same']].to_string(index=False))
    print('\nflags per row (kept):', sus[sus.kept_for_training].groupby('row').size().value_counts().sort_index().to_dict())

    # D. uniformity: does the flag count per pack scale with how common the pack is?
    st['flagged_all'] = st.flagged_kept + st.flagged_dropped
    print(f"\n== uniformity ==\nflagged rows per pack (kept + dropped): mean {st.flagged_all.mean():.0f}, "
          f"sd {st.flagged_all.std():.0f}, range {st.flagged_all.min()}-{st.flagged_all.max()} "
          f"while rows per pack range {st.n_kept.min():,}-{st.n_kept.max():,}")
    st['size_bin'] = pd.qcut(st.n_kept, 5)
    print(st.groupby('size_bin', observed=True)['flagged_all'].agg(['mean', 'std']).round(0).to_string())

    # E. reverse direction: billed items with no CRM pack that is billed as them ("orphan" BIL)
    is_bpack = np.char.endswith(bil, '_PACK')
    orphan = np.zeros(len(X), int)
    orphan_items = [[] for _ in range(len(X))]
    for b in np.where(partner.any(0) & is_bpack)[0]:
        o = np.where((Y[:, b] == 1) & (X[:, np.where(partner[:, b])[0]].max(1) == 0))[0]
        orphan[o] += 1
        for r in o:
            orphan_items[r].append(bil[b])
    unbilled = np.bincount(rows, minlength=len(X))
    for name, m in [('kept', kept), ('dropped', ~kept)]:
        t = pd.crosstab(np.clip(unbilled[m], 0, 4), np.clip(orphan[m], 0, 4))
        t.index.name, t.columns.name = 'unbilled CRM packs (4 = 4+)', 'orphan BIL items'
        print(f'\n== {name} rows ({m.sum():,}) ==\n{t.to_string()}')
    cp = X[:, np.char.endswith(crm, '_PACK')].sum(1)
    bp = Y[:, is_bpack].sum(1)
    g = pd.DataFrame(dict(rare=np.clip(rare, 0, 8), crm=cp, bil=bp)).groupby('rare').mean().round(2)
    print('\nmean packs per row by # rare CRM packs (additions, not swaps, if both grow):\n' + g.to_string())

    # F. one line per customer
    by_row = sus.groupby('row').agg(unbilled_crm_packs=('crm_pack', ';'.join))
    cust = pd.DataFrame(dict(row=by_row.index, MSISDN=msisdn[by_row.index],
                             n_unbilled_crm=unbilled[by_row.index],
                             unbilled_crm_packs=by_row.unbilled_crm_packs.to_numpy(),
                             n_orphan_bil=orphan[by_row.index],
                             orphan_bil_items=[';'.join(orphan_items[r]) for r in by_row.index],
                             kept_for_training=kept[by_row.index], n_rare_packs=rare[by_row.index]))
    cust['pattern'] = np.where(cust.n_orphan_bil == 0, 'CRM extra, billing unchanged',
                               'CRM extra + billing extra')
    cust = cust.sort_values(['pattern', 'n_unbilled_crm'], ascending=[True, False])
    cust.to_csv(os.path.join(out, 'unbilled_crm_customers.csv'), index=False)
    print(f'\n== customers ({len(cust):,} = {len(cust) / len(X):.1%} of train) ==')
    print(pd.crosstab(cust.pattern, cust.kept_for_training, margins=True).to_string())


if __name__ == '__main__':
    main()
