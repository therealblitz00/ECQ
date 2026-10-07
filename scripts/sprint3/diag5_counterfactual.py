"""Diag 5 - model-based fraud check: switch each active CRM pack from 1 to 0 and see whether
the Stage 1 prediction then matches the customer's actual billing.

Follows Step 17 (diag4_unbilled_crm.py), which only covers packs with one clear billing item.
Here the Stage 1 model (per-label LightGBM, mcs=5, lambda=1, unique configs + consensus labels,
corrupted rows dropped) supplies the "expected billing" of any CRM configuration.

1. Out-of-fold predictions: 5-fold GroupKFold on the exact CRM configuration over all train
   rows; each fold's model is trained on the kept rows (test-like, < 4 rare packs) of the other
   folds, so no customer is scored by a model that saw its configuration.
2. Missing billing: items predicted 1 but billed 0. For every customer with at least one, and
   every active CRM pack p, predict again with p = 0. p is a *culprit* if that removes at least
   one missing item and creates no new disagreement. Then all culprits are removed together:
   if the prediction now equals the actual billing exactly, the customer is "CRM extra,
   billing unchanged" (model-verified).
3. Score = highest predicted probability among missing items explained by a culprit.
4. Synthetic check: on kept clean rows that the model gets exactly right, add one random CRM
   pack (billing unchanged) and measure how often the added pack is found.

Uses train.csv only (no test labels, no solution.csv).
Outputs: data/derived/fraud_model_customers.csv, fraud_model_packs.csv.
Usage: python scripts/sprint3/diag5_counterfactual.py [n_synthetic_per_fold] [quick]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src import data, validation, experiment as E
from src.models import BinaryRelevanceLGBM

STAGE1 = dict(min_child_samples=5, reg_lambda=1.0)
THR = 0.5


def counterfactual(model, X, Y, P, pack_cols):
    """Rows of (X, Y, P) with missing billing: try removing each active CRM pack.
    Returns {local row index: dict(culprits, fixed, exact_after, score)}."""
    B = P >= THR
    miss = B & (Y == 0)
    wrong = B != (Y == 1)
    cand = np.where(miss.any(1))[0]
    pairs = [(r, p) for r in cand for p in pack_cols if X[r, p]]
    if not pairs:
        return {}
    pr = np.array([r for r, _ in pairs])
    pp = np.array([p for _, p in pairs])
    out = {}
    for s in range(0, len(pairs), 50_000):
        r, p = pr[s:s + 50_000], pp[s:s + 50_000]
        Xm = X[r].copy()
        Xm[np.arange(len(r)), p] = 0
        wrong_m = (model.predict_proba(Xm) >= THR) != (Y[r] == 1)
        fixed = miss[r] & ~wrong_m
        ok = fixed.any(1) & ~(~wrong[r] & wrong_m).any(1)
        for i in np.where(ok)[0]:
            o = out.setdefault(int(r[i]), dict(culprits=[], fixed=set()))
            o['culprits'].append(int(p[i]))
            o['fixed'] |= set(np.where(fixed[i])[0].tolist())
    # remove all culprits together
    rr = list(out)
    if rr:
        Xa = X[rr].copy()
        for i, r in enumerate(rr):
            Xa[i, out[r]['culprits']] = 0
        exact = ((model.predict_proba(Xa) >= THR) == (Y[rr] == 1)).all(1)
        for i, r in enumerate(rr):
            out[r]['exact_after'] = bool(exact[i])
            out[r]['score'] = float(P[r, sorted(out[r]['fixed'])].max())
    return out


def main(n_syn=1000, quick=''):
    n_syn = int(n_syn)
    rng = np.random.default_rng(0)
    d = data.load()
    X, Y = d['X_train'], d['Y_raw']
    if quick:  # smoke test: first 20k rows, nothing written to data/derived
        X, Y = X[:20_000], Y[:20_000]
        d['msisdn_train'] = d['msisdn_train'][:20_000]
        data.DERIVED_DIR = os.path.join(data.DATA_DIR, 'cache')
    crm, bil = np.array(d['crm_cols']), np.array(d['bil_cols'])
    pack_cols = np.where(np.char.endswith(crm, '_PACK'))[0]
    groups, _ = validation.factorize_rows(X)
    Y_cons, _ = validation.consensus_targets(groups, Y)
    test_like = validation.test_like_mask(X, d['crm_cols'])
    rare = validation.rare_pack_count(X, d['crm_cols'])
    kept = test_like & (rare < 4)
    clean0 = kept & (rare == 0)

    P_oof = np.zeros(Y.shape, np.float16)
    found, syn = {}, []
    for f, (tr, te) in enumerate(validation.grouped_kfold(groups, 5)):
        t0 = time.time()
        trk = tr[kept[tr]]
        Xu, Yu, _ = E.dedup_configs(X[trk], Y_cons[trk])
        model = BinaryRelevanceLGBM(STAGE1).fit(Xu, Yu)
        P = model.predict_proba(X[te])
        P_oof[te] = P
        res = counterfactual(model, X[te], Y[te], P, pack_cols)
        found.update({int(te[i]): v for i, v in res.items()})

        # synthetic: clean rows predicted exactly right, one random CRM pack added
        ok = np.where(clean0[te] & ((P >= THR) == (Y[te] == 1)).all(1))[0]
        pick = te[rng.choice(ok, size=min(n_syn, len(ok)), replace=False)]
        Xs = X[pick].copy()
        add = np.array([rng.choice(pack_cols[x[pack_cols] == 0]) for x in Xs])
        Xs[np.arange(len(pick)), add] = 1
        Ps = model.predict_proba(Xs)
        res = counterfactual(model, Xs, Y[pick], Ps, pack_cols)
        changes = ((Ps >= THR) != (Y[pick] == 1)).any(1)
        for i in range(len(pick)):
            o = res.get(i)
            syn.append(dict(fold=f, added=crm[add[i]], billing_changes=bool(changes[i]),
                            found=o is not None and int(add[i]) in o['culprits'],
                            flagged=o is not None, exact_after=bool(o and o['exact_after'])))
        print(f'fold {f}: {len(Xu):,} train configs, {len(te):,} scored, {time.time() - t0:.0f}s', flush=True)

    os.makedirs(os.path.join(data.DATA_DIR, 'cache'), exist_ok=True)
    np.save(os.path.join(data.DATA_DIR, 'cache', 'proba_fraud_oof.npy'), P_oof)

    # --- results
    B = P_oof >= THR
    n_miss = (B & (Y == 0)).sum(1)
    n_extra = (~B & (Y == 1)).sum(1)
    rows = np.array(sorted(found))
    cust = pd.DataFrame(dict(
        row=rows, MSISDN=d['msisdn_train'][rows],
        score=[round(found[r]['score'], 4) for r in rows],
        culprit_crm_packs=[';'.join(crm[found[r]['culprits']]) for r in rows],
        n_culprits=[len(found[r]['culprits']) for r in rows],
        missing_billing=[';'.join(bil[sorted(found[r]['fixed'])]) for r in rows],
        n_missing_total=n_miss[rows], n_extra_billing=n_extra[rows],
        billing_unchanged=[found[r]['exact_after'] for r in rows],
        kept_for_training=kept[rows], n_rare_packs=rare[rows]))
    step17 = os.path.join(data.DERIVED_DIR, 'unbilled_crm_customers.csv')
    if os.path.exists(step17):
        cust['in_step17'] = cust.row.isin(pd.read_csv(step17, usecols=['row']).row)
    cust = cust.sort_values(['billing_unchanged', 'score'], ascending=False)
    cust.to_csv(os.path.join(data.DERIVED_DIR, 'fraud_model_customers.csv'), index=False)

    pk = cust.assign(p=cust.culprit_crm_packs.str.split(';')).explode('p')
    has_kept = pd.Series(X[kept][:, pack_cols].sum(0), index=crm[pack_cols])
    has_all = pd.Series(X[:, pack_cols].sum(0), index=crm[pack_cols])
    packs = pd.DataFrame(dict(customers=pk.groupby('p').size(),
                              customers_kept=pk[pk.kept_for_training].groupby('p').size(),
                              billing_unchanged=pk[pk.billing_unchanged].groupby('p').size())).fillna(0).astype(int)
    packs['n_rows'] = has_all.reindex(packs.index).to_numpy()
    packs['n_kept'] = has_kept.reindex(packs.index).to_numpy()
    packs['rate_kept'] = (packs.customers_kept / packs.n_kept.clip(lower=1)).round(4)
    packs = packs.sort_values('customers', ascending=False)
    packs.index.name = 'crm_pack'
    packs.to_csv(os.path.join(data.DERIVED_DIR, 'fraud_model_packs.csv'))

    pd.set_option('display.width', 220, 'display.max_columns', 20, 'display.max_colwidth', 50)
    print(f'\n== OOF Stage 1 on kept rows: exact match vs raw billing {(~(B != (Y == 1)).any(1))[kept].mean():.2%}, '
          f'vs consensus {((B == (Y_cons == 1)).all(1))[kept].mean():.2%}')
    print(f'rows with >= 1 missing billing item: {(n_miss > 0).sum():,} (kept {(n_miss > 0)[kept].sum():,})')
    print(f'\n== customers with a culprit CRM pack: {len(cust):,} ({len(cust) / len(X):.1%} of train) ==')
    print(pd.crosstab(cust.billing_unchanged, cust.kept_for_training, margins=True).to_string())
    if 'in_step17' in cust:
        print('\noverlap with Step 17 (rule-based):')
        print(pd.crosstab(cust.in_step17, cust.kept_for_training, margins=True).to_string())
        s17 = pd.read_csv(step17, usecols=['row', 'kept_for_training'])
        print(f'Step 17 customers also found here: {s17.row.isin(cust.row).mean():.1%}')
    print('\nculprit packs per customer:', cust.n_culprits.clip(upper=6).value_counts().sort_index().to_dict())
    print('\nscore quantiles:', cust.score.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    print(f'\n== culprit packs: {len(packs)} distinct; per pack customers mean {packs.customers.mean():.0f}, '
          f'sd {packs.customers.std():.0f}')
    print('top 15 by customers:\n' + packs.head(15).to_string())
    print('top 15 by kept rate (n_kept >= 200):\n'
          + packs[packs.n_kept >= 200].sort_values('rate_kept', ascending=False).head(15).to_string())

    s = pd.DataFrame(syn)
    print(f'\n== synthetic: one random CRM pack added to {len(s):,} clean rows the model gets right ==')
    print(f"prediction changes (pack is billable for this customer): {s.billing_changes.mean():.1%}")
    sb = s[s.billing_changes]
    print(f"among those: flagged {sb.flagged.mean():.1%}, added pack named as culprit {sb.found.mean():.1%}, "
          f"billing-unchanged verdict {sb.exact_after.mean():.1%}")
    print(f"over all added packs: added pack found {s.found.mean():.1%}")


if __name__ == '__main__':
    main(*sys.argv[1:])
