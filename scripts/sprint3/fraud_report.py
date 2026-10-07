"""Step 20 - consolidated fraud deliverable from Steps 17-19.

Reads the outputs of diag4_unbilled_crm.py, diag5_counterfactual.py and
diag6_crm_only_detector.py (run those first) and writes:

  data/derived/fraud_customers_train.csv  one line per suspected train customer:
      confidence  high   = flagged by both the data rule (Step 17) and the model (Step 18)
                  medium = flagged by one method, CRM status Active
                  low    = flagged by one method, CRM status not Active (billing may depend
                           on the status, e.g. Barring/Suspend - see Step 18)
      altered_crm_packs  packs named by both methods if any, otherwise by the one method
      missing_billing    billing items the altered packs should have produced
  data/derived/fraud_columns.csv  one line per CRM pack: suspected customers, rate, ranks
  data/derived/fraud_test_flagged.csv  test customers above the CRM-only threshold (Step 19)

Uses train.csv only, plus test.csv inputs for the test scores (no solution.csv).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src import data, validation


def split(s):
    return set(s.split(';')) if isinstance(s, str) and s else set()


def main():
    d = data.load()
    X, Xt, crm = d['X_train'], d['X_test'], d['crm_cols']
    D = data.DERIVED_DIR
    s17 = pd.read_csv(os.path.join(D, 'unbilled_crm_customers.csv')).set_index('row')
    s17p = pd.read_csv(os.path.join(D, 'unbilled_crm_suspects.csv'))
    s18 = pd.read_csv(os.path.join(D, 'fraud_model_customers.csv')).set_index('row')
    s19 = pd.read_csv(os.path.join(D, 'fraud_crm_only_train_oof.csv')).set_index('row')
    t19 = pd.read_csv(os.path.join(D, 'fraud_crm_only_test_scores.csv'))

    # billing items each (row, pack) is missing, from Step 17
    miss17 = s17p.groupby('row').missing_bil.apply(lambda v: set(';'.join(v).split(';')))

    status_cols = [i for i, c in enumerate(crm) if c.startswith('CRM_SUBSCRIBER_STATUS_ORIG_')]
    status_names = np.array([crm[i].replace('CRM_SUBSCRIBER_STATUS_ORIG_', '') for i in status_cols])
    Xs = X[:, status_cols]
    status = np.array(['+'.join(status_names[r == 1]) or 'none' for r in Xs])

    rows = sorted(set(s17.index) | set(s18.index))
    recs = []
    for r in rows:
        a = split(s17.unbilled_crm_packs.get(r))
        b = split(s18.culprit_crm_packs.get(r))
        both = r in s17.index and r in s18.index
        packs = (a & b) or (a | b)
        miss = split(s18.missing_billing.get(r)) | miss17.get(r, set())
        if both:
            conf = 'high'
        elif status[r] == 'Active':
            conf = 'medium'
        else:
            conf = 'low'
        recs.append(dict(MSISDN=d['msisdn_train'][r], confidence=conf,
                         found_by='both' if both else ('data rule' if r in s17.index else 'model'),
                         altered_crm_packs=';'.join(sorted(packs)), n_altered_packs=len(packs),
                         missing_billing=';'.join(sorted(miss)),
                         billing_otherwise_unchanged=bool(s18.billing_unchanged.get(r, False))
                         or (s17.pattern.get(r) == 'CRM extra, billing unchanged'),
                         model_score=s18.score.get(r, np.nan), crm_only_score=s19.oof_score[r],
                         crm_status=status[r], kept_for_training=bool(s19.kept_for_training[r]), row=r))
    cust = pd.DataFrame(recs)
    order = {'high': 0, 'medium': 1, 'low': 2}
    cust = cust.sort_values(['confidence', 'model_score'], key=lambda c: c.map(order) if c.name == 'confidence' else -c.fillna(0))
    cust.to_csv(os.path.join(D, 'fraud_customers_train.csv'), index=False)

    # columns: suspected customers per CRM pack (high + medium)
    hm = cust[cust.confidence.isin(['high', 'medium'])]
    ex = hm.assign(p=hm.altered_crm_packs.str.split(';')).explode('p')
    pk = np.where(np.char.endswith(np.array(crm), '_PACK'))[0]
    n_tr = pd.Series(X[:, pk].sum(0), index=np.array(crm)[pk])
    n_te = pd.Series(Xt[:, pk].sum(0), index=np.array(crm)[pk])
    tflag = t19[t19.flagged].top_crm_pack.value_counts()
    cols = pd.DataFrame(dict(suspected_customers=ex.groupby('p').size(),
                             high_confidence=ex[ex.confidence == 'high'].groupby('p').size()))
    cols = cols.reindex(n_tr.index[n_tr > 0]).fillna(0).astype(int)
    cols['customers_with_pack'] = n_tr.reindex(cols.index)
    cols['suspected_rate'] = (cols.suspected_customers / cols.customers_with_pack).round(4)
    cols['test_customers_with_pack'] = n_te.reindex(cols.index)
    cols['test_flagged_as_top_pack'] = tflag.reindex(cols.index).fillna(0).astype(int)
    cols = cols[cols.suspected_customers > 0]
    cols['rank_by_rate'] = cols.suspected_rate.rank(ascending=False, method='min').astype(int)
    cols['rank_by_count'] = cols.suspected_customers.rank(ascending=False, method='min').astype(int)
    cols.index.name = 'crm_pack'
    cols = cols.sort_values('suspected_rate', ascending=False)
    cols.to_csv(os.path.join(D, 'fraud_columns.csv'))

    tf = t19[t19.flagged].sort_values('fraud_score', ascending=False)
    tf.to_csv(os.path.join(D, 'fraud_test_flagged.csv'), index=False)

    pd.set_option('display.width', 200, 'display.max_columns', 20, 'display.max_colwidth', 45)
    print('== train customers ==')
    print(pd.crosstab(cust.confidence, cust.kept_for_training, margins=True).loc[['high', 'medium', 'low', 'All']].to_string())
    print(f'total suspected: {len(cust):,} ({len(cust) / len(X):.1%} of {len(X):,})')
    print('found_by:', cust.found_by.value_counts().to_dict())
    print('billing otherwise unchanged (high):', int(cust[cust.confidence == 'high'].billing_otherwise_unchanged.sum()))
    print('altered packs per customer (high):',
          cust[cust.confidence == 'high'].n_altered_packs.clip(upper=6).value_counts().sort_index().to_dict())
    print('low-confidence statuses:', cust[cust.confidence == 'low'].crm_status.value_counts().head(5).to_dict())
    print(f'\n== columns: {len(cols)} CRM packs with suspected customers ==')
    print(f'suspected customers per pack: median {cols.suspected_customers.median():.0f}, '
          f'IQR {cols.suspected_customers.quantile(.25):.0f}-{cols.suspected_customers.quantile(.75):.0f}')
    print('top 15 by rate (packs with >= 100 customers):')
    print(cols[cols.customers_with_pack >= 100].head(15).to_string())
    print('top 10 by count:')
    print(cols.sort_values('suspected_customers', ascending=False).head(10).to_string())
    print(f'\n== test: {len(tf)} of {len(t19):,} customers flagged by the CRM-only detector ==')
    print(tf.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
