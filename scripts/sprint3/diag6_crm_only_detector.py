"""Diag 6 - CRM-only fraud detector: can a customer with an added CRM pack be recognised
from the CRM alone, so it can be scored on test.csv (which has no billing)?

Labels come from Steps 17-18 (train only):
  positive = flagged by both the data rule (diag4) and the model counterfactual (diag5);
  negative = flagged by neither; flagged by only one = ambiguous, left out of training.
Features: the 745 raw CRM columns + number of rare CRM packs + number of CRM packs.

1. 5-fold GroupKFold on the exact CRM configuration: out-of-fold ROC AUC / average precision,
   on all rows and on the kept rows (test-like, < 4 rare packs: the lightly altered customers),
   against simple baselines (rare-pack count; "not test-like").
2. Per-customer explanation: the CRM pack with the largest positive contribution
   (LightGBM pred_contrib); how often it is one of the packs Steps 17-18 named.
3. Final model on all labelled train rows -> scores for test.csv (inputs only; no
   solution.csv). Threshold = best F1 on the out-of-fold scores.

Outputs: data/derived/fraud_crm_only_test_scores.csv, fraud_crm_only_train_oof.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src import data, validation

PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=20,
              reg_lambda=1.0, n_jobs=16, verbose=-1)


def feats(X, crm_cols, X_ref):
    pk = np.char.endswith(np.array(crm_cols), '_PACK')
    extra = np.column_stack([validation.rare_pack_count(X, crm_cols, X_ref), X[:, pk].sum(1)])
    return np.hstack([X.astype(np.float32), extra.astype(np.float32)])


def top_pack(model, F, names, pack_idx):
    """Name of the CRM pack (active in the row) with the largest positive contribution."""
    C = model.predict(F, pred_contrib=True)[:, :-1]
    C = C[:, pack_idx] * (F[:, pack_idx] > 0)
    best = C.argmax(1)
    return np.where(C[np.arange(len(F)), best] > 0, names[pack_idx][best], '')


def report(name, y, s):
    return dict(set=name, n=len(y), positives=int(y.sum()), auc=round(roc_auc_score(y, s), 4),
                ap=round(average_precision_score(y, s), 4))


def main():
    d = data.load()
    X, Xt, crm = d['X_train'], d['X_test'], d['crm_cols']
    names = np.array(crm + ['n_rare_packs', 'n_packs'])
    pack_idx = np.where(np.char.endswith(names, '_PACK'))[0]
    test_like = validation.test_like_mask(X, crm)
    rare = validation.rare_pack_count(X, crm)
    kept = test_like & (rare < 4)
    groups, _ = validation.factorize_rows(X)

    s17 = pd.read_csv(os.path.join(data.DERIVED_DIR, 'unbilled_crm_customers.csv'),
                      usecols=['row', 'unbilled_crm_packs'])
    s18 = pd.read_csv(os.path.join(data.DERIVED_DIR, 'fraud_model_customers.csv'),
                      usecols=['row', 'culprit_crm_packs'])
    in17 = np.zeros(len(X), bool); in17[s17.row] = True
    in18 = np.zeros(len(X), bool); in18[s18.row] = True
    named = {}
    for df, col in [(s17, 'unbilled_crm_packs'), (s18, 'culprit_crm_packs')]:
        for r, p in zip(df.row, df[col]):
            named.setdefault(r, set()).update(p.split(';'))
    label = np.full(len(X), -1)
    label[in17 & in18] = 1
    label[~in17 & ~in18] = 0
    lab = label >= 0
    print(f'labels: positive {(label == 1).sum():,}, negative {(label == 0).sum():,}, '
          f'ambiguous (one method only) {(~lab).sum():,}')

    F = feats(X, crm, X)
    oof = np.zeros(len(X), np.float32)
    top = np.empty(len(X), object)
    for f, (tr, te) in enumerate(validation.grouped_kfold(groups, 5)):
        trl = tr[lab[tr]]
        m = lgb.LGBMClassifier(**PARAMS).fit(F[trl], label[trl])
        oof[te] = m.predict_proba(F[te])[:, 1]
        top[te] = top_pack(m.booster_, F[te], names, pack_idx)
        print(f'fold {f} done', flush=True)

    y, s = label[lab], oof[lab]
    nt = (~test_like).astype(float)
    res = [report('all labelled', y, s),
           report('  baseline: # rare packs', y, rare[lab]),
           report('  baseline: # rare packs + not test-like', y, rare[lab] + 100 * nt[lab]),
           report('kept rows', label[lab & kept], oof[lab & kept]),
           report('  baseline: # rare packs', label[lab & kept], rare[lab & kept]),
           report('kept rows with no rare pack', label[lab & kept & (rare == 0)], oof[lab & kept & (rare == 0)])]
    print('\n== out-of-fold detection (train) ==\n' + pd.DataFrame(res).to_string(index=False))

    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    i = int(np.argmax(f1[:-1]))
    thr = float(t[i])
    print(f'\nbest-F1 threshold {thr:.3f}: precision {p[i]:.3f}, recall {r[i]:.3f}, F1 {f1[i]:.3f}')
    for nm, msk in [('kept', lab & kept), ('dropped', lab & ~kept)]:
        yy, ss = label[msk], oof[msk] >= thr
        print(f'  {nm}: recall {ss[yy == 1].mean():.3f}, precision {yy[ss].mean() if ss.any() else float("nan"):.3f}, '
              f'flag rate {ss.mean():.3%}')
    print(f'ambiguous customers above threshold: {(oof[~lab] >= thr).mean():.1%}')

    pos = np.where(label == 1)[0]
    hit = np.array([top[r] in named.get(r, ()) for r in pos])
    print(f'\ntop-contributing CRM pack is one of the packs Steps 17-18 named: {hit.mean():.1%} of positives '
          f'(kept {hit[kept[pos]].mean():.1%}, dropped {hit[~kept[pos]].mean():.1%})')

    # final model -> test
    m = lgb.LGBMClassifier(**PARAMS).fit(F[lab], label[lab])
    Ft = feats(Xt, crm, X)
    st = m.predict_proba(Ft)[:, 1]
    tt = top_pack(m.booster_, Ft, names, pack_idx)
    rare_t = validation.rare_pack_count(Xt, crm, X)
    tl_t = validation.test_like_mask(Xt, crm)
    q = [.5, .9, .99, .999]
    print('\n== score distribution ==')
    print(pd.DataFrame({'train (OOF, all)': np.quantile(oof, q), 'train kept (OOF)': np.quantile(oof[kept], q),
                        'test': np.quantile(st, q)}, index=q).round(4).to_string())
    flag_t = st >= thr
    print(f'\ntest customers above threshold: {flag_t.sum():,} of {len(st):,} ({flag_t.mean():.2%}); '
          f'train OOF: {(oof >= thr).mean():.2%} (kept rows {(oof[kept] >= thr).mean():.2%})')
    print('test flagged by # rare packs:', pd.Series(rare_t[flag_t]).value_counts().sort_index().to_dict())
    print(f'test rows not test-like: {(~tl_t).sum()}')

    out = pd.DataFrame(dict(MSISDN=d['msisdn_test'], fraud_score=st.round(4), flagged=flag_t,
                            top_crm_pack=tt, n_rare_packs=rare_t))
    out.sort_values('fraud_score', ascending=False).to_csv(
        os.path.join(data.DERIVED_DIR, 'fraud_crm_only_test_scores.csv'), index=False)
    pd.DataFrame(dict(row=np.arange(len(X)), MSISDN=d['msisdn_train'], label=label, oof_score=oof.round(4),
                      top_crm_pack=top, kept_for_training=kept)).to_csv(
        os.path.join(data.DERIVED_DIR, 'fraud_crm_only_train_oof.csv'), index=False)
    print('\ntop 10 test customers:\n' + out.sort_values('fraud_score', ascending=False).head(10).to_string(index=False))
    print('\nmost frequent top pack among flagged test customers:\n'
          + out[flag_t].top_crm_pack.value_counts().head(10).to_string())


if __name__ == '__main__':
    main()
