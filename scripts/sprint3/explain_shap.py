"""Step 6 - explainability of the Sprint 3 champion with SHAP (TreeSHAP).

Retrains the champion on the seed-42 training fold (k=4 corrupted-row filter,
min_child_samples=5, reg_lambda=1) and explains its predictions on clean-like validation
rows, where the correct answer is known. SHAP values come from LightGBM's built-in TreeSHAP
(booster.predict(pred_contrib=True)): identical to shap.TreeExplainer, in log-odds units.

  1. Selected BIL columns: top CRM drivers vs. the "expected" driver (best co-occurring
     CRM column in the cleaned training data).
  2. Global map over all 731 columns: does the top SHAP driver match the expected one, how
     concentrated is each explanation, does any column lean on rare (injection-prone) packs.
     Importance = mean |SHAP| over the rows where the column is positive (true or predicted).
  3. Worked examples of wrong rows.

Outputs: data/cache/explain_shap.txt (via stdout redirect), data/derived/sprint3_shap_drivers.csv,
docs/sprint3/figures/*.png.   Run:  .venv/Scripts/python scripts/sprint3/explain_shap.py > data/cache/explain_shap.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import data, experiment as E, metrics as M, validation as V
from src.models import BinaryRelevanceLGBM
from scripts.sprint3.diag3_remaining_errors import flip_rate

CHAMPION = dict(min_child_samples=5, reg_lambda=1.0)
DROP_K = 4
N_SAMPLE = 400
SELECTED = ['BIL_3921_PACK', 'BIL_3748_PACK', 'BIL_SUBSCRIBER_STATUS_ORIG_Active',
            'BIL_4167_PACK', 'BIL_3940_PACK', 'BIL_3803_PACK']
FIG_DIR = os.path.join(data.ROOT, 'docs', 'sprint3', 'figures')

# Reference palette (dataviz skill): light surface, ink, categorical slots 1-2.
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def section(title):
    print('\n' + '=' * 100 + f'\n{title}\n' + '=' * 100)


def contribs(model, X):
    """TreeSHAP values (n_rows, n_features) in log-odds; None for constant columns."""
    if isinstance(model, float):
        return None
    return model.booster_.predict(X, pred_contrib=True)[:, :-1]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def short(c):
    return c.replace('_PACK', '').replace('SUBSCRIBER_STATUS_ORIG_', 'STATUS_').replace('SUBSCRIBER_TYPE_ORIG_', 'TYPE_')


def main():
    d = E.setup(seed=42)
    tr, va = d['tr'], d['va']
    crm, bil = np.array(d['crm_cols']), np.array(d['bil_cols'])
    keep = tr[d['test_like'][tr] & (d['rare_count'][tr] < DROP_K)]
    Xu, Yu, _ = E.dedup_configs(d['X_train'][keep], d['Y_cons'][keep])

    with E.Timer() as t:
        model = BinaryRelevanceLGBM(CHAMPION).fit(Xu, Yu)
    Xv = d['X_train'][va].astype(np.float32)
    P = model.predict_proba(Xv)
    pred = data.apply_additive_rules((P >= .5).astype(np.uint8), d['X_train'][va], d['crm_cols'], d['bil_cols'])
    cl = d['clean_like'][va]
    Yv = d['Y_cons'][va]
    section('Champion retrained on the seed-42 training fold')
    print(f'fit {t.s:.0f}s | clean-like val EMR {M.exact_match_ratio(Yv[cl], pred[cl]):.4%} '
          f'(Step 4 reported 96.81%)')

    # Expected driver: CRM column that best co-occurs with each BIL column in cleaned training data
    Xf, Yf = Xu.astype(np.float32), Yu.astype(np.float32)
    inter = Xf.T @ Yf
    score = np.minimum(inter / np.maximum(Xf.sum(0)[:, None], 1), inter / np.maximum(Yf.sum(0)[None, :], 1))
    partner, partner_score = score.argmax(0), score.max(0)

    cp = np.array([c.endswith('_PACK') for c in crm])
    rare_crm = cp & (d['X_train'].mean(0) < V.RARE_FREQ)
    rule_cols = set(data.additive_rules())
    flips = flip_rate(d) >= 0.5

    # Per-column validation quality (clean-like rows, final predictions)
    yt, yp = Yv[cl], pred[cl]
    tp = ((yt == 1) & (yp == 1)).sum(0)
    fp = ((yt == 0) & (yp == 1)).sum(0)
    fn = ((yt == 1) & (yp == 0)).sum(0)
    f1 = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)

    # ---- Global SHAP map --------------------------------------------------------------------
    # Importance is averaged over the rows where the column is positive (true or predicted):
    # "why does this column fire?". A random sample would almost never contain a rare
    # column's driver, so common packs with tiny effects would outrank it.
    rng = np.random.default_rng(0)
    cl_idx = np.flatnonzero(cl)
    S = Xv[rng.choice(cl_idx, N_SAMPLE, replace=False)]
    pos_rows = {}
    imp = np.zeros((len(bil), len(crm)), np.float32)
    for j, m in enumerate(model.models_):
        rows = cl_idx[(Yv[cl_idx, j] == 1) | (pred[cl_idx, j] == 1)]
        if len(rows) > N_SAMPLE:
            rows = rng.choice(rows, N_SAMPLE, replace=False)
        R = Xv[rows] if len(rows) else S
        pos_rows[j] = R
        c = contribs(m, R)
        if c is not None:
            imp[j] = np.abs(c).mean(0)
    mass = imp.sum(1)
    has = mass > 0
    top = imp.argmax(1)
    conc = np.where(has, imp.max(1) / np.maximum(mass, 1e-12), np.nan)
    in_top3 = np.array([partner[j] in np.argsort(-imp[j])[:3] for j in range(len(bil))])
    rare_share = np.where(has, imp[:, rare_crm].sum(1) / np.maximum(mass, 1e-12), np.nan)

    drivers = pd.DataFrame({
        'bil_column': bil,
        'prevalence_train': Yu.mean(0).round(6),
        'val_positives_cleanlike': yt.sum(0),
        'val_f1_cleanlike': np.round(f1, 4),
        'top1_shap_driver': crm[top],
        'top1_share_of_shap': np.round(conc, 4),
        'top2_shap_driver': crm[np.argsort(-imp, 1)[:, 1]],
        'top3_shap_driver': crm[np.argsort(-imp, 1)[:, 2]],
        'expected_driver': crm[partner],
        'expected_driver_score': partner_score.round(4),
        'top1_is_expected': top == partner,
        'expected_in_top3': in_top3,
        'shap_share_on_rare_crm_packs': np.round(rare_share, 4),
        'flipping_column': flips,
        'set_by_additive_rule': [b in rule_cols for b in bil],
    })
    drivers.loc[~has, ['top1_shap_driver', 'top2_shap_driver', 'top3_shap_driver']] = ''
    drivers.to_csv(os.path.join(data.DERIVED_DIR, 'sprint3_shap_drivers.csv'), index=False)

    section('Global map: does the model rely on the CRM columns it should? (all 731 BIL columns)')
    g = drivers[has]
    strong = g.expected_driver_score >= 0.8
    print(f'columns explained: {has.sum()} of {len(bil)} (constant in training: {(~has).sum()})')
    print(f'top SHAP driver == expected co-occurrence driver: {g.top1_is_expected.mean():.1%} of columns | '
          f'expected driver in top 3: {g.expected_in_top3.mean():.1%}')
    print(f'  ... for columns with a strong expected driver (score >= 0.8, {strong.sum()} cols): '
          f'top1 {g[strong].top1_is_expected.mean():.1%}, top3 {g[strong].expected_in_top3.mean():.1%}')
    print(f'  ... for columns without one ({(~strong).sum()} cols): '
          f'top1 {g[~strong].top1_is_expected.mean():.1%}, top3 {g[~strong].expected_in_top3.mean():.1%}')
    bands = pd.cut(g.top1_share_of_shap, [0, .3, .5, .7, .9, 1.0])
    tab = g.groupby(bands, observed=True).agg(columns=('bil_column', 'size'),
                                              median_f1=('val_f1_cleanlike', 'median'),
                                              share_flipping=('flipping_column', 'mean'))
    print('\nhow concentrated each explanation is (share of SHAP mass on the single top driver):')
    print(tab.round(3).to_string())
    ok = g.val_f1_cleanlike.notna()
    rho = spearmanr(g[ok].top1_share_of_shap, g[ok].val_f1_cleanlike).statistic
    print(f'Spearman(concentration, validation F1) over {ok.sum()} columns with positives: {rho:+.3f}')
    print(f'\nnoise check - share of SHAP mass on rare CRM packs (< {V.RARE_FREQ:.1%} of rows):')
    print(f'  median over columns {g.shap_share_on_rare_crm_packs.median():.3f}; '
          f'columns with > 50% of their mass on rare packs: {(g.shap_share_on_rare_crm_packs > .5).sum()}')
    hi = g[g.shap_share_on_rare_crm_packs > .5]
    if len(hi):
        print(f'  of those, expected driver is itself a rare pack: '
              f'{np.mean([rare_crm[list(crm).index(e)] for e in hi.expected_driver]):.1%} '
              f'(i.e. rare BIL items explained by their own rare CRM product)')
    catm = ~cp
    cat_top = g[g.top1_shap_driver.isin(crm[catm])]
    print(f'\ncolumns whose top driver is a categorical CRM column: {len(cat_top)}')
    print(cat_top[['bil_column', 'top1_shap_driver', 'top1_share_of_shap', 'val_f1_cleanlike']].to_string(index=False))

    # ---- Selected columns -------------------------------------------------------------------
    section('Selected BIL columns: top CRM drivers vs the expected driver')
    sel = list(SELECTED)
    rare_ok = g[(g.prevalence_train < .005) & (g.val_f1_cleanlike >= .95) & ~g.set_by_additive_rule]
    if len(rare_ok):
        sel.insert(3, rare_ok.sort_values('val_positives_cleanlike', ascending=False).bil_column.iloc[0])
    detail = {}
    bidx = {b: i for i, b in enumerate(bil)}
    for b in sel:
        j = bidx[b]
        R = pos_rows[j]
        c = contribs(model.models_[j], R)
        order = np.argsort(-imp[j])[:5]
        rows = []
        for f in order:
            on = R[:, f] == 1
            rows.append((crm[f], float(imp[j, f]), float(c[on, f].mean()) if on.any() else np.nan))
        detail[b] = rows
        r = drivers.iloc[j]
        tag = 'flipping' if r.flipping_column else ('rule' if r.set_by_additive_rule else '')
        print(f'\n{b}  prevalence {r.prevalence_train:.4f} | val positives {r.val_positives_cleanlike} | '
              f'F1 {r.val_f1_cleanlike} | expected driver {r.expected_driver} (score {r.expected_driver_score}) {tag}')
        for name, v, push in rows:
            mark = '  <- expected' if name == r.expected_driver else ''
            eff = f'effect when present {push:+7.2f}' if push == push else 'acts through its absence  '
            print(f'    {name:40s} mean|SHAP| {v:7.3f}   {eff}{mark}')

    # ---- Worked examples of wrong rows ------------------------------------------------------
    section('Worked examples: why the model got a row wrong (clean-like val rows, exactly 1 wrong bit)')
    err = M.row_errors(Yv, pred)
    one = np.flatnonzero(cl & (err == 1))
    diff = pred != Yv
    picks = {}
    for i in one:
        j = int(np.flatnonzero(diff[i])[0])
        kind = 'missed item' if Yv[i, j] == 1 else 'extra item'
        key = ('flipping column' if flips[j] else kind) + (' (row has rare pack)' if d['rare_count'][va][i] > 0 else '')
        picks.setdefault(key, (i, j))
    for key, (i, j) in list(picks.items())[:4]:
        c = contribs(model.models_[j], Xv[i:i + 1])[0]
        base = model.models_[j].booster_.predict(Xv[i:i + 1], pred_contrib=True)[0, -1]
        active = np.flatnonzero(Xv[i] == 1)
        order = active[np.argsort(-np.abs(c[active]))][:6]
        print(f'\n[{key}] {bil[j]}: truth {Yv[i, j]}, predicted {pred[i, j]} (p={P[i, j]:.3f}); '
              f'row lists {int(Xv[i, cp].sum())} CRM packs, {int(d["rare_count"][va][i])} rare; '
              f'expected driver {crm[partner[j]]} present: {bool(Xv[i, partner[j]])}')
        print(f'    baseline log-odds {base:+.2f}; largest contributions from the CRM columns this customer has:')
        for f in order:
            print(f'      {crm[f]:40s} {c[f]:+7.2f}{"  (rare pack)" if rare_crm[f] else ""}')

    # ---- Figures ----------------------------------------------------------------------------
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})

    n = len(sel)
    fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(10, 1.55 * np.ceil(n / 2) + 0.6), facecolor=SURFACE)
    for ax, b in zip(axes.ravel(), sel):
        rows = detail[b][::-1]
        exp = drivers.iloc[bidx[b]].expected_driver
        colors = [BLUE if r[0] == exp else '#9ec5f4' for r in rows]
        ax.barh([short(r[0]) for r in rows], [r[1] for r in rows], color=colors, height=0.6)
        style(ax)
        r = drivers.iloc[bidx[b]]
        ax.set_title(f'{short(b)}  (F1 {r.val_f1_cleanlike:.2f}{", flips" if r.flipping_column else ""})',
                     fontsize=9, color=INK, loc='left')
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle('Top CRM drivers per BIL column (mean |SHAP|, log-odds); dark bar = expected driver',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shap_selected_columns.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    gg = g[ok]
    for flag, color, label in [(False, BLUE, 'consistent column'), (True, ORANGE, 'flipping column')]:
        s = gg[gg.flipping_column == flag]
        ax.scatter(s.top1_share_of_shap, s.val_f1_cleanlike, s=12, color=color, alpha=0.7,
                   edgecolors=SURFACE, linewidths=0.5, label=f'{label} ({len(s)})')
    style(ax)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_xlabel('Share of SHAP mass on the single top CRM driver', color=INK2)
    ax.set_ylabel('Validation F1 (clean-like rows)', color=INK2)
    ax.set_title('Simple one-driver columns are predicted well; flipping columns are not',
                 fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shap_concentration_vs_f1.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f'\nfigures written to {FIG_DIR}')


if __name__ == '__main__':
    main()
