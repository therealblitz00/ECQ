"""Exp 14 - can the long-tail billing items be recovered from the structure of the CRM -> BIL
mapping? (Step 13)

Stage 1 = current champion (Exp 13 baseline probabilities, threshold 0.5, no rules).
Problem labels are chosen from the TRAINING fold only: items whose row-weighted out-of-fold
(OOF) F1 at 0.5 is below 0.5.

Subcommands
  diag  <seed>                 per-label table, signal classification, identity permutation test
  fit   <seed> <variant>       Stage-2 specialists for the problem labels (see VARIANTS)
  eval  <seed> <variant> <tau> merge (add-only) and full metric report vs Stage 1
  rules <seed>                 deterministic rule override (precision-gated on the training fold)

Leakage guards
  * Stage-2 training features that come from a model are Stage-1 OOF probabilities on the
    training fold's unique configurations (5-fold, Exp 13). At validation, Stage-1
    probabilities come from the model fitted on the whole training fold (standard stacking).
  * Communities, rules and neighbour lists are mined on the training fold only.
  * Merge thresholds are fixed in advance (0.5 and 0.8); validation sweeps are reported as
    trade-off curves only, never used to pick a setting.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import data, experiment as E
from scripts.sprint3 import exp13_macro_f1 as X13

CACHE = os.path.join(data.DATA_DIR, 'cache')
RARE_FREQ = 0.002
MIN_POS = 3          # minimum positive configurations behind any single-column / rule / neighbour claim
STRONG = 0.5         # confidence that counts as a usable signal
B = 1000


# ================================================================ shared set-up
def setup(seed):
    d = X13.setup(int(seed))
    X = d['X_train']
    d['rare_crm'] = np.array([c.endswith('_PACK') for c in d['crm_cols']]) & (X.mean(0) < RARE_FREQ)
    P, oof = X13.load(d, 'base')
    d['P1'], d['oof1'] = P, oof                          # P1: all validation rows; oof1: training configs
    w = d['wu'].astype(np.float32)
    f1_oof = X13.f1_from(*X13.counts(d['Yu'], oof >= 0.5, w))
    d['problem'] = (d['Yu'].sum(0) > 0) & (np.nan_to_num(f1_oof, nan=1.0) < 0.5)
    d['f1_oof'] = f1_oof
    return d


def tag(d, name):
    return f'E14-{name}{d["sfx"]}'


# ================================================================ evaluation (same baseline for every experiment)
def report(d, pred_full, name, params, extra=None):
    """All requested metrics on clean-like validation rows vs the Stage-1 champion."""
    cl, Y = d['cl'], d['Yv']
    base = (d['P1'][cl] >= 0.5).astype(np.uint8)
    pred = pred_full[cl]
    prob = d['problem']

    def stats(Pr):
        tp, fp, fn = X13.counts(Y, Pr)
        act = (tp + fn) > 0
        f1 = X13.f1_from(tp, fp, fn)
        ok = (Pr == Y).all(1)
        sup = (tp + fn)
        return dict(
            emr=float(ok.mean()), exact_rows=int(ok.sum()),
            macro_f1=float(np.nanmean(f1[act])), micro_f1=float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())),
            weighted_f1=float(np.nansum(f1[act] * sup[act]) / sup[act].sum()),
            wrong_bits=int(fp.sum() + fn.sum()), hamming=float((fp.sum() + fn.sum()) / Y.size),
            fp=int(fp.sum()), fn=int(fn.sum()), zero_labels=int((act & (tp == 0)).sum()),
            macro_f1_problem=float(np.nanmean(f1[act & prob])), macro_f1_other=float(np.nanmean(f1[act & ~prob])),
            recall_problem=float(tp[prob].sum() / max((tp + fn)[prob].sum(), 1)),
            fp_problem=int(fp[prob].sum())), ok

    s0, ok0 = stats(base)
    s1, ok1 = stats(pred)
    rc = d['rare_count'][d['va']][cl]
    seg = {f'emr_rare{k}': [float(ok0[rc == k].mean()), float(ok1[rc == k].mean())] for k in (0, 1, 2)}
    G, W = X13.boot_weights(d['gv'])
    f1a, emra = X13.boot_stats(G, W, Y, base)
    f1b, emrb = X13.boot_stats(G, W, Y, pred)
    out = dict(
        variant=name, seed=int(d['seed']),
        emr_base=round(s0['emr'] * 100, 3), emr=round(s1['emr'] * 100, 3),
        emr_delta_pt=round((s1['emr'] - s0['emr']) * 100, 3),
        emr_ci_pt=np.percentile((emrb - emra) * 100, [2.5, 97.5]).round(3).tolist(),
        macro_f1_base=round(s0['macro_f1'], 4), macro_f1=round(s1['macro_f1'], 4),
        macro_f1_delta=round(s1['macro_f1'] - s0['macro_f1'], 4),
        macro_f1_ci=np.percentile(f1b - f1a, [2.5, 97.5]).round(4).tolist(),
        rows_recovered=int((ok1 & ~ok0).sum()), rows_broken=int((~ok1 & ok0).sum()),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s1.items() if k not in ('emr', 'macro_f1')},
        base_fp=s0['fp'], base_fn=s0['fn'], base_zero_labels=s0['zero_labels'],
        base_macro_f1_problem=round(s0['macro_f1_problem'], 4), base_recall_problem=round(s0['recall_problem'], 4),
        n_problem_labels=int(prob.sum()), **{k: [round(x * 100, 2) for x in v] for k, v in seg.items()},
        **(extra or {}))
    res = E.score(d, pred_full)
    res.update({k: v for k, v in out.items() if k not in ('variant', 'seed', 'emr')})
    res['emr_cleanlike_pct'] = out['emr']
    res['f1_macro_cleanlike'] = s1['macro_f1']
    E.log(tag(d, name), 'raw+consensus (corrupted rows dropped, k=4)', 'Stage 1 champion + long-tail Stage 2',
          {**params, 'split_seed': int(d['seed'])}, res, 0)
    print(json.dumps(out))
    return out


# ================================================================ diagnostics
def single_predictors(Xu, Yu, w, labels):
    Xf, Yf = Xu.astype(np.float32), Yu[:, labels].astype(np.float32)
    pos_cfg = Xf.T @ Yf                                        # configs with CRM c and label j
    conf = ((Xf * w[:, None]).T @ Yf) / np.maximum((Xf * w[:, None]).sum(0)[:, None], 1e-9)   # row-weighted P(j|c)
    conf = np.where(pos_cfg >= MIN_POS, conf, 0)
    return conf, pos_cfg


def greedy_rule(Xu, y, w, max_len=3):
    """Conditional pattern growth for one label: start from the CRM items frequent among its
    positive configurations; greedily add the item that most raises row-weighted confidence
    while keeping >= MIN_POS positive configurations. Returns (items, conf, n_pos_cfg, n_cfg)."""
    pos = y.astype(bool)
    cand = np.flatnonzero(Xu[pos].sum(0) >= MIN_POS)
    if len(cand) == 0:
        return [], 0.0, 0, 0
    cover = np.ones(len(Xu), bool)
    items, best = [], (0.0, 0, 0)
    for _ in range(max_len):
        Xc = Xu[cover][:, cand].astype(np.float32)
        wc, yc = w[cover], pos[cover]
        n_pos = Xc[yc].sum(0)
        conf = (Xc * (wc * yc)[:, None]).sum(0) / np.maximum((Xc * wc[:, None]).sum(0), 1e-9)
        conf[n_pos < MIN_POS] = -1
        k = int(np.argmax(conf))
        if conf[k] <= best[0] + 1e-9:
            break
        items.append(int(cand[k]))
        cover &= Xu[:, cand[k]] == 1
        best = (float(conf[k]), int(n_pos[k]), int(cover.sum()))
    return items, best[0], best[1], best[2]


def communities(Yu):
    Yf = Yu.astype(np.float32)
    inter = Yf.T @ Yf
    n = np.diag(inter)
    jac = inter / np.maximum(n[:, None] + n[None, :] - inter, 1)
    np.fill_diagonal(jac, 0)
    G = nx.Graph()
    G.add_nodes_from(range(Yu.shape[1]))
    ii, jj = np.where(np.triu(jac >= 0.1, 1))
    G.add_weighted_edges_from((int(a), int(b), float(jac[a, b])) for a, b in zip(ii, jj))
    comms = nx.community.louvain_communities(G, weight='weight', seed=0)
    lab = np.full(Yu.shape[1], -1)
    for k, c in enumerate(sorted(comms, key=len, reverse=True)):
        lab[list(c)] = k
    return lab, jac


def identity_test(d, labels, n_perm=50, seed=0):
    """Among training configurations with >= 1 rare CRM pack: does the IDENTITY of the rare CRM
    pack tell which long-tail BIL item is billed? Compared with a null where the BIL rows are
    shuffled among the same configurations (keeps every count, breaks the link)."""
    Xu, Yu = d['Xu'], d['Yu']
    m = Xu[:, d['rare_crm']].sum(1) > 0
    Xr = Xu[m][:, d['rare_crm']].astype(np.float32)
    Yr = Yu[m][:, labels].astype(np.float32)

    def stat(Yx):
        O = Xr.T @ Yx
        n_c = Xr.sum(0)[:, None]
        strong = (O >= MIN_POS) & (O / np.maximum(n_c, 1) >= STRONG)
        return int(strong.any(0).sum()), float(O.max(0).sum() / max(Yx.sum(), 1))
    obs = stat(Yr)
    rng = np.random.default_rng(seed)
    null = np.array([stat(Yr[rng.permutation(len(Yr))]) for _ in range(n_perm)])
    return obs, null.mean(0), np.percentile(null, 95, axis=0), int(m.sum())


def diag(seed):
    d = setup(seed)
    seed = int(seed)
    Xu, Yu, w = d['Xu'], d['Yu'], d['wu'].astype(np.float32)
    cl, Yv = d['cl'], d['Yv']
    Xv = d['X_train'][d['va']][cl]
    pv = (d['P1'][cl] >= 0.5)
    prob = np.flatnonzero(d['problem'])
    bil, crm = np.array(d['bil_cols']), np.array(d['crm_cols'])
    rare_v = Xv[:, d['rare_crm']].sum(1) > 0
    rare_t = Xu[:, d['rare_crm']].sum(1) > 0
    common = ~d['problem']

    conf, pos_cfg = single_predictors(Xu, Yu, w, prob)
    comm, jac = communities(Yu)
    # P(j | k) over training configs (row-weighted) for k among NON-problem labels (predictable at Stage 1)
    Yf = Yu.astype(np.float32)
    co_w = (Yf * w[:, None]).T @ Yf
    nk = np.diag(co_w)
    co_cfg = Yf.T @ Yf

    tp, fp, fn = X13.counts(Yv, pv)
    rows = []
    for i, j in enumerate(prob):
        top5 = np.argsort(-conf[:, i])[:5]
        best_c = int(top5[0])
        items, rconf, rpos, rcfg = greedy_rule(Xu, Yu[:, j], w)
        nb = np.where(common & (co_cfg[:, j] >= MIN_POS), co_w[:, j] / np.maximum(nk, 1e-9), 0)
        nb[j] = 0
        k = int(np.argmax(nb))
        ypos = Yv[:, j] == 1
        fire_c = Xv[:, best_c] == 1
        fire_r = np.all(Xv[:, items] == 1, axis=1) if items else np.zeros(len(Xv), bool)
        rows.append(dict(
            label=bil[j], val_pos=int(ypos.sum()), val_prev=float(ypos.mean()),
            val_f1=float(X13.f1_from(tp[j:j + 1], fp[j:j + 1], fn[j:j + 1])[0]) if ypos.any() else np.nan,
            val_precision=float(tp[j] / max(tp[j] + fp[j], 1)), val_recall=float(tp[j] / max(tp[j] + fn[j], 1)),
            val_pred_pos=int(pv[:, j].sum()), val_pred_neg=int((~pv[:, j]).sum()),
            oof_f1=float(np.nan_to_num(d['f1_oof'][j])),
            train_pos_cfg=int(Yu[:, j].sum()),
            best_crm=crm[best_c], best_crm_conf=float(conf[best_c, i]),
            top5_crm='; '.join(f'{crm[c]}:{conf[c, i]:.2f}' for c in top5 if conf[c, i] > 0),
            best_crm_val_precision=float((fire_c & ypos).sum() / max(fire_c.sum(), 1)),
            share_pos_rare_train=float(rare_t[Yu[:, j] == 1].mean()),
            share_pos_rare_val=float(rare_v[ypos].mean()) if ypos.any() else np.nan,
            rule=' & '.join(crm[items]) if items else '', rule_conf=rconf, rule_pos_cfg=rpos, rule_cfg=rcfg,
            rule_val_fires=int(fire_r.sum()), rule_val_precision=float((fire_r & ypos).sum() / max(fire_r.sum(), 1)),
            best_bil_neighbour=bil[k] if nb[k] > 0 else '', neighbour_conf=float(nb[k]),
            community=int(comm[j]), community_size=int((comm == comm[j]).sum()),
            community_has_common=bool((common & (comm == comm[j])).sum() > 0) if comm[j] >= 0 else False,
        ))
    t = pd.DataFrame(rows)

    def klass(r):
        if r.best_crm_conf >= STRONG:
            return '1 CRM column'
        if r.rule_conf >= STRONG and len(r.rule.split(' & ')) > 1:
            return '2 CRM combination'
        if r.neighbour_conf >= STRONG:
            return '3 other BIL labels'
        if r.share_pos_rare_train >= 0.8:
            return '5 likely injection residue'
        return '4 diffuse'
    t['group'] = t.apply(klass, axis=1)
    out = os.path.join(CACHE, f'e14_label_table{d["sfx"]}.csv')
    t.to_csv(out, index=False)

    print(f'seed {seed}: {len(prob)} problem labels (training-fold OOF F1 < 0.5); '
          f'{int((t.val_pos > 0).sum())} have a clean-like validation positive; table -> {out}')
    g = t.groupby('group').agg(labels=('label', 'size'), val_pos=('val_pos', 'sum'), mean_val_f1=('val_f1', 'mean'),
                               median_train_pos_cfg=('train_pos_cfg', 'median'),
                               median_share_rare=('share_pos_rare_train', 'median'),
                               median_best_crm_conf=('best_crm_conf', 'median'),
                               median_rule_conf=('rule_conf', 'median'),
                               median_rule_val_precision=('rule_val_precision', 'median'),
                               median_neighbour_conf=('neighbour_conf', 'median'))
    print(g.round(3).to_string())
    print(f'\nproblem labels sharing a Louvain community with at least one common label: '
          f'{t.community_has_common.mean():.0%}; communities: {comm.max() + 1}')
    # Where do the strong single predictors / rules point? To rare CRM packs?
    for grp in ('1 CRM column', '2 CRM combination'):
        s = t[t.group == grp]
        if len(s):
            rr = s.best_crm.map(lambda c: bool(d['rare_crm'][d['crm_cols'].index(c)])).mean()
            print(f'{grp}: best CRM column is itself a rare pack for {rr:.0%}; '
                  f'validation precision of that column: median {s.best_crm_val_precision.median():.2f}; '
                  f'rule validation precision (when it fires): median {s.rule_val_precision.median():.2f}')

    print('\nIdentity test (training configurations with >= 1 rare CRM pack): number of labels with a rare '
          'CRM pack that bills them >= 50% of the time (>= 3 configs), and mean share of a label\'s positives '
          'explained by its single best rare pack')
    good_rare = np.flatnonzero(~d['problem'] & (Yu.sum(0) > 0) & (Yu.mean(0) < 0.01))
    for nm, lab in (('problem labels', prob), ('control: well-predicted rare labels', good_rare)):
        obs, nmean, n95, nrows = identity_test(d, lab)
        print(f'  {nm} ({len(lab)}): observed {obs[0]} labels / {obs[1]:.2f} | shuffled null mean '
              f'{nmean[0]:.1f} / {nmean[1]:.2f} (95th pct {n95[0]:.0f} / {n95[1]:.2f}) | {nrows:,} configs')
    np.save(os.path.join(CACHE, f'e14_comm{d["sfx"]}.npy'), comm)


# ================================================================ Stage 2
def neighbour_lists(Yu, labels, k=10):
    """Top-k NON-problem labels by row-free Jaccard with each problem label (training fold)."""
    Yf = Yu.astype(np.float32)
    inter = Yf.T @ Yf
    n = np.diag(inter)
    jac = inter / np.maximum(n[:, None] + n[None, :] - inter, 1)
    return {int(j): np.argsort(-jac[:, j])[1:k + 1] for j in labels}


def rule_features(d, labels, A, Xsrc=None, Ysrc=None, wsrc=None):
    """For each problem label: 1 if its greedy CRM rule (mined on Xsrc/Ysrc, default the whole
    training fold) fires on the rows of A."""
    Xsrc = d['Xu'] if Xsrc is None else Xsrc
    Ysrc = d['Yu'] if Ysrc is None else Ysrc
    wsrc = d['wu'].astype(np.float32) if wsrc is None else wsrc
    F = np.zeros((len(A), len(labels)), np.float32)
    for i, j in enumerate(labels):
        items, conf, npos, _ = greedy_rule(Xsrc, Ysrc[:, j], wsrc)
        if items:
            F[:, i] = np.all(A[:, items] == 1, axis=1)
    return F


def rule_features_oof(d, labels):
    """Training-fold rule indicators where each configuration's rules were mined WITHOUT it
    (5-fold, same folds as the Stage-1 OOF), so the specialist never sees a rule fitted to its
    own label."""
    from sklearn.model_selection import KFold
    Xu, Yu, w = d['Xu'], d['Yu'], d['wu'].astype(np.float32)
    F = np.zeros((len(Xu), len(labels)), np.float32)
    for a, b in KFold(5, shuffle=True, random_state=0).split(Xu):
        F[b] = rule_features(d, labels, Xu[b], Xu[a], Yu[a], w[a])
    return F


VARIANTS = {
    # name: (feature blocks, scale_pos_weight cap or None)
    'crm-cw':    (['crm'], 10),                    # class weights only (the old plan's Step 4)
    'ctx':       (['crm', 'p1all'], None),         # + all Stage-1 probabilities (cascade)
    'ctx-cw':    (['crm', 'p1all'], 10),
    'chain':     (['crm', 'p1nb'], None),          # + Stage-1 probs of 10 related common labels (targeted chain)
    'comm':      (['crm', 'comm'], None),          # + community activation (sum of Stage-1 probs per community)
    'rules':     (['crm', 'rules', 'rarecnt'], None),  # + rule indicators + rare-pack count
    'chain-cw':  (['crm', 'p1nb'], 10),
    'comm-cw':   (['crm', 'comm'], 10),
    'rules-cw':  (['crm', 'rules', 'rarecnt'], 10),
    'rulesoof-cw': (['crm', 'rules', 'rarecnt'], 10),    # rule indicators mined out-of-fold on training rows
    'stage1':    (['none: Stage-1 probabilities re-thresholded (ablation)'], None),
}


def build(d, blocks, A, P1, labels, comm, rf):
    parts = []
    if 'crm' in blocks:
        parts.append(A.astype(np.float32))
    if 'p1all' in blocks:
        parts.append(P1.astype(np.float32))
    if 'comm' in blocks:
        K = comm.max() + 1
        M = np.zeros((P1.shape[1], K), np.float32)
        M[np.arange(P1.shape[1])[comm >= 0], comm[comm >= 0]] = 1
        parts.append(P1.astype(np.float32) @ M)
    if 'rules' in blocks:
        parts.append(rf)
    if 'rarecnt' in blocks:
        parts.append(A[:, d['rare_crm']].sum(1, keepdims=True).astype(np.float32))
    return np.hstack(parts)


def fit(seed, variant):
    d = setup(seed)
    blocks, cap = VARIANTS[variant]
    labels = np.flatnonzero(d['problem'])
    Xu, Yu = d['Xu'], d['Yu']
    Xv = d['X_train'][d['va']]
    comm = np.load(os.path.join(CACHE, f'e14_comm{d["sfx"]}.npy'))
    rf_tr = rf_va = None
    if 'rules' in blocks:
        rf_tr = rule_features_oof(d, labels) if variant.startswith('rulesoof') else rule_features(d, labels, Xu)
        rf_va = rule_features(d, labels, Xv)
    nb = neighbour_lists(Yu, labels) if 'p1nb' in blocks else None
    Ftr_all = build(d, [b for b in blocks if b != 'p1nb'], Xu, d['oof1'], labels, comm, rf_tr)
    Fva_all = build(d, [b for b in blocks if b != 'p1nb'], Xv, d['P1'], labels, comm, rf_va)

    def one(i, j):
        y = Yu[:, j]
        Ftr, Fva = Ftr_all, Fva_all
        if nb is not None:
            Ftr = np.hstack([Ftr, d['oof1'][:, nb[j]]])
            Fva = np.hstack([Fva, d['P1'][:, nb[j]]])
        if 'rules' in blocks:       # only the label's own rule indicator
            keep = np.r_[np.arange(Xu.shape[1]), Xu.shape[1] + i, Ftr.shape[1] - 1]
            Ftr, Fva = Ftr[:, keep], Fva[:, keep]
        spw = 1.0 if cap is None else float(np.clip(np.sqrt((len(y) - y.sum()) / max(y.sum(), 1)), 1, cap))
        m = lgb.LGBMClassifier(n_jobs=2, scale_pos_weight=spw, **X13.BASE).fit(Ftr, y)
        return m.predict_proba(Fva)[:, 1].astype(np.float32)
    with E.Timer() as t:
        cols = Parallel(n_jobs=8, backend='threading')(delayed(one)(i, j) for i, j in enumerate(labels))
    P2 = np.column_stack(cols)
    np.save(os.path.join(CACHE, f'proba_{tag(d, variant)}.npy'), P2.astype(np.float16))
    print(f'{variant}: {len(labels)} specialists, {Ftr_all.shape[1]} shared features, {t.s:.0f}s')


def gate_mask(d, labels, gate):
    """Where Stage 2 is allowed to add a positive (validation rows x problem labels).
    none   : everywhere
    rare   : rows with >= 1 rare CRM pack (training fold: > 90% of problem-label positives are there)
    driver : rows containing one of the label's CRM columns with training-fold P(label | column) >= 0.3
             on >= 3 positive configurations"""
    Xv = d['X_train'][d['va']]
    if gate == 'none':
        return np.ones((len(Xv), len(labels)), bool)
    if gate == 'rare':
        return np.repeat((Xv[:, d['rare_crm']].sum(1) > 0)[:, None], len(labels), 1)
    conf, _ = single_predictors(d['Xu'], d['Yu'], d['wu'].astype(np.float32), labels)
    drivers = (conf >= 0.3).astype(np.float32)                       # CRM x labels
    return (Xv.astype(np.float32) @ drivers) > 0


def merged(d, P2, labels, tau, gate='none', gm=None):
    """Add-only merge: a problem label is switched on when Stage 1 says 0, Stage 2 >= tau and the
    gate allows it. Stage-1 positives are never removed; non-problem labels are untouched."""
    pred = (d['P1'] >= 0.5).astype(np.uint8)
    sub = pred[:, labels]
    gm = gate_mask(d, labels, gate) if gm is None else gm
    pred[:, labels] = np.where((sub == 0) & (P2 >= tau) & gm, 1, sub)
    return pred


def eval_variant(seed, variant, tau, gate='none'):
    d = setup(seed)
    d['seed'] = int(seed)
    tau = float(tau)
    labels = np.flatnonzero(d['problem'])
    if variant == 'stage1':
        P2 = d['P1'][:, labels]
    else:
        P2 = np.load(os.path.join(CACHE, f'proba_{tag(d, variant)}.npy')).astype(np.float32)
    gm = gate_mask(d, labels, gate)
    pred = merged(d, P2, labels, tau, gm=gm)
    cl, Y = d['cl'], d['Yv']
    # trade-off curve (reported, never used to choose tau)
    curve = []
    for t in (0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
        pr = merged(d, P2, labels, t, gm=gm)[cl]
        curve.append([t, round(float((pr == Y).all(1).mean()) * 100, 3), round(X13.macro_f1(Y, pr), 4)])
    blocks, cap = VARIANTS[variant]
    name = f'{variant}-t{tau}' + ('' if gate == 'none' else f'-g{gate}')
    report(d, pred, name, {'stage2_features': blocks, 'scale_pos_weight_cap': cap,
                           'merge': f'add-only, Stage-2 p >= {tau}, gate={gate}'},
           extra={'curve_tau_emr_macroF1': curve})


def rules(seed):
    """Deterministic override: switch a problem label on when its training-fold rule fires,
    only for rules with training precision >= 0.9 on >= 5 positive configurations."""
    d = setup(seed)
    d['seed'] = int(seed)
    labels = np.flatnonzero(d['problem'])
    w = d['wu'].astype(np.float32)
    Xv = d['X_train'][d['va']]
    pred = (d['P1'] >= 0.5).astype(np.uint8)
    used = 0
    for j in labels:
        items, conf, npos, _ = greedy_rule(d['Xu'], d['Yu'][:, j], w)
        if items and conf >= 0.9 and npos >= 5:
            pred[:, j] |= np.all(Xv[:, items] == 1, axis=1).astype(np.uint8)
            used += 1
    report(d, pred, 'rule-override', {'rule': 'greedy CRM pattern, training conf >= 0.9, >= 5 positive configs',
                                      'n_rules': used})


def compare2(seed, va_, vb_):
    """Paired test of two Stage-2 variants, both merged with the rare gate at 0.5 (B minus A)."""
    d = setup(seed)
    labels = np.flatnonzero(d['problem'])
    gm = gate_mask(d, labels, 'rare')
    ld = lambda v: np.load(os.path.join(CACHE, f'proba_{tag(d, v)}.npy')).astype(np.float32)
    a = merged(d, ld(va_), labels, 0.5, gm=gm)[d['cl']]
    b = merged(d, ld(vb_), labels, 0.5, gm=gm)[d['cl']]
    G, W = X13.boot_weights(d['gv'])
    fa, ea = X13.boot_stats(G, W, d['Yv'], a)
    fb, eb = X13.boot_stats(G, W, d['Yv'], b)
    print(json.dumps(dict(seed=int(seed), comparison=f'{vb_} minus {va_} (rare gate, 0.5)',
                          macro_f1_delta=round(X13.macro_f1(d['Yv'], b) - X13.macro_f1(d['Yv'], a), 4),
                          macro_f1_ci=np.percentile(fb - fa, [2.5, 97.5]).round(4).tolist(),
                          emr_delta_pt=round(float(((b == d['Yv']).all(1).mean() - (a == d['Yv']).all(1).mean()) * 100), 3),
                          emr_ci_pt=np.percentile((eb - ea) * 100, [2.5, 97.5]).round(3).tolist())))


def compare(seed):
    """Paired test: class-weighted specialists (crm-cw, rare gate, 0.5) vs the simpler ablation
    (Stage-1 probabilities at 0.3, rare gate). Same rows, configuration bootstrap."""
    d = setup(seed)
    labels = np.flatnonzero(d['problem'])
    gm = gate_mask(d, labels, 'rare')
    P2 = np.load(os.path.join(CACHE, f'proba_{tag(d, "crm-cw")}.npy')).astype(np.float32)
    a = merged(d, d['P1'][:, labels], labels, 0.3, gm=gm)[d['cl']]
    b = merged(d, P2, labels, 0.5, gm=gm)[d['cl']]
    G, W = X13.boot_weights(d['gv'])
    fa, ea = X13.boot_stats(G, W, d['Yv'], a)
    fb, eb = X13.boot_stats(G, W, d['Yv'], b)
    print(json.dumps(dict(seed=int(seed), comparison='crm-cw rare 0.5 minus Stage-1 0.3 rare',
                          macro_f1_delta=round(X13.macro_f1(d['Yv'], b) - X13.macro_f1(d['Yv'], a), 4),
                          macro_f1_ci=np.percentile(fb - fa, [2.5, 97.5]).round(4).tolist(),
                          emr_delta_pt=round(float(((b == d['Yv']).all(1).mean() - (a == d['Yv']).all(1).mean()) * 100), 3),
                          emr_ci_pt=np.percentile((eb - ea) * 100, [2.5, 97.5]).round(3).tolist())))


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    {'diag': diag, 'fit': fit, 'eval': eval_variant, 'rules': rules, 'compare': compare, 'compare2': compare2}[cmd](*args)
