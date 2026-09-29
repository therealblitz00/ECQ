"""Exp 6 - does the Sprint 1 cleaning help? raw vs s10 vs s11 with the tuned models.

The feature sets were compared once in exp3 (E3d 74.31, E3e 47.42 against E3a2 80.05) using
the UNTUNED LightGBM: min_child_samples=5, no reg_lambda, which scores 80.05 on raw where the
tuned champion reaches 87.84. Whether cleaning still hurts once the model is regularised was
never re-checked, and the feature sets have never been run through a neural network or the
neighbour model.

Design (copied from exp3_binary_relevance.py lines 50-57): deduplicate on the RAW
configuration identity, then take each pipeline's features at those same representative rows.
Training rows are therefore identical across pipelines (53,947) and only the feature matrix
varies. Targets are consensus throughout, so the s10 relabelled targets are a separate
question this script does not touch.

s11 is champion-only on purpose. It is 742 binary CRM columns plus 45 float SVD components,
and both neural/neighbour models coerce X to uint8 (models_widedeep.fit: np.asarray(X,
np.uint8); models_nndelta.fit: np.ascontiguousarray(X, np.uint8) then packbits), which would
truncate the SVD floats to zero SILENTLY -- no error, just a wrong number.

Caveat on nndelta + s10: its mechanism is "configuration pairs differing by exactly one pack".
s10 drops 11 CRM columns, so pairs that differed only by a dropped column become identical and
can contradict each other. If s10 hurts nndelta disproportionately, suspect that first.

Requires train_clean.csv / train_clean_v2.csv / test_clean_v2.csv in data/ for s10 / s11;
data.load() skips whichever pipeline's exports are absent and features() will say so.

    python scripts/exp6_pipeline_comparison.py                  # raw s10 s11, seed 42
    python scripts/exp6_pipeline_comparison.py --seeds 42 7
    python scripts/exp6_pipeline_comparison.py --pipelines raw s10 --no-log
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src import data, experiment as E, validation as V
from src.models import BinaryRelevanceLGBM
from src.models_nndelta import NeighbourDelta
from src.models_widedeep import WideDeepMultiLabel

THR = 0.5
N_BOOT = 2000
OUT = os.path.join(data.DERIVED_DIR, 'sprint2_pipeline_comparison.csv')
SIG_OUT = os.path.join(data.DERIVED_DIR, 'sprint2_pipeline_significance.csv')

CHAMP_PARAMS = dict(min_child_samples=20, reg_lambda=1.0)
WD_PARAMS = dict(hidden=(512,), dropout=0.2, wide=True, lr=2e-3,
                 batch_size=256, max_epochs=80, patience=8)
NND_PARAMS = dict(min_pairs=2, min_rate=0.5, max_nb=5)

# model -> pipelines it can legitimately consume (see the s11 note above)
CAN_USE = {'champ': ('raw', 's10', 's11'),
           'wd512': ('raw', 's10'),
           'nndelta': ('raw', 's10')}
FAMILY = {'champ': 'BR LightGBM (variance reduction)',
          'wd512': 'Wide & Deep multi-label NN',
          'nndelta': 'Neighbour + pack-delta',
          'blend': 'avg(wd512, nndelta, BR mcs20)'}


def make(name, seed):
    if name == 'champ':
        return BinaryRelevanceLGBM(CHAMP_PARAMS)
    if name == 'wd512':
        return WideDeepMultiLabel(seed=seed, **WD_PARAMS)
    return NeighbourDelta(**NND_PARAMS)


def params_of(name, seed):
    base = {'champ': CHAMP_PARAMS, 'wd512': WD_PARAMS, 'nndelta': NND_PARAMS}[name]
    p = dict(base, thr=THR, split_seed=seed)
    if name == 'wd512':
        p['model_seed'] = seed
    return p


def run_seed(seed, pipelines, do_log):
    d = E.setup(seed=seed, val_size=0.2)
    available = data.cached_pipelines(d)
    pipelines = [p for p in pipelines if p in available]
    print(f'seed {seed}: cached pipelines {available} -> running {pipelines}')

    tr, va = d['tr'], d['va']
    X_raw = data.features(d, 'raw')
    _, Y_tr, _ = E.dedup_configs(X_raw[tr], d['Y_cons'][tr])
    codes, _ = V.factorize_rows(X_raw[tr])
    _, first = np.unique(codes, return_index=True)
    Y_va = d['Y_cons'][va]
    cfg, cfg_first, cfg_size = np.unique(d['groups'][va], return_index=True, return_counts=True)
    w = cfg_size.astype(np.float64)
    print(f'  train rows {len(first):,} (identical across pipelines) | '
          f'val {len(va):,} rows / {len(cfg):,} configs')

    P, ok, rows = {}, {}, []
    for pipe in pipelines:
        X = data.features(d, pipe)
        X_tr_p, X_va_p = X[tr][first], X[va]
        print(f'  {pipe}: {X_tr_p.shape[1]} features, dtype {X_tr_p.dtype}', flush=True)
        done = []
        for name, allowed in CAN_USE.items():
            if pipe not in allowed:
                print(f'    [{name}] skipped on {pipe} (would silently truncate float features)')
                continue
            with E.Timer() as t:
                P[name, pipe] = np.asarray(make(name, seed).fit(X_tr_p, Y_tr)
                                           .predict_proba(X_va_p), dtype=np.float32)
            done.append((name, pipe, t.s))
        if len(done) == 3:
            P['blend', pipe] = sum(P[n, pipe] for n, _, _ in done) / 3.0
            done.append(('blend', pipe, 0.0))

        for name, _, secs in done:
            pred = (P[name, pipe] >= THR).astype(np.uint8)
            res = E.score(d, pred)
            ok[name, pipe] = ((pred != Y_va).sum(1) == 0)[cfg_first]
            rows.append(dict(model=name, pipeline=pipe, split_seed=seed,
                             n_features=X_tr_p.shape[1], **res))
            print(f'    {name:<8} EMR {res["emr"]:.5f}  HL {res["hamming_loss"]:.6f} '
                  f'({secs:.0f}s)', flush=True)
            if do_log:
                E.log(f'andre-{name}-{pipe}', f'{pipe}+consensus', FAMILY[name],
                      params_of(name, seed) if name != 'blend'
                      else dict(members='wd512, nndelta, BR mcs20', weights='equal',
                                thr=THR, split_seed=seed),
                      res, secs,
                      notes=f'{pipe} features, unique configs, consensus labels, unweighted')

    # paired: each pipeline against the SAME model on raw
    rng = np.random.default_rng(0)
    sig = []
    for (name, pipe), o in ok.items():
        if pipe == 'raw' or (name, 'raw') not in ok:
            continue
        ref = ok[name, 'raw']
        b, c = int((o & ~ref).sum()), int((~o & ref).sum())
        diff = float((w * (o.astype(float) - ref)).sum() / w.sum())
        dv = (o.astype(float) - ref) * w
        boots = []
        for _ in range(0, N_BOOT, 200):
            pw = rng.poisson(1.0, size=(200, len(w)))
            boots.extend((pw @ dv) / (pw @ w))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig.append(dict(split_seed=seed, model=name, comparison=f'{pipe} - raw',
                        dEMR_pts=diff * 100, ci_lo=lo * 100, ci_hi=hi * 100, b=b, c=c,
                        mcnemar_p=binomtest(b, b + c, 0.5).pvalue if b + c else 1.0,
                        significant='yes' if (lo > 0 or hi < 0) else 'no'))
    return rows, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--pipelines', nargs='+', default=['raw', 's10', 's11'])
    ap.add_argument('--no-log', action='store_true',
                    help='skip experiment.log so the shared tracker is untouched')
    a = ap.parse_args()

    t0 = time.perf_counter()
    rows, sig = [], []
    for s in a.seeds:
        r, g = run_seed(s, a.pipelines, not a.no_log)
        rows += r
        sig += g

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT, index=False)
    print(f'\n{"=" * 78}\nvalidation EMR by feature set (consensus targets)\n{"=" * 78}')
    for s in a.seeds:
        piv = (tbl[tbl.split_seed == s].pivot(index='model', columns='pipeline', values='emr')
               .reindex(['champ', 'wd512', 'nndelta', 'blend']))
        print(f'\nseed {s}:')
        print(piv.to_string(float_format=lambda v: f'{v * 100:7.2f}' if pd.notna(v) else '      -'))

    if sig:
        sg = pd.DataFrame(sig)
        sg.to_csv(SIG_OUT, index=False)
        print(f'\n{"=" * 78}\npaired vs the same model on raw features\n{"=" * 78}')
        print(sg.to_string(index=False, float_format=lambda v: f'{v:>9.4g}'))

    print(f'\n{len(rows)} runs in {time.perf_counter() - t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
