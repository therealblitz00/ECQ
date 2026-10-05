"""Exp 14 follow-up: where does the winning Stage 2 (crm-cw, rare gate, 0.5) gain, by
diagnostic group of the problem labels (groups from the training-fold label table)?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

from scripts.sprint3 import exp13_macro_f1 as X13
from scripts.sprint3 import exp14_longtail as X14

for seed in (42, 7):
    d = X14.setup(seed)
    labels = np.flatnonzero(d['problem'])
    t = pd.read_csv(os.path.join(X14.CACHE, f'e14_label_table{d["sfx"]}.csv'))
    P2 = np.load(os.path.join(X14.CACHE, f'proba_{X14.tag(d, "crm-cw")}.npy')).astype(np.float32)
    a = (d['P1'] >= 0.5).astype(np.uint8)[d['cl']][:, labels]
    b = X14.merged(d, P2, labels, 0.5, gate='rare')[d['cl']][:, labels]
    Y = d['Yv'][:, labels]
    fa, fb = X13.f1_from(*X13.counts(Y, a)), X13.f1_from(*X13.counts(Y, b))
    tpa, tpb = X13.counts(Y, a)[0], X13.counts(Y, b)[0]
    fpa, fpb = X13.counts(Y, a)[1], X13.counts(Y, b)[1]
    act = Y.sum(0) > 0
    t = t.assign(f1_base=np.nan_to_num(fa), f1_new=np.nan_to_num(fb), new_tp=tpb - tpa, new_fp=fpb - fpa, act=act)[act]
    g = t.groupby('group').agg(labels=('label', 'size'), val_pos=('val_pos', 'sum'), f1_base=('f1_base', 'mean'),
                               f1_new=('f1_new', 'mean'), new_tp=('new_tp', 'sum'), new_fp=('new_fp', 'sum'),
                               labels_now_nonzero=('f1_new', lambda s: int(((s > 0) & (t.loc[s.index, 'f1_base'] == 0)).sum())))
    print(f'seed {seed}'); print(g.round(3).to_string()); print()
