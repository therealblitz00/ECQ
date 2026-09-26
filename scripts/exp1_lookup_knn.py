"""Exp 1 - global mode, exact CRM lookup, Hamming k-NN over unique training configurations."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import data, experiment as E, validation as V


def knn_hamming(Xtr, Xq, k, chunk=2048):
    A = Xtr.astype(np.float32)
    na = A.sum(1)
    idx = np.empty((len(Xq), k), np.int64)
    dist = np.empty((len(Xq), k), np.float32)
    for s in range(0, len(Xq), chunk):
        B = Xq[s:s + chunk].astype(np.float32)
        D = B.sum(1)[:, None] + na[None, :] - 2 * (B @ A.T)
        part = np.argpartition(D, k - 1, axis=1)[:, :k] if k < D.shape[1] else np.argsort(D, 1)
        pd_ = np.take_along_axis(D, part, 1)
        order = np.argsort(pd_, 1, kind='stable')
        idx[s:s + chunk] = np.take_along_axis(part, order, 1)
        dist[s:s + chunk] = np.take_along_axis(pd_, order, 1)
    return idx, dist


def main():
    d = E.setup()
    tr, va, Yc = d['tr'], d['va'], d['Y_cons']

    # 1a: global mode configuration
    with E.Timer() as t:
        codes, _ = V.factorize_rows(Yc[tr])
        mode_row = Yc[tr][np.bincount(codes).argmax()]
        pred = np.tile(mode_row, (len(va), 1))
    E.log('E1a', 'raw', 'Global mode config', {}, E.score(d, pred), t.s)

    # 1b: exact CRM lookup coverage (val groups are disjoint from train by construction)
    seen = set(d['groups'][tr].tolist())
    cov = float(np.mean([g in seen for g in d['groups'][va]]))
    test_keys = set(V.row_keys(d['X_train'][tr]))
    test_cov = float(np.mean([k in test_keys for k in V.row_keys(d['X_test'])]))
    print(f'exact lookup coverage: val={cov:.4%} test={test_cov:.4%}')

    for pipe in ['raw', 's10']:
        X = data.features(d, pipe, 'train')
        Xu, Yu, cnt = E.dedup_configs(X[tr], Yc[tr])
        with E.Timer() as t:
            idx, dist = knn_hamming(Xu, X[va], k=15)
        base_t = t.s
        for k in [1, 3, 5, 15]:
            if k == 1:
                pred = Yu[idx[:, 0]]
            else:
                w = (1.0 / (1.0 + dist[:, :k])) * np.log1p(cnt[idx[:, :k]])
                votes = (Yu[idx[:, :k]] * w[:, :, None]).sum(1) / w.sum(1, keepdims=True)
                pred = (votes >= 0.5).astype(np.uint8)
            res = E.score(d, pred)
            E.log(f'E1-knn{k}', pipe, 'Hamming k-NN (unique configs)',
                  {'k': k, 'n_train_configs': len(Xu)}, res, base_t,
                  notes=f'val exact-lookup coverage {cov:.2%}; test {test_cov:.4%}')
        print('median NN hamming distance (val):', np.median(dist[:, 0]),
              ' share dist<=1:', np.mean(dist[:, 0] <= 1))


if __name__ == '__main__':
    main()
