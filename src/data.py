"""Load raw data + Sprint 1 pipeline outputs once, cache as compact .npz for fast reuse."""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
DERIVED_DIR = os.path.join(DATA_DIR, 'derived')
CACHE = os.path.join(DATA_DIR, 'cache', 'arrays.npz')

# The Sprint 1 exports each pipeline needs on top of train.csv / test.csv. They are gitignored
# (data/*.csv), so a teammate may legitimately have only the raw files.
PIPELINE_SOURCES = {
    'raw': (),
    's10': ('train_clean.csv',),
    's11': ('train_clean_v2.csv', 'test_clean_v2.csv'),
}


def _header(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), nrows=0).columns.tolist()


def _read(name, usecols, dtype):
    dmap = {c: dtype for c in usecols if c != 'MSISDN'}
    dmap['MSISDN'] = str
    return pd.read_csv(os.path.join(DATA_DIR, name), usecols=usecols, dtype=dmap)[usecols]


def _available(pipeline):
    return all(os.path.exists(os.path.join(DATA_DIR, n)) for n in PIPELINE_SOURCES[pipeline])


def build_cache():
    """Cache every pipeline whose Sprint 1 exports are present; record which those were.

    's10' and 's11' are optional: 'raw' needs only train.csv/test.csv, so a missing Sprint 1
    export must not block work that never asks for those feature sets. features() and
    train_target() raise a clear error if you request a pipeline that was not cached.
    """
    head = _header('train.csv')
    crm_cols = [c for c in head if c.startswith('CRM')]
    bil_cols = [c for c in head if c.startswith('BIL')]

    train = _read('train.csv', ['MSISDN'] + crm_cols + bil_cols, 'uint8')
    test = _read('test.csv', ['MSISDN'] + crm_cols, 'uint8')

    arrays = dict(
        X_train=train[crm_cols].to_numpy(np.uint8),
        X_test=test[crm_cols].to_numpy(np.uint8),
        Y_raw=train[bil_cols].to_numpy(np.uint8),
        msisdn_train=train['MSISDN'].to_numpy(str),
        msisdn_test=test['MSISDN'].to_numpy(str),
        crm_cols=np.array(crm_cols), bil_cols=np.array(bil_cols),
    )
    pipelines = ['raw']

    if _available('s10'):
        s10_head = _header('train_clean.csv')
        s10_cols = [c for c in s10_head if c.startswith('CRM')]
        s10 = _read('train_clean.csv', ['MSISDN'] + [c for c in s10_head if c.startswith('BIL')], 'uint8')
        assert (s10['MSISDN'].to_numpy() == train['MSISDN'].to_numpy()).all(), 'S10 row order differs'
        assert s10.columns[1:].tolist() == bil_cols, (
            f'train_clean.csv carries {len(s10.columns) - 1} BIL columns, train.csv has '
            f'{len(bil_cols)}: stale Section 10 export, regenerate it')
        arrays.update(Y_s10=s10[bil_cols].to_numpy(np.uint8), s10_cols=np.array(s10_cols))
        pipelines.append('s10')

    if _available('s11'):
        s11_head = _header('train_clean_v2.csv')
        s11_cols = [c for c in s11_head if c.startswith('CRM')]
        svd_cols = [c for c in s11_head if c.startswith('SVD')]
        svd_tr = _read('train_clean_v2.csv', ['MSISDN'] + svd_cols, 'float32')
        svd_te = _read('test_clean_v2.csv', ['MSISDN'] + svd_cols, 'float32')
        assert (svd_tr['MSISDN'].to_numpy() == train['MSISDN'].to_numpy()).all()
        assert (svd_te['MSISDN'].to_numpy() == test['MSISDN'].to_numpy()).all()
        arrays.update(svd_train=svd_tr[svd_cols].to_numpy(np.float32),
                      svd_test=svd_te[svd_cols].to_numpy(np.float32),
                      s11_cols=np.array(s11_cols))
        pipelines.append('s11')

    arrays['pipelines'] = np.array(pipelines)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **arrays)
    skipped = [p for p in PIPELINE_SOURCES if p not in pipelines]
    print(f'cache built with pipelines {pipelines}'
          + (f' (skipped {skipped}: exports missing from data/)' if skipped else ''))


def cached_pipelines(d):
    """Pipelines this cache can serve. Caches built before `pipelines` was recorded held all three."""
    return list(d.get('pipelines', list(PIPELINE_SOURCES)))


def _require(d, pipeline):
    if pipeline in cached_pipelines(d):
        return
    raise FileNotFoundError(
        f"pipeline '{pipeline}' is not in {CACHE}: needs {list(PIPELINE_SOURCES[pipeline])} in "
        f"{DATA_DIR}. Add them and call data.build_cache(), or use pipeline='raw'.")


def _stale():
    """True if an export has appeared since the cache was built, so it can now serve more."""
    z = np.load(CACHE, allow_pickle=False)
    cached = set(z['pipelines'].tolist()) if 'pipelines' in z.files else set(PIPELINE_SOURCES)
    return any(p not in cached and _available(p) for p in PIPELINE_SOURCES)


def load():
    if not os.path.exists(CACHE) or _stale():
        build_cache()
    z = np.load(CACHE, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ('crm_cols', 'bil_cols', 's10_cols', 's11_cols', 'pipelines'):
        if k in d:
            d[k] = d[k].tolist()
    return d


def features(d, pipeline, split='train'):
    """CRM feature matrix for a pipeline: 'raw' (745), 's10' (Section 10), 's11' (Section 11 + SVD)."""
    X = d['X_train'] if split == 'train' else d['X_test']
    if pipeline == 'raw':
        return X
    _require(d, pipeline)
    idx = {c: i for i, c in enumerate(d['crm_cols'])}
    cols = d['s10_cols'] if pipeline == 's10' else d['s11_cols']
    Xp = X[:, [idx[c] for c in cols]]
    if pipeline == 's11':
        svd = d['svd_train'] if split == 'train' else d['svd_test']
        return np.hstack([Xp.astype(np.float32), svd])
    return Xp


def train_target(d, pipeline):
    """Training labels as each pipeline defines them: S10 = majority-relabeled, else raw."""
    if pipeline != 's10':
        return d['Y_raw']
    _require(d, 's10')
    return d['Y_s10']


def additive_rules():
    with open(os.path.join(DERIVED_DIR, 'additive_rules_v2.json')) as f:
        return json.load(f)


def model_target_mask(bil_cols):
    """Boolean mask over bil_cols: True where Section 11 leaves the column to the ML model."""
    rules = additive_rules()
    return np.array([c not in rules for c in bil_cols])


def apply_additive_rules(Y_pred, X_raw, crm_cols, bil_cols):
    """Overwrite the 42 deterministic BIL columns with OR over their CRM triggers."""
    Y = np.array(Y_pred, dtype=np.uint8, copy=True)
    cidx = {c: i for i, c in enumerate(crm_cols)}
    rules = additive_rules()
    bidx = {c: i for i, c in enumerate(bil_cols)}
    for b, trigs in rules.items():
        Y[:, bidx[b]] = X_raw[:, [cidx[t] for t in trigs]].max(axis=1)
    return Y
