"""Shared harness for testing a pre-processing idea against a fixed modelling protocol.

Everyone runs the same split, the same targets, the same threshold and the same models, so the
only thing that varies between two people's numbers is their pre-processing function. Results
are appended to experiments_log.csv.

    from testing_pipeline import run_experiment

    def my_idea(X_train, Y_train, X_val):
        ...
        return X_train, Y_train, X_val

    run_experiment(author="John", idea_name="Drop_Feature10", preprocess_func=my_idea)

    # one model while iterating, and score with and without the additive rules
    run_experiment(..., models='br-lgbm', rules='both')

What is frozen (do not change, or results stop being comparable):
  split      E.setup(seed=42, val_size=0.2), grouped on raw CRM configuration
  targets    consensus targets from raw train.csv -- scoring never sees your pre-processing
  dedup      on the RAW configuration, before your function runs, so every idea starts from
             the same 53,947 training rows
  threshold  0.5, additive rules applied on the RAW validation CRM matrix

Baselines with pre-processing = identity, seed 42
(data/derived/sprint2_all_models_comparison.csv):
                      no rules     with rules
    ovr-l1-lr                -     86.4367 %
    cooc-chain-lgbm   87.5141 %    87.7092 %
    br-lgbm           87.6543 %    87.8411 %
"""
import datetime
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src import data, experiment as E, metrics as M
from src.models import BinaryRelevanceLGBM, CoOccurrenceChainLGBM

LOG_FILE = os.path.join(data.ROOT, "experiments_log.csv")
CACHE_DIR = os.path.join(data.DATA_DIR, "cache", "experiments")

SEED = 42
THRESHOLD = 0.5
CHAMP = dict(min_child_samples=20, reg_lambda=1.0)

# Written in this order. Anything added here must also be produced in _result_row.
LOG_COLUMNS = [
    'Date', 'Author', 'Idea_Name', 'Model', 'Rules_Applied',
    'EMR', 'EMR_vs_raw', 'Hamming_Loss', 'Bit_Accuracy',
    'F1_micro', 'F1_samples', 'F1_macro', 'Median_perlabel_F1',
    'Precision_micro', 'Recall_micro', 'ROC_AUC_micro', 'PR_AUC_micro',
    'TP', 'FN', 'FP', 'TN',
    'Share_dist1', 'Share_dist2', 'Share_dist3plus',
    'Train_Rows', 'N_Features', 'Train_Seconds', 'Seed', 'Input_Hash',
]


def build_models():
    """Instantiated fresh per experiment so no state leaks between runs."""
    return {
        'ovr-l1-lr': OneVsRestClassifier(
            LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=SEED),
            n_jobs=-1),
        'br-lgbm': BinaryRelevanceLGBM(CHAMP),
        'cooc-chain-lgbm': CoOccurrenceChainLGBM(CHAMP, n_parents=30),
    }


def init_log_file():
    """Ensures the log file exists and has the correct headers.

    Rows are appended with header=False, so a log written under an older column set would be
    silently misaligned. If the header does not match, archive it rather than corrupt it.
    """
    if os.path.exists(LOG_FILE):
        existing = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
        if existing == LOG_COLUMNS:
            return
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        backup = f'{LOG_FILE}.bak-{stamp}'
        os.rename(LOG_FILE, backup)
        print(f"note: experiments_log.csv had an older column set -> archived as "
              f"{os.path.basename(backup)}")
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_FILE, index=False)


def _fingerprint(X_tr, Y_tr, X_va):
    """Content hash of what the model will actually see.

    Keyed on the arrays rather than on idea_name: two people who write the same transformation
    under different names share a cache hit, and editing your function invalidates it. A cache
    key that ignored the data would silently return the previous idea's numbers.
    """
    h = hashlib.blake2b(digest_size=8)
    for A in (X_tr, Y_tr, X_va):
        A = np.ascontiguousarray(A)
        h.update(str(A.shape).encode())
        h.update(str(A.dtype).encode())
        h.update(A.tobytes())
    return h.hexdigest()


def _check_contract(X_tr, Y_tr, X_va, n_val_expected, n_train_before):
    """Reject the pre-processing mistakes that produce a plausible-looking wrong number."""
    if X_va.shape[0] != n_val_expected:
        raise ValueError(
            f"pre-processing returned {X_va.shape[0]} validation rows, expected {n_val_expected}. "
            "Never add or drop validation rows -- scoring is against the full fixed fold.")
    if X_tr.shape[1] != X_va.shape[1]:
        raise ValueError(
            f"train has {X_tr.shape[1]} features but validation has {X_va.shape[1]}. "
            "Whatever you do to the training columns must be done to the validation columns.")
    if X_tr.shape[0] != Y_tr.shape[0]:
        raise ValueError(
            f"X_train has {X_tr.shape[0]} rows but Y_train has {Y_tr.shape[0]}. "
            "If you filter rows, filter X and Y with the same mask.")
    if X_tr.shape[0] == 0:
        raise ValueError("pre-processing removed every training row.")
    dropped = n_train_before - X_tr.shape[0]
    if dropped:
        print(f"  note: {dropped:,} training rows dropped ({dropped / n_train_before:.1%})")


def _micro_aucs(Y_true, P):
    """ROC-AUC and PR-AUC over all 26.6M cells. ~10s, and independent of threshold and rules,
    so it is computed once per model rather than once per rules variant."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(Y_true).ravel(), np.asarray(P).ravel()
    return {'ROC_AUC_micro': float(roc_auc_score(y, p)),
            'PR_AUC_micro': float(average_precision_score(y, p))}


def _result_row(d, pred, Y_va, proba_aucs, full_metrics):
    """EMR family always; the F1 / confusion block only when full_metrics."""
    res = E.score(d, pred)
    row = {
        'EMR': res['emr'], 'EMR_vs_raw': res['emr_vs_raw'],
        'Hamming_Loss': res['hamming_loss'], 'Bit_Accuracy': 1 - res['hamming_loss'],
        'Share_dist1': res['share_dist1'], 'Share_dist2': res['share_dist2'],
        'Share_dist3plus': res['share_dist3plus'],
        'F1_micro': np.nan, 'F1_samples': np.nan, 'F1_macro': np.nan,
        'Median_perlabel_F1': np.nan, 'Precision_micro': np.nan, 'Recall_micro': np.nan,
        'ROC_AUC_micro': np.nan, 'PR_AUC_micro': np.nan,
        'TP': np.nan, 'FN': np.nan, 'FP': np.nan, 'TN': np.nan,
    }
    cm = None
    if full_metrics:
        summary, cm, _ = M.multilabel_report(Y_va, pred, proba=None)
        row.update({
            'F1_micro': summary['F1 micro'], 'F1_samples': summary['F1 samples'],
            'F1_macro': summary['F1 macro'],
            'Median_perlabel_F1': summary['median per-label F1'],
            'Precision_micro': summary['precision micro'],
            'Recall_micro': summary['recall micro'],
            'TP': int(cm.iloc[0, 0]), 'FN': int(cm.iloc[0, 1]),
            'FP': int(cm.iloc[1, 0]), 'TN': int(cm.iloc[1, 1]),
        })
        row.update(proba_aucs)
    return row, cm


def _print_result(model_name, use_rules, row, cm):
    tag = 'with rules' if use_rules else 'no rules  '
    print(f"\n[{model_name}] {tag}")
    print(f"  EMR                 {row['EMR']:.5f}      EMR vs raw labels {row['EMR_vs_raw']:.5f}")
    print(f"  bit accuracy        {row['Bit_Accuracy']:.5f}      Hamming loss      "
          f"{row['Hamming_Loss']:.5f}")
    if cm is None:
        return
    print(f"  F1 micro            {row['F1_micro']:.5f}      F1 samples        {row['F1_samples']:.5f}")
    print(f"  F1 macro            {row['F1_macro']:.5f}      median per-label  "
          f"{row['Median_perlabel_F1']:.5f}")
    print(f"  precision micro     {row['Precision_micro']:.5f}      recall micro      "
          f"{row['Recall_micro']:.5f}")
    if not np.isnan(row['ROC_AUC_micro']):
        print(f"  ROC-AUC micro       {row['ROC_AUC_micro']:.5f}      PR-AUC micro      "
              f"{row['PR_AUC_micro']:.5f}")
    print('  ' + cm.to_string().replace('\n', '\n  '))


def run_experiment(author, idea_name, preprocess_func, models=None, rules='both',
                   full_metrics=True, compute_auc=True, use_cache=True):
    """Executes the full pipeline to test a data pre-processing idea.

    models        subset of ('ovr-l1-lr', 'br-lgbm', 'cooc-chain-lgbm'); None runs all three.
                  Pass one name while iterating -- all three cost about 10 minutes.
    rules         True, False, or 'both'. The 42 additive rules are applied to the thresholded
                  predictions. 'both' logs one row each and costs almost nothing, since the
                  model is only fitted once.
    full_metrics  F1 micro/samples/macro, median per-label F1, precision/recall and the cell
                  confusion matrix. Adds ~5s per rules variant.
    compute_auc   micro ROC-AUC and PR-AUC. Adds ~10s per model (not per variant). Requires
                  full_metrics.
    use_cache     reuse probabilities when the same arrays + model were scored before.
    """
    print(f"\n{'=' * 50}")
    print(f"STARTING EXPERIMENT: {idea_name} (Author: {author})")
    print(f"{'=' * 50}")

    # 1. Load the immutable raw data and the split
    d = E.setup(seed=SEED)
    tr, va = d['tr'], d['va']

    X_tr_raw = d['X_train'][tr]
    Y_tr_raw = d['Y_cons'][tr]

    # The validation set (X and Y) must NEVER be filtered to ensure fair comparisons.
    # X_va_raw also stays intact because the additive rules are defined on raw CRM columns.
    X_va_raw = d['X_train'][va]
    Y_va = d['Y_cons'][va]

    # Baseline deduplication on the RAW configuration, before the custom pre-processing, so
    # every idea starts from the same training rows and only the features differ.
    X_tr_dedup, Y_tr_dedup, _ = E.dedup_configs(X_tr_raw, Y_tr_raw)

    # 2. APPLY THE CUSTOM PRE-PROCESSING FUNCTION
    print("Applying custom pre-processing...")
    X_tr_clean, Y_tr_clean, X_va_clean = preprocess_func(X_tr_dedup, Y_tr_dedup, X_va_raw)
    _check_contract(X_tr_clean, Y_tr_clean, X_va_clean, len(va), X_tr_dedup.shape[0])

    print(f"Final Training Shape: X={X_tr_clean.shape}, Y={Y_tr_clean.shape}")
    print(f"Final Validation Shape: X={X_va_clean.shape}")

    fp = _fingerprint(X_tr_clean, Y_tr_clean, X_va_clean)
    print(f"Input fingerprint: {fp}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    init_log_file()

    selected = build_models()
    if models is not None:
        wanted = {str(m) for m in np.atleast_1d(models)}
        missing = wanted - set(selected)
        if missing:
            raise KeyError(f"unknown model(s) {sorted(missing)}; available: {sorted(selected)}")
        selected = {k: v for k, v in selected.items() if k in wanted}

    if str(rules) == 'both':
        rules_variants = [False, True]
    else:
        rules_variants = [bool(rules)]

    # 3. Training and Evaluation Loop
    for model_name, model in selected.items():
        cache_path = os.path.join(CACHE_DIR, f"proba_{model_name}_{fp}_s{SEED}.npy")

        if use_cache and os.path.exists(cache_path):
            print(f"\n[{model_name}] cached -> {os.path.basename(cache_path)}")
            P_val, secs = np.load(cache_path), 0.0
        else:
            print(f"\n[{model_name}] Training...", flush=True)
            with E.Timer() as t:
                # liblinear wants a sparse matrix; the LightGBM wrappers take dense arrays.
                if model_name == 'ovr-l1-lr':
                    model.fit(csr_matrix(X_tr_clean), Y_tr_clean)
                    P_val = np.asarray(model.predict_proba(csr_matrix(X_va_clean)), np.float32)
                else:
                    model.fit(X_tr_clean, Y_tr_clean)
                    P_val = np.asarray(model.predict_proba(X_va_clean), np.float32)
            secs = t.s
            np.save(cache_path, P_val)
            print(f"[{model_name}] Training completed in {secs:.1f}s")

        # AUCs rank the raw probabilities, so they are unchanged by the threshold and by the
        # rules -- compute once and reuse across variants.
        aucs = {}
        if full_metrics and compute_auc:
            print(f"[{model_name}] computing micro AUCs...", flush=True)
            aucs = _micro_aucs(Y_va, P_val)

        for use_rules in rules_variants:
            # Threshold, then optionally apply the additive rules. These use the RAW validation
            # CRM matrix: the 42 rules are defined over original column names, so they must not
            # see a transformed feature space.
            pred = (P_val >= THRESHOLD).astype(np.uint8)
            if use_rules:
                pred = data.apply_additive_rules(pred, X_va_raw, d['crm_cols'], d['bil_cols'])

            # E.score compares against the fixed validation targets, never the pre-processed
            # ones, which is what keeps two people's numbers comparable.
            row, cm = _result_row(d, pred, Y_va, aucs, full_metrics)
            _print_result(model_name, use_rules, row, cm)

            pd.DataFrame([{
                'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Author': author, 'Idea_Name': idea_name, 'Model': model_name,
                'Rules_Applied': use_rules,
                'Train_Rows': int(X_tr_clean.shape[0]),
                'N_Features': int(X_tr_clean.shape[1]),
                'Train_Seconds': round(secs, 1), 'Seed': SEED, 'Input_Hash': fp,
                **row,
            }])[LOG_COLUMNS].to_csv(LOG_FILE, mode='a', header=False, index=False)

    print("\nResults successfully saved to", LOG_FILE)
