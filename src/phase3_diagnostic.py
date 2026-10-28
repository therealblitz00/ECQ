"""Phase 3 Diagnostic & Improvement Simulation Script.

Tests and compares alternative retrieval mechanisms:
  0. Current Engine Baseline (Linear Late Fusion, No Diversity Cap)
  1. Diversity Cap (Max 1 colourway per candidate PROD_REF)
  2. Softened ERP Penalties (Reduced CLR_DES & THEME weights in categorical block)
  3. Two-Stage Retrieval (Top-N visual candidates -> Attribute/Material Re-ranking)
  4. Two-Stage Retrieval + Diversity Cap

Evaluates impact on:
  - The Parfois Brief example (167718_BU) ranks
  - Recommendation diversity (distinct models in top 5)
  - Visual agreement vs Catalogue agreement across a test sample
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import phase1_clean
import phase3_engine

ROOT = config.ROOT
KEY = config.KEY
BRIEF_QUERY = "167718_BU"
BRIEF_ANSWERS = ["169607_IG", "169607_PE", "169597_FU", "177781_LB"]


def load_all_data():
    items = phase1_clean.load_items()
    weights = phase3_engine.load_weights()
    img_name = phase3_engine.image_embedding()
    blocks = phase3_engine.load_blocks(items, image=img_name)
    has = phase3_engine.has_block(blocks)
    return items, weights, blocks, has


def build_softened_categorical(items: pd.DataFrame, blocks: dict) -> np.ndarray:
    """Softens CLR_DES, THEME, and FINISHING weights inside the categorical matrix."""
    meta = json.loads((config.PROCESSED_DIR / "feature_meta.json").read_text(encoding="utf-8"))
    cat_block = blocks["categorical"].copy()
    columns = meta["blocks"]["categorical"]["columns"]

    # Scale down specific attributes
    # Original weights: CLR_DES=2.0, THEME=1.5, FINISHING=1.5
    scales = {
        "CLR_DES": np.sqrt(0.5 / 2.0),
        "THEME": np.sqrt(0.3 / 1.5),
        "FINISHING": np.sqrt(0.5 / 1.5),
    }
    for prefix, factor in scales.items():
        indices = [i for i, col in enumerate(columns) if col.startswith(f"{prefix}=")]
        if indices:
            cat_block[:, indices] *= np.float32(factor)

    # Re-normalize rows to unit length
    norms = np.linalg.norm(cat_block, axis=1, keepdims=True)
    return np.divide(cat_block, norms, out=np.zeros_like(cat_block), where=norms > 0).astype(np.float32)


def rank_with_diversity(candidate_keys: list[str], scores: np.ndarray, items_df: pd.DataFrame,
                        max_per_model: int = 1, k: int = 5) -> list[tuple[str, float, int]]:
    """Selects top-k candidates enforcing a maximum number of items per PROD_REF."""
    it = items_df.set_index(KEY)
    order = np.argsort(-scores)
    selected = []
    seen_models: dict[str, int] = {}

    for idx in order:
        if not np.isfinite(scores[idx]):
            continue
        cand_key = candidate_keys[idx]
        score = scores[idx]
        prod_ref = str(it.loc[cand_key, "PROD_REF"])

        count = seen_models.get(prod_ref, 0)
        if count < max_per_model:
            selected.append((cand_key, float(score), len(selected) + 1))
            seen_models[prod_ref] = count + 1
            if len(selected) >= k:
                break

    return selected


def evaluate_query(query_key: str, items: pd.DataFrame, blocks: dict, has: dict, weights: dict,
                   mode: str = "baseline", max_per_model: int = 999, k: int = 5,
                   stage1_pool_size: int = 50):
    """Evaluates candidates for a single query under different retrieval strategies."""
    it = items.set_index(KEY)
    q_idx = items.index[items[KEY] == query_key][0]
    cat = it.loc[query_key, "CAT_DES_EN"]

    # Candidate pool: same category, exclude query & query siblings
    pool_mask = (items["CAT_DES_EN"] == cat).to_numpy()
    pool_indices = np.flatnonzero(pool_mask)

    # Exclude query itself and same model
    q_ref = it.loc[query_key, "PROD_REF"]
    q_equiv = it.loc[query_key, "PROD_REF_EQUIV"]
    valid_mask = []
    for c_idx in pool_indices:
        if c_idx == q_idx:
            valid_mask.append(False)
        elif str(items.at[c_idx, "PROD_REF"]) == str(q_ref) and str(q_ref) != "":
            valid_mask.append(False)
        elif str(items.at[c_idx, "PROD_REF_EQUIV"]) == str(q_equiv) and str(q_equiv) != "":
            valid_mask.append(False)
        else:
            valid_mask.append(True)

    valid_pool = pool_indices[valid_mask]
    cand_keys = items.iloc[valid_pool][KEY].tolist()

    # Mode logic
    if mode == "baseline" or mode == "diversity_only" or mode == "softened_cat":
        # Standard weighted score
        scores = phase3_engine.score_matrix(blocks, has, weights, np.array([q_idx]), valid_pool)[0]

    elif mode.startswith("two_stage"):
        # Stage 1: Visual similarity ranking
        if has["image"][q_idx] > 0:
            vis_scores = blocks["image"][q_idx] @ blocks["image"][valid_pool].T
            vis_scores = np.where(has["image"][valid_pool] > 0, vis_scores, -np.inf)
        else:
            vis_scores = blocks["categorical"][q_idx] @ blocks["categorical"][valid_pool].T

        # Top N candidates by Stage 1
        top_stage1_idx = np.argsort(-vis_scores)[:stage1_pool_size]
        stage1_candidates = valid_pool[top_stage1_idx]

        # Stage 2: Re-rank only the Stage 1 winners using composite weights
        rerank_scores = phase3_engine.score_matrix(blocks, has, weights, np.array([q_idx]), stage1_candidates)[0]

        # Build score vector aligned with cand_keys (-inf for items not in stage 1)
        scores = np.full(len(valid_pool), -np.inf, dtype=np.float32)
        scores[top_stage1_idx] = rerank_scores

    else:
        raise ValueError(f"Unknown mode {mode}")

    # Get ranks of all candidates
    sorted_order = np.argsort(-scores)
    all_ranks = {cand_keys[idx]: rank + 1 for rank, idx in enumerate(sorted_order)}

    # Get top-k with optional diversity filter
    top_k = rank_with_diversity(cand_keys, scores, items, max_per_model=max_per_model, k=k)

    return all_ranks, top_k


def run_diagnostic():
    print("=" * 80)
    print("PHASE 3 DIAGNOSTIC & IMPROVEMENT SIMULATION")
    print("=" * 80)

    items, weights, blocks, has = load_all_data()
    print(f"Loaded {len(items):,} items. Image encoder: {phase3_engine.image_embedding()}")
    print("Weights:", weights)

    # Create softened categorical block
    blocks_soft = blocks.copy()
    blocks_soft["categorical"] = build_softened_categorical(items, blocks)

    configs = [
        ("0. Baseline (Sprint 3 Current)", blocks, weights, "baseline", 999),
        ("1. Baseline + Diversity Filter (Max 1/model)", blocks, weights, "diversity_only", 1),
        ("2. Softened ERP + Diversity (Lower CLR/THEME)", blocks_soft, weights, "softened_cat", 1),
        ("3. Two-Stage (Visual Top 50 -> Re-rank)", blocks, weights, "two_stage", 999),
        ("4. Two-Stage + Softened + Diversity", blocks_soft, weights, "two_stage", 1),
    ]

    print("\n" + "#" * 80)
    print("TEST CASE: THE BRIEF'S EXAMPLE (167718_BU - Beaded Dangle Earring)")
    print("#" * 80)

    it = items.set_index(KEY)
    q_info = it.loc[BRIEF_QUERY]
    print(f"QUERY: {BRIEF_QUERY} | {q_info['GFS_DES_EN']} | {q_info['THEME']} | {q_info['FINISHING']} | {q_info['CLR_DES']} | {q_info['PRICE_BASE_W_VAT']} €")
    print("-" * 80)

    brief_results = {}
    for name, blk, w, mode, max_m in configs:
        all_ranks, top_k = evaluate_query(BRIEF_QUERY, items, blk, has, w, mode=mode, max_per_model=max_m, k=5)
        brief_ranks = [all_ranks.get(ans, 9999) for ans in BRIEF_ANSWERS]
        brief_results[name] = (brief_ranks, top_k)

        print(f"\n[{name}]")
        print(f"  Brief answers ranks: {brief_ranks} (Answers: {BRIEF_ANSWERS})")
        print("  Top 5 Recommendations:")
        distinct_models = len(set(it.loc[c[0], "PROD_REF"] for c in top_k))
        for key, sc, rk in top_k:
            row = it.loc[key]
            print(f"    #{rk} {key} (score {sc:.3f}) | Model {row['PROD_REF']} | {row['GFS_DES_EN'][:15]:15} | {row['THEME'][:16]:16} | {row['CLR_DES'][:10]:10} | {row['PRICE_BASE_W_VAT']} €")
        print(f"  Distinct Models in Top 5: {distinct_models} / 5")

    print("\n" + "#" * 80)
    print("BATCH BENCHMARK OVER 50 RANDOM TEST QUERIES (Jewellery, Apparel, Bags)")
    print("#" * 80)

    rng = np.random.default_rng(42)
    sample_queries = rng.choice(items.loc[items["img_trusted"], KEY].to_numpy(), size=50, replace=False)

    summary = []
    for name, blk, w, mode, max_m in configs:
        distinct_counts = []
        visual_cosines = []
        for q in sample_queries:
            q_idx = items.index[items[KEY] == q][0]
            _, top_k = evaluate_query(q, items, blk, has, w, mode=mode, max_per_model=max_m, k=5)
            models = set(it.loc[c[0], "PROD_REF"] for c in top_k)
            distinct_counts.append(len(models))

            # Measure visual cosine of top matches
            c_indices = [items.index[items[KEY] == c[0]][0] for c in top_k]
            v_cos = blk["image"][q_idx] @ blk["image"][c_indices].T
            visual_cosines.append(float(v_cos.mean()))

        summary.append({
            "Configuration": name,
            "Avg Distinct Models (Top 5)": np.mean(distinct_counts),
            "Avg Visual Cosine": np.mean(visual_cosines),
            "Brief Best Rank": min(brief_results[name][0]),
            "Brief Rank Sum": sum(brief_results[name][0]),
        })

    summary_df = pd.DataFrame(summary)
    print("\nSUMMARY COMPARISON TABLE:")
    print(summary_df.to_string(index=False))
    print("\nDiagnostic complete.")


if __name__ == "__main__":
    run_diagnostic()
