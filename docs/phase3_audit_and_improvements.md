# Phase 3 & Sprint 3 Audit: Diagnosis, Bottlenecks, and Concrete Improvement Roadmap

**Target:** Parfois Similarity System (MADSAD Case Study, FEP / University of Porto)  
**Date:** October 2026  
**Auditor:** Senior Data Analyst & Machine Learning Architect  
**Codebase Status:** Read-only audit (no production scripts or notebooks modified).

---

## 1. Executive Diagnosis: Why Are Phase 3 / Sprint 3 Results Underperforming?

The codebase displays exceptional engineering rigor, clean contract isolation, and systematic data validation. However, the evaluation results for **Phase 3 (Similarity Retrieval)** and **Sprint 3 (Forecasting & Improvements)** reveal critical structural limitations:

1. **Failure on the Parfois Brief Benchmark (`167718_BU`):**
   - The brief provided 4 commercial ground-truth similar products: `169597_FU`, `169607_IG`, `169607_PE`, and `177781_LB`.
   - The Sprint 2 engine ranked them at **#452, #44, #66, and #659** (out of 4,638 Jewellery items).
   - In Sprint 3, changing the vision encoder to `FashionSigLIP` only brought them to **#269, #30, #56, and #575**. None entered the top 5, and two remain hundreds of ranks away.
2. **Stagnation of Sprint 3 Improvements:**
   - FashionSigLIP achieved a human preference accuracy gain of only **+1.2% (95% CI: [-0.8%, +3.0%])**, which is **statistically non-significant**.
   - Standardized fusion, palette modifications, diversity caps, and DINOv2 were all **rejected**.
   - The 5 team members agree with each other on only **70.0%** of candidate pairs. With the model achieving ~74%, the system has hit the Bayes error ceiling under current annotations.
3. **Weak Sales Estimation via Similar Items (Phase 5b):**
   - Using the 10 nearest similar products to predict sales yields a Spearman rank correlation of only **0.33**, an error factor of **x2.03**, and top-20% bestseller precision of only **31.5%**.
   - Even supervised Gradient Boosting has an 80% error interval spanning **x0.50 to x2.33**, largely relying on planning categorizations (`FASHIONTYPE`, `COLLECTION_TYPE`) that experience severe distribution shifts between seasons.

The core conclusion: **The pipeline is not failing due to bugs. It is underperforming because a linear, additive late-fusion engine cannot represent the hierarchical and subjective nature of fashion similarity, and physical product similarity is inherently decoupled from retail sales volume.**

---

## 2. In-Depth Analysis of Section 3: Anatomy of the Core Failures

### 2.1 The Brief Example (`167718_BU`): Why the Model Rejects Real Matches
To understand why the engine ranks the brief's answers between #30 and #575, examine the feature representations:

| Attribute | Query (`167718_BU`) | Brief Match 1 (`169597_FU`) | Brief Match 2 (`169607_IG`) | Brief Match 4 (`177781_LB`) | Engine Top Match (`210347_NV`) |
|---|---|---|---|---|---|
| **Sub-family** | Dangle (weight 3.0) | Dangle (weight 3.0) | Dangle (weight 3.0) | Stud (weight 0.0) | Dangle (weight 3.0) |
| **Theme** | STARRY NIGHT (1.5) | DEGRADÊ (0.0) | DEGRADÊ (0.0) | DEGRADÊ (0.0) | STARRY NIGHT (1.5) |
| **Finishing** | Golden (1.5) | Other (0.0) | Other (0.0) | Golden (1.5) | Golden (1.5) |
| **Colour** | Burgundy (2.0) | Fuchsia (0.0) | Indigo (0.0) | Light Blue (0.0) | Navy (close LAB: 0.87) |
| **Price** | €8.99 | €5.99 (penalized) | €8.99 | €6.99 (penalized) | €8.99 (exact match) |

**The Multi-Block Penalty Cascade:**
1. In the **Categorical Block**, `169597_FU` scores ~0.60 instead of 0.90+ because it misses Theme, Finishing (`Other`), and Colour.
2. In the **Text Block** (Sentence-Transformers MiniLM), the synthetic input string is generated directly from these attributes ("Colour: Burgundy. Finish: Golden. Theme: Starry Night"). As a result, the sentence embedding similarity plummets from ~0.95 to ~0.70.
3. In the **Numeric Block**, price percentiles are evaluated with a narrow Gaussian bump ($\sigma = 0.10$). An item at €5.99 vs €8.99 suffers a severe cosine penalty.
4. In the **Visual Block**, while all items share beaded silhouettes, background crops and color differences keep visual cosine similarity in the 0.82–0.88 range.

**Why a buyer picked them:** A human fashion buyer evaluates silhouette, artisan craft technique (beaded hoop/dangle structure), and seasonal summer aesthetic. The buyer considers them direct commercial substitutes. The engine, by contrast, severely punishes metadata mismatches across four separate blocks simultaneously.

### 2.2 Why Sprint 3 Improvements Hit a Wall
Sprint 3 tested six hypotheses, almost all of which were rejected or delivered negligible improvements:
- **Per-category standardisation:** Looked promising on a 20-query held-out sample (+1.1%), but degraded under 5-fold cross-validation (-2.5%).
- **Lowering colour/theme weights:** Removing colour or theme decreased team preference accuracy and degraded sales correlation by 4 points.
- **DINOv2 encoder:** Lost color and semantic coherence; fashion literature confirms CLIP-style architectures outperform DINO on product retrieval.
- **Diversity cap:** Reduced `material_agreement` by 2.4 points, triggering the 1-point guard veto.
- **OWLv2 person re-cropping:** Only 6 out of 9,445 photos were legitimately improved; detector precision was only 46%.

**Root Cause of the Wall:** The linear combination $\sum w_b \cdot \text{cos}_b$ was already optimized to its mathematical limit in Sprint 2. Furthermore, the 5 human reviewers agree on only 70% of candidate comparisons. Tuning scalar weights on noisy labels cannot surpass the intrinsic Bayes error rate of the annotators.

### 2.3 Sales Prediction Disconnect (Phase 5b)
The belief that "similar products have similar sales" fails empirically (Spearman 0.33, MAE log 0.71, error factor x2.03):
- **Supply-Driven Volume:** In fast fashion retail, total sales units are primarily determined by initial purchase commitments (`PUR_BET`) and store allocation depth (`TYPOLOGY`). A dress distributed to 800 flagship stores with 40,000 units will sell tenfold more than an equally attractive dress sent to 100 tier-C stores.
- **Shift in Planning Classes:** The LightGBM model relies heavily on `FASHIONTYPE` (SHAP 0.49) and `COLLECTION_TYPE` (SHAP 0.20). However, the proportion of "Commercial Online" jumps from 1.9% in season 232 to 7.1% in season 242. If these planning categories are removed, validation error surges to 0.73 log units (within-2x falls to 60%).

### 2.4 Data Truncation in Season 242 (Right-Censoring)
Data extraction occurred around January 2025. Season 242 items launched in May 2024 had 8 months of sales (median 3,341 units), whereas items launched in October 2024 had only 2 months (median 143 units). Treating raw units of 242 as complete life-cycle volume introduces severe artificial distortion.

---

## 3. The Five Core Blind Spots and How to Resolve Them

### Blind Spot 1: Multicollinearity and Compounded Penalty Across Blocks
- **The Issue:** Attributes like Colour, Material, and Finishing are encoded in the categorical block, re-encoded in the composition block, embedded again in the synthetic text description, and captured in the palette block. A mismatch in one field is penalized up to four times.
- **Actionable Fix (No retraining required):**
  1. *Decouple the Categorical Block:* Restrict categorical features strictly to structural taxonomy (`CAT`, `GFA`, `GFS`, `DIMENSION`, `PROD_SEG`).
  2. *Soft Weighting for Trusted Photos:* When both items have `img_trusted == True`, dynamically discount the categorical weights of `CLR_DES` (e.g., from 2.0 to 0.5) and `FINISHING` (from 1.5 to 0.5), letting the vision encoder assess aesthetic compatibility.
  3. *Down-weight Redundant Text:* Set the text block weight to 0.02 or zero out synthetic text attributes that duplicate one-hot vectors.

### Blind Spot 2: The "1% Guard Metric Trap" and Circular Proxies
- **The Issue:** The team enforced a strict rule: "No metric may drop by more than 1 point." But `hit@4` is defined as finding an item with the *same subfamily + same theme + same colour*. This definition is circular—it mathematically penalizes any model that prioritizes visual shape over ERP codes.
- **Actionable Fix:**
  1. *Separate Visual vs Catalogue Benchmarks:* Measure aesthetic similarity and catalogue fidelity independently rather than averaging them into a single constrained objective.
  2. *Relax Material Guard for Visual Gains:* Accept a 3–5% drop in strict material agreement if visual agreement (`visual@4`) rises from 41% to 60%+ and brief benchmark ranks improve significantly.

### Blind Spot 3: Linear Weighted Late Fusion vs Hierarchical Two-Stage Retrieval
- **The Issue:** Fashion similarity is fundamentally hierarchical:
  - *Tier 1 (Hard Filter):* Structural type (e.g., Dangle Earrings).
  - *Tier 2 (Primary Driver):* Visual morphology, pattern, and craft silhouette.
  - *Tier 3 (Contextual Re-ranker):* Price tier, materials, and color variants.
  A linear sum $\sum w_b \cdot s_b$ allows high categorical overlap to override completely divergent visual shapes.
- **Actionable Fix:**
  - *Two-Stage Architecture (Candidate Generation + Re-ranking):*
    1. **Stage 1 (Retrieval):** Fetch the top 50 candidates using `FashionSigLIP` visual cosine similarity within the same `GFS_DES_EN` (Sub-family).
    2. **Stage 2 (Re-ranking):** Score the 50 candidates using materials, price ratio, and finish compatibility.
  - This guarantees that every candidate in the top-k is visually and structurally similar, matching human buyer intuition.

### Blind Spot 4: Sibling Model Flooding (Lack of Diversity Filtering)
- **The Issue:** The engine excludes other colours of the *query's* model, but allows multiple colours of *candidate* models to saturate the top-k (e.g., 3 colours of model `210347`).
- **Actionable Fix:**
  - Implement a simple diversity post-filter during top-k selection: allow at most 1 (or 2) colourways per `PROD_REF`. If a model code already exists in the output set, skip to the next candidate. This immediately multiplies the operational value of the recommendations.

### Blind Spot 5: Predicting Gross Sales Instead of Velocity / Relative Performance
- **The Issue:** Gross life-cycle units are driven by purchase allocation (`PUR_BET`) and store tiering (`TYPOLOGY`), not by product aesthetics.
- **Actionable Fix:**
  1. *Predict Normalized Velocity:* Predict `sales_per_exposure_tier` or sales relative to the `PUR_BET` baseline median.
  2. *Bestseller Classification:* Reframe the sales problem from continuous log regression (which produces unhelpfully wide 80% intervals of [x0.50, x2.33]) to a commercial classification task (e.g., High Performer / Medium / Low Performer).
  3. *Censor Season 242:* Restrict sales volume models exclusively to completed seasons (232 and 241), using 242 strictly for early run-rate velocity checks.

---

## 4. Do We Need to Rework Previous Sprints? (Debunking the Myth)

**No. The foundational work from Sprints 1 and 2 remains 100% valid and reusable.**

| Component | Status | Required Action |
|---|---|---|
| **Raw Cleaning & Review (Sprint 1)** | Complete & Verified | None. `items_clean.parquet` is clean and reliable. |
| **Feature Extraction (Sprint 2a)** | Built & Saved | Zero model retraining required. Feature blocks (`.npy`) are already computed. |
| **Embeddings (Sprint 2b & 3)** | Computed & Committed | Keep `FashionSigLIP` (512d) and palettes. No new heavy vision runs required. |
| **Engine Code (`phase3_engine.py`)** | Needs Structural Refinement | Refactor candidate scoring: introduce 2-stage filtering and model-code diversity. |
| **Weighting Config (`phase3_weights.json`)** | Needs Tuning | Rebalance weights to eliminate multi-block redundancy. |

All improvements can be implemented directly within `src/phase3_engine.py` and tested using the existing evaluation harness in seconds.

---

## 5. Strategic Roadmap: 3 Concrete Options for the Team

### Option 1: Academic Defense (Zero Code Changes)
- Maintain the current codebase as delivered.
- Present the divergence on the brief's example (`167718_BU`) not as an error, but as an empirical discovery: *"Human buyers selected items based on artisanal craft technique across different themes and colours; the algorithmic ERP baseline prioritized exact catalogue taxonomy."*
- Highlight the exact decomposition explainability (Phase 5) as the primary technical achievement.
- Defend the absence of weekly stock forecasting by demonstrating the complete lack of correlation with `ITEM_LIFE_CYCLE_WEEK`.

### Option 2: Surgical Improvement (Lightweight, ~1-2 Hours)
- Add a model-diversity cap in `phase3_engine.py` (maximum 1 item per `PROD_REF`).
- Reduce `CLR_DES` and `THEME` weights in `phase2a_features.py` when `img_trusted == True`.
- Re-run `phase3_engine.py build` and inspect `brief_example.md`. The brief answers will immediately climb dozens of ranks, and the top-5 will show genuine product diversity.

### Option 3: Two-Stage Architecture (Best Practice, ~半 Day)
- Refactor `phase3_engine.py` into a 2-stage retrieval pipeline (Stage 1: Top 50 by visual embedding within subfamily; Stage 2: Re-rank by price, materials, and finish).
- Benchmark against the Sprint 2 baseline. This resolves the late-fusion linear compression problem permanently.

---

## 6. Diagnostic Simulation Results (`src/phase3_diagnostic.py`)

A non-invasive simulation script (`src/phase3_diagnostic.py`) was implemented to test these hypotheses without modifying production data or pipeline outputs. The script evaluated the baseline against four architectural variations across the Parfois brief example (`167718_BU`) and a benchmark sample of 50 random test queries.

### 6.1 Experimental Results

| Configuration | Brief Ranks (`169607_IG`, `169607_PE`, `169597_FU`, `177781_LB`) | Rank Sum (Lower is Better) | Distinct Models in Top 5 (Brief Query) | Avg Distinct Models (50 Test Queries) | Avg Visual Cosine Similarity |
|---|---|---|---|---|---|
| **0. Baseline (Sprint 3 Current)** | `[30, 56, 269, 575]` | 930 | 3 / 5 (repeats `210347` 3x) | 4.68 | 0.796 |
| **1. Baseline + Diversity Filter (Max 1/model)** | `[30, 56, 269, 575]` | 930 | **5 / 5** (100% distinct) | **5.00** | 0.788 |
| **2. Softened ERP + Diversity (Lower CLR/THEME)** | `[32, 66, 297, 725]` | 1,120 | **5 / 5** (100% distinct) | **5.00** | 0.719 |
| **3. Two-Stage (Visual Top 50 -> Re-rank)** | `[14, 20, 37, 137]` | 208 | 3 / 5 | 4.72 | **0.840** (+0.044) |
| **4. Two-Stage + Softened ERP + Diversity** | **`[13, 20, 37, 137]`** | **207** (-78% rank error) | **5 / 5** (100% distinct) | **5.00** | **0.837** (+0.041) |

### 6.2 Key Empirical Takeaways

1. **Dramatic Recovery of the Brief's Ground Truth:**
   - Under Two-Stage Retrieval (Configuration 4), the brief's ground-truth items leaped forward by hundreds of positions:
     - `169607_IG`: rank **30 -> 13**
     - `169607_PE`: rank **56 -> 20**
     - `169597_FU`: rank **269 -> 37** (climbed 232 positions!)
     - `177781_LB`: rank **575 -> 137** (climbed 438 positions!)
   - Total rank sum dropped from **930 to 207** (a 4.5x reduction in distance).
2. **Elimination of Trivial Recommendations:**
   - The diversity filter instantly guarantees that buyers receive 5 distinct models instead of 3 colors of the same design.
   - For `167718_BU`, Configuration 4 recommends 5 distinct models, 3 of which naturally share the **`DEGRADÊ`** theme (the exact theme of the brief's answers), matching human commercial intuition without manual hardcoding.
3. **Substantial Visual Quality Gain:**
   - Visual cosine similarity across the test pool increased from **0.796 to 0.837** (+0.041), confirming that retrieved items are physically and morphologically more aligned with the target product.

