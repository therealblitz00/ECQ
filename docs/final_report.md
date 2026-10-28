# Parfois: similar products, sales and price estimates (final report)

MADSAD case study (FEP, University of Porto), team: André, Pedro Correia, Pedro Meireles, Manuel, Zé. The delivery is
this repository; this page is the entry point. Every number below comes from a file in `outputs/` written by a script in
`src/`; the notebooks redo each step without calling project code.

| Read next | For |
|---|---|
| [notebooks/phase1_preprocessing.ipynb](../notebooks/phase1_preprocessing.ipynb) | Sprint 1: data, checks, cleaning |
| [notebooks/phase2_features.ipynb](../notebooks/phase2_features.ipynb) | Sprint 2: features, engine, evaluation, weights from the team |
| [notebooks/phase3_explanations_estimates.ipynb](../notebooks/phase3_explanations_estimates.ipynb) | Sprint 3: explanations, sales and price estimates, improvements |
| [docs/progress.md](progress.md) | Every step with its numbers (§25: Sprint 3) |
| [docs/roadmap.md](roadmap.md) | Status and decisions |

## 1. The problem and what is delivered

The brief (Parfois slides, `docs/problem.md`) asks, **for each product, at least 4 similar products and the most relevant
characteristics for a business person to evaluate**, so that buyers can estimate quantities for new products and avoid
overstock of similar items. In class, the professor added the expected unit price and the stock for a sales period,
derived from the similar products (to be confirmed in writing, §9).

| Output | File | One row per |
|---|---|---|
| 5 similar products with the reasons (brief's format) | `outputs/phase5/similar_products.csv` | product × match (52,775) |
| Expected whole-life units and price, 80% range, similar past products, reasons | `outputs/phase5b/estimates.csv` | product (10,555) |
| A buyer's view (photos, shares, reasons, estimates) | `outputs/phase6/report.html`, `python src/phase6_deliver.py query KEY` | product |

## 2. Data and cleaning (Sprint 1)

- 17,125 SKUs → **10,555 colourways** (model × colour): the unit of the brief's example and of the sales and photos.
- 21 automatic checks; image pre-screen with AI (another item type, another colour, a shared photo); every product
  reviewed by a team member in a browser app.
- **The preprocessing rule** (table, description and photo must describe the same article; fix the one that disagrees):
  the table was fixed (147 products: 141 colours, 10 family/sub-family, 1 finish), wrong photos were unlinked (118),
  **0 products discarded**. The free-text descriptions were not edited: the text block of the engine is rebuilt from the
  fixed attributes (`src/phase2a_text.py`), so it follows the fixes. Every change is logged
  (`outputs/phase1/changes_log.csv`).

## 3. Features and the similarity engine (Sprint 2)

Six blocks, each a vector per product: **photo** (image embedding), **catalogue attributes** (one-hot), **description**
(sentence embedding of the attributes), **materials** (composition), **photo colours** (palette), **price and launch**.
The score is a weighted mean of the block cosines, among products of the same category, never another colour of the same
model. The weights were learned from the team: each member sorted the engine's candidates for 28 queries (138 answers on
99 queries), and a Bradley–Terry fit gave the photo more weight (0.36 → 0.62; a 25% blend applied, 0.42, because the full
weights cost 9 points of material agreement).

## 4. Evaluation

There is no ground truth. Three signals, strongest first:

| Signal | Sprint 2 (CLIP photos) | Sprint 3 (FashionSigLIP photos) |
|---|---|---|
| Team's preferences ordered right, 5-fold cross-validation | 72.8% | 74.0% (+1.2, CI −0.8 to +3.0: **not significant**) |
| Photo alone: 4 nearest photos of the same sub-family / material | 57.7% / 30.3% | 68.1% / 39.1% |
| `hit@4` / same sub-family / same material (top 4) | 91.5% / 90.2% / 73.9% | 91.0% / 89.7% / 74.5% |
| Sales correlation of a product with its matches | 0.618 | 0.614 |

- **Ceiling:** two team members agree on 70% of the pairs, so no weights can do much better on the team's judgements.
- **FashionSigLIP is kept** (better at product type and material from the photo, no guard down by more than 1 point),
  but its gain on the team's judgements is within the noise.
- **The sales correlation is partly biased** by the cut 242 sales (`outputs/phase4/sales_corr_check.md`): on whole lives
  only it is 0.541 (0.552 for Sprint 2); the launch month alone gives 0.363 as measured, 0.144 on whole lives. The ranking
  of the two engines does not change.
- Rejected with evidence (`progress.md` §25): standardised fusion, lower colour/theme weights, DINOv2, a diversity cap,
  blind automatic re-cropping.

## 5. Explanations

Each score is a weighted sum of dot products of named slices, so every match is explained by an **exact
decomposition** (no surrogate): the share of each block, the shared attributes and materials with their points, and the
differences (price, colour, finish, sub-family). Complete to 1e-6 on every match. The photo is the block that best
separates what the team calls similar (Cohen's d 0.96). Details: `outputs/phase5/explain_report.md`.

## 6. Sales and price estimates

**Sales are whole-life units.** Sales are accumulated until the extraction (about January 2025). Seasons 232 and 241 had
finished selling; 242 had not (October launches: a median of 143 units against about 3,300). So the model is trained on
232 and tested on the new models of 241 (one random half for the choices, the other half reported once); 242 only checks
the ranking among products launched in the same month. **Price = landed cost × markup**, tested on 242 (list prices are
not cut). Buyer decisions (`PUR_BET`, `TYPOLOGY`) are never features.

| Sales, test (1,436 new products of 241) | Ranking, same launch month | Error factor | Within ×2 | Top-20% precision |
|---|---|---|---|---|
| Category median | 0.00 | ×2.09 | 67% | 22% |
| **Similar past products** (engine, 10 nearest) | 0.33 | ×2.03 | 66% | 32% |
| **Gradient boosting** (attributes + photo) | **0.50** | **×1.69** | **76%** | 37% |

| Price, test (3,728 new products of 242) | Mean error | Exact list price | Right or next | Bias |
|---|---|---|---|---|
| Sub-family median | 4.16 € | 33% | 60% | −6.5% |
| **Similar past products** (engine, no price block) | 3.77 € | 31% | 68% | −5.6% |
| **Gradient boosting** (landed cost × markup) | **2.68 €** | **38%** | **78%** | −4.5% |

**Similar products or gradient boosting?** The similar past products are the method closest to the professor's
description (find similar items, then their sales and price), and every estimate lists them. Gradient boosting is more
accurate on the same products (paired bootstrap over models, 95% CI): same-month ranking **+0.10 to +0.22**, units error saved **+0.14 to +0.23** log units (sales, 241 test half); price error saved **+0.90 to +1.33 €** (242). Both intervals exclude 0, so the evidence that it errs less is strong; the similar products stay as the explanation a buyer can check, listed with every estimate.

- **Against the buyer** (`outputs/phase5b/buyer_report.md`): the model ranks the products as well as the buyer's own
  purchase depth and store tiers and errs less in units; model + buyer is better than either; when they disagree, the
  buyer is right in 55–63% of the pairs.
- **What drives them** (TreeSHAP): fashion type and collection type for sales (without them the error grows by about 0.2
  log units; their shares shift between seasons), the landed cost for price.
- **Biases:** sales −6% (mostly jewellery, not a single season growth, no correction testable with one whole-season
  transition); price −4.5% (markups rise between seasons; three corrections tried, none gains).
- **Range:** each estimate carries an 80% range from the test residuals (sales ×0.50 to ×2.33 of the estimate, price ×0.89 to ×1.23).
- **Stock for a sales period: not delivered yet.** The data has whole-life units only, no dates and no stores.
  `ITEM_LIFE_CYCLE_WEEK` (4–12 weeks) is not the selling time: season 241 stayed about 48 weeks at full price, with no
  correlation to it. The definition is a question to the professor (§9).

## 7. The brief's example

`outputs/phase6/brief_example.md` (photos side by side, attributes). The brief's four answers for `167718_BU` are beaded
earrings of theme DEGRADÊ, three with finish *Other* and one a stud; the engine ranks them 30, 56, 269 and 575 of 4,638.
The engine's top 5 are beaded dangle earrings with the query's golden finish and price (one of the same theme, two of
DEGRADÊ). The difference is one of criterion; one example cannot evaluate the engine. The brief's `117781_LB` does not
exist; `177781_LB` is used (its photo is the one on the slide).

## 8. Limitations

- The team's judgements: 99 queries, 70% agreement between members.
- Sales: accumulated units only (no weeks, no stores), 242 cut by the extraction, one whole-season transition (232 → 241)
  to test on.
- The planning classes that drive the sales (fashion type, collection type) shift between seasons; when Parfois assigns
  them is unknown.
- The engine's weights were tuned with the sales correlation in the objective, so the similar-products sales estimate
  is slightly optimistic.

## 9. Open questions for the professor

1. Is the brief's example a reference answer or an illustration of the format? Which criterion matters most (look,
   finish, theme)? Do several colours of one model count as separate suggestions?
2. Are the sales quantities and the price part of the deliverable?
3. What is the "sales period" for the stock (whole life, a fixed week or month, or `ITEM_LIFE_CYCLE_WEEK`)?

## 10. How to run

Setup and the light pipeline: `README.md`. Sprint 3 outputs (light environment, `requirements.txt` + LightGBM):

```bash
python src/phase5_explain.py build          # explanations of every match
python src/phase5b_forecast.py run          # sales and price estimates (~30 min) -> outputs/phase5b/
python src/phase5b_buyer.py run             # the buyer benchmark (~10 min)
python src/phase4_compare.py salescheck     # the sales-correlation check
python src/phase6_deliver.py report         # outputs/phase6/report.html
python src/phase6_deliver.py brief          # the brief's example
python -m pytest                            # data contracts and unit tests
```
