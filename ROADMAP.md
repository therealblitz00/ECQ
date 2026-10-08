# Roadmap: Parfois Product Similarity System

**Goal:** for each product colourway (`PROD_CLR_EQUIV`), return at least 4 similar products and explain why each one matches (visual and categorical reasons), so commercial buyers can check the result.

**Team:** 5 members plus AI coding agents. Agents do the work in small tasks, each with a clear owner, and a human reviews and merges every task. See [§ How agents should work](#how-agents-should-work).

**Repository:** branch [`parfois`](https://github.com/therealblitz00/ECQ/tree/parfois) of `therealblitz00/ECQ`. Setup for Windows and Mac is in `README.md`.

**Related docs:** `problem_description.md` (the brief), `data_description.md` (data dictionary and data-quality notes), `SPRINT1_GUIDE.md` (how to do the Sprint 1 review).

---

## Status (last updated 2026-10-07)

| Sprint | Roadmap phases | Status |
|---|---|---|
| **Sprint 1: data cleaning and validation** | 0 (partly), 1 (partly), 1b (rule-based part and human review) | 🟡 **In progress**: automatic checks done, 4-way manual review under way |
| Sprint 2: features | 1 (rest), 1b (CLIP pre-screen), 2a, 2b | ⚪ Not started |
| Sprint 3: similarity and evaluation | 3, 4 | ⚪ Not started |
| Sprint 4: explanations and delivery | 5, 6 | ⚪ Not started |

**Done in Sprint 1 so far:**
- `src/sprint1_preprocess.py` with three commands: `check` (automatic checks), `batch` (a deterministic, balanced 4-way review split with HTML review pages) and `merge` (applies the review decisions and logs every change).
- Automatic check results (`outputs/sprint1/check_report.md`):
  - no duplicate keys, and every sales row joins to a product;
  - 4 high-priority issues (invalid image path, corrupt image, 2 type conflicts between description and family);
  - 296 medium-priority items (252 generic images, 100 shared images, 25 descriptions without the colour, 3 images in another category's folder);
  - 1,100 colourways without an image.
- Repository, `requirements.txt` and double-click setup/review/merge scripts for Windows and Mac.

**Next:**
1. Team calibration: all 5 members review the same ~40 flagged items and compare decisions.
2. Each member reviews their batch and commits their CSV. Then run `merge`, which produces `data/processed/items_clean.parquet`, the Sprint 1 deliverable.
3. *(Recommended before or during the review)* CLIP pre-screen (Phase 1b) to rank images that probably show the wrong product type, which the rule-based checks cannot see.

---

## Architecture decision: no multi-agent system (MAS) in the product

**Decision:** build the similarity system as a **deterministic retrieval pipeline**, not as a multi-agent system. LLMs are optional and limited to one narrow job: turning the explanation into readable text. Build agents only after the core works (Phase 6), and only if a buyer-facing chat interface is actually needed.

**Why:**

| Question | Answer |
|---|---|
| What is the core problem? | Representation and nearest-neighbour search: features → embeddings → top-k. This is an ML problem, not a reasoning problem. |
| How big is the data? | About 10.5k colourways. All embeddings fit in memory, and exact search takes milliseconds. Nothing here needs to be orchestrated. |
| How is the output explained? | It can be calculated directly: how much each feature block contributes to the score, plus which attributes are shared. A computed explanation is more faithful than one an LLM writes after the fact. |
| What do buyers need? | Results that are reproducible, auditable and stable. Several agents talking to each other add randomness, cost, latency and failure points without making the matches better. |
| How will we evaluate it? | A deterministic pipeline can be scored offline (Phase 4). A MAS is much harder to evaluate and to debug. |

**When a MAS (or a single agent) is worth it:**
- **Buyer assistant (Phase 6, stretch goal):** a *single* agent with tools such as `find_similar(sku)`, `filter(category, price…)` and `explain(a, b)`, so buyers can ask *"show me cheaper alternatives to 167718_BU in gold"*. One agent with tools is enough. Several agents are only justified if the work later splits into independent jobs (for example trend reports plus assortment planning).
- **If the course or challenge explicitly rewards an agentic architecture:** put it on top of the finished pipeline as an interface layer. It should never sit in the path that computes the matches.

**For the development work:** use AI agents, but don't build a framework to coordinate them. A human lead, this roadmap, and tasks that run in parallel are enough (see below).

---

## Phases overview

| Phase | Outcome | Depends on | Can run in parallel with |
|---|---|---|---|
| 0. Setup | Repo, environment, conventions, `CLAUDE.md` | – | – |
| 1. Data foundation | Clean colourway-level table (`items_clean.parquet`) | 0 | 2a |
| 1b. Image audit | Image↔item link, mismatch flags, human review | 1 | 2a |
| 2a. Tabular/text features | Categorical, numeric and text embeddings | 1 | 1b, 2b |
| 2b. Visual features | Image embeddings and colour palettes | 1b | 2a |
| 3. Similarity engine | Top-k retrieval combining the blocks, with candidate filtering | 2a (2b optional at first) | – |
| 4. Evaluation | Offline metrics and a review set for buyers | 3 | 5 |
| 5. Explainability and output | Reasons for each match, in the required output format | 3 | 4 |
| 6. Delivery and interface | Batch output, demo UI, optional agent assistant | 4, 5 | – |

**Critical path:** 0 → 1 → 2a → 3 → 4/5 → 6. The visual branch runs 1 → 1b → 2b and joins at Phase 3.
**Biggest data risk:** images that **don't match their CSV row** (wrong type, wrong colour or wrong item; the professor's warning). Phase 1b audits the images before any of them are used for similarity. Items without a trustworthy image (about 10% have no file at all) fall back to similarity based on tabular data only.

---

## Phase 0: Setup

- [x] Git repository: branch `parfois`, with data, docs and `.gitattributes` for Windows and Mac line endings.
- [x] Python environment: `.venv` + `requirements.txt` (pandas, pyarrow, pillow), created by `1_setup_windows.bat` / `1_setup_mac.command`.
- [x] `README.md` with a quick start for Windows and Mac.
- [ ] Project layout for the next sprints:
  ```
  data/csv/          # original CSVs (read-only)
  data/images/       # product images, 9,496 files (read-only)
  data/processed/    # parquet outputs of each phase (not in git)
  src/               # sprint1_preprocess.py now; later a package: features/, retrieval/, explain/, eval/
  notebooks/         # exploration only, never imported
  tests/
  outputs/
  ```
- [ ] Add packages as phases need them: scikit-learn, numpy, sentence-transformers / open_clip, faiss-cpu (optional), pytest.
- [ ] `CLAUDE.md` with the project conventions agents must follow: data contracts, "never edit `data/csv` or `data/images`", how to run the tests, code style.
- [ ] CI that runs `pytest` and a linter.

**Done when:** `pytest` passes on an empty test suite, and an agent can read `CLAUDE.md` and work out where to put new code.

## Phase 1: Data foundation

- [x] Loader: UTF-8, whitespace and non-breaking-space (`\xa0`) cleanup, `COMPOSITION` split into `;`-separated materials (`sprint1_preprocess.py`). *The `�` characters reported earlier were a terminal display issue, not a data problem.*
- [ ] Date parsing for all three date formats.
- [ ] Drop the empty, constant and duplicate columns (list in `data_description.md` §3.3).
- [ ] Treat placeholders (`UNDEFINED`, `Not Applicable`, `Without Block`, `Others`) as missing values.
- [ ] Normalise casing (`THEME`, `L1_DES`…) and tokenise `COMPOSITION` into a list of materials.
- [x] **Collapse to colourway level** (`PROD_CLR_EQUIV`), with the list of sizes and a size-free description (`PROD_DES_BASE`).
- [ ] Check that the attributes are constant within each colourway.
- [x] Left-join the sales data.
- [ ] Add `log_sales_qty` and `realised_price` (guarding against division by zero).
- [x] Link each colourway to its image: take the stem of the last part of `PROG_IMAGE` and match it to a file stem in `data/images/` (9,455 / 10,555 = 89.6% coverage). Stored as `img_file` and `has_image`.
- [x] Missing-value and duplicate reports (`outputs/sprint1/`).
- [ ] Apply the Sprint 1 review decisions (`merge`): fixes, dropped images, discarded items.

**Output:** `data/processed/items_clean.parquet`, with one row per colourway (about 10,555 rows, minus discarded items).
**Done when:** the review is merged, and tests check that the key is unique, all sales rows are joined and the row count is stable.

## Phase 1b: Image audit and label consistency

Some images don't match their row (e.g. the row says necklace, the image shows earrings). This phase finds them **before** visual features are trusted. Details and first findings are in `data_description.md` §5.

**Done in Sprint 1 (rule-based, `sprint1_preprocess.py check`):**
- [x] Path validity, missing files and corrupt files.
- [x] Image file name vs model (`PROD_REF`), colour code (`CLR_COD`) and category folder.
- [x] Generic images (no colour in the file name).
- [x] Images shared across colours or models (by path and by MD5 hash).
- [x] Colour code ↔ colour name ↔ description, and description type ↔ family (`GFA_DES_EN`).
- [ ] Human review of all images in 5 batches (`SPRINT1_GUIDE.md`) → merge.

**Still to do (vision model):**
- [ ] **Install the vision stack:** `torch` (CPU) and `open_clip_torch`, or `transformers`. There is no GPU, so CLIP ViT-B/32 on CPU will take roughly 10–30 minutes for 9.4k images. Cache the embeddings to disk once and reuse them in Phase 2b.
- [ ] **Type check (zero-shot):** classify each image against text prompts for every `GFA_DES_EN` in its category, and also across categories (*"a photo of earrings"*, *"a photo of a necklace"*…). Flag the item when the label's probability is low **and** another class wins by a clear margin.
- [ ] **Neighbourhood check:** in CLIP image space, flag items whose k nearest visual neighbours mostly have a different `GFA_DES_EN`. This catches mismatches that the prompts miss.
- [ ] **Colour check:** compare the dominant colour (LAB k-means, background removed) with `CLR_DES`. Skip multicolour, gold and silver labels, which are too ambiguous.
- [ ] **Near-duplicate check:** perceptual hashes (pHash) on top of the exact hashes already done. Classify each group as same item re-coded (fine), same photo for different colours (colour unreliable) or different models (wrong item).
- [ ] **Quality check:** not a white-background packshot, very small, or several items in one picture.
- [ ] **Human validation:** review the top ~150 flags in a contact sheet (image + label) and record precision. Adjust the thresholds.
- [ ] **Decision per item:** made by the reviewers (`ok` / `fix` / `drop_image` / `discard` / `team_review`). Never relabel automatically.

**Output:** a CLIP mismatch score per item added to the review files, plus `outputs/image_audit_report.md`. The final status per item comes from the human review (`review_status` in `items_clean.parquet`).
**Done when:** every colourway has an audit status, and the manual precision of the flags is at least 80% on the review sample.
**Use downstream:** Phase 2b only embeds `ok` images (and `colour_unreliable` images, for shape only). Phase 3 falls back to tabular data for everything else. Phase 6 delivers the flag list to the business as a data-quality result in its own right.

## Phase 2a: Tabular and text features

- [ ] **Categorical block:** hierarchy (CAT › GFA › GFS, `CATEGORY_MATRIX`), colour (`CLR_DES`, `CLR_TYPE`), `FINISHING`, `MATERIAL`, `PRINT_TYPE`, `DIMENSION`, `OUTFIT`, `THEME`, `FASHIONTYPE`, `PROD_SEG`, `NUMBER_OF_UNITS`. Use one-hot or weighted encodings, and give the hierarchy levels larger weights.
- [ ] **Composition block:** multi-hot vector of material tokens.
- [ ] **Numeric block:** price band, launch season and date.
- [ ] **Text block:** sentence embedding of a description assembled from the attributes, e.g. *"Hoop earring, golden finish, pearl and zinc, Golden Delicates theme"*.
- [ ] Keep each block as a separate matrix with the same row order, so the score can later be broken down by block.

**Output:** `data/processed/features_{block}.npy` and `feature_meta.json`.

## Phase 2b: Visual features *(only images that pass the 1b audit)*

- [ ] Image embeddings with a pretrained vision model (CLIP / SigLIP / DINOv2), reusing the Phase 1b cache. Start without fine-tuning.
- [ ] Optional: compare a DINOv2 embedding (closer to shape and texture) with CLIP (closer to meaning) in the Phase 4 ablations.
- [ ] Dominant colour palette for each image (k-means in LAB colour space), so explanations can say things like *"same colour palette"*.
- [ ] Optional: zero-shot CLIP tags (shape, style, motif) to fill sparse attributes such as `SHAPE` and `PRODUCT_DETAILS`.

**Output:** `features_visual.npy`, `palettes.parquet`, `image_tags.parquet`.

## Phase 3: Similarity engine

- [ ] Per-block cosine similarity combined as a weighted sum: `score = Σ w_b · sim_b`. Load the weights from a config file.
- [ ] Candidate filtering: same `CAT_DES_EN` by default, with an option to restrict further to the same `GFA` family. Exclude the query item itself, and exclude other colours of the same `PROD_REF` unless asked for (otherwise they make the results trivially easy).
- [ ] Top-k search (k ≥ 4) for every item, run as a batch.
- [ ] Fallback for items without a trustworthy image (no file, or flagged in 1b): use only the tabular blocks and reweight. For `colour_unreliable` items, keep the shape signal from the image but take colour from the CSV.
- [ ] Never return a flagged item as a match because of its image alone.

**Output:** `outputs/neighbours.parquet` with columns (query, rank, candidate, total score, score per block).

## Phase 4: Evaluation

There is no labelled ground truth, so combine several signals:

- [ ] **Proxy metrics:** hit rate at k on held-out "known similar" pairs. These are pairs that share `THEME` + `GFS` + `CLR_DES` but have a different `PROD_REF`, or re-coded items linked by `*_EQUIV`.
- [ ] **Attribute agreement:** for the top-k results, the share with the same family, colour, finish or material.
- [ ] **Commercial sanity check:** similar items should have similar prices and sales profiles (correlation of log sales).
- [ ] **Ablations:** tabular only, visual only, and combined. Tune the block weights.
- [ ] **Human review set:** about 50 query items × top-5 results in a scoring sheet for buyers or the team.
- [ ] A fixed evaluation script, so every change reports the same numbers.

**Done when:** an `outputs/eval_report.md` is generated automatically, and the chosen weights are justified by the ablations.

## Phase 5: Explainability and output

- [ ] **Explanation for each match**, calculated from the data:
  - the blocks that contributed most to the score (e.g. *visual 0.45, category 0.30, material 0.15*);
  - attributes shared with the query (*same family: Hoop Earrings; same finish: Golden; shared materials: Zinc, Pearl*);
  - the main difference (*price €8.99 vs €12.99, different colour*);
  - visual evidence: palette swatches, and an optional Grad-CAM image or patch-similarity overlay.
- [ ] Optional: an LLM writes a one-sentence summary **using only the computed facts** (no new claims).
- [ ] Output in the format the brief requires: query → ≥4 SKUs, each with its matching attributes. Export as JSON and CSV.

## Phase 6: Delivery and interface

- [ ] Batch CLI: `parfois-sim build` and `parfois-sim query 167718_BU`.
- [ ] Demo UI (Streamlit, or a static HTML report): the query image next to the matches, with the explanations and sales data.
- [ ] Final report or slides: method, evaluation, limitations, business use cases (finding substitutes, overstock risk, trends).
- [ ] *(Stretch)* **Buyer assistant agent:** a single LLM agent with tools wrapping the engine (`find_similar`, `filter`, `explain`, `sales_summary`), plus a few test conversations.
- [ ] *(Stretch)* Trend clustering: cluster the embeddings per season and track which clusters grow.

---

## How agents should work

We don't need a MAS framework to build this. The lead (a human or a main Claude session) sends off tasks and reviews the results.

0. **Humans vs agents:** the 5 team members own decisions and the manual review. Agents write and test the code for each checklist item.
1. **One task = one checklist item** above, with a clear input, an output file, and tests that define "done".
2. **Data contracts:** phases only talk to each other through files in `data/processed/` whose schemas are documented. An agent working on Phase 3 should never need to read Phase 1 code.
3. **Tasks that can run in parallel:** 1b ∥ 2a, 2a ∥ 2b, and 4 ∥ 5. Give each agent its own branch or worktree.
4. **Context for every agent:** `CLAUDE.md`, `data_description.md` and this roadmap. Give it the relevant phase section, not the whole history.
5. **Review step:** tests pass, the evaluation report doesn't get worse, and a human reads the diff before merging.
6. **No agent edits `data/csv/` or `data/images/`, and none changes another phase's output schema without updating the contract.**

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Images don't match their CSV row | Wrong matches that look confident | Phase 1b audit. Flagged items use tabular data only, and the flags are reported to the business |
| About 10% of colourways have no image (Apparel 17%) | Visual block missing | Fall back to tabular data. Report coverage per category |
| No ground truth | Hard to prove quality | Proxy metrics plus the human review set (Phase 4) |
| Results dominated by trivial matches (same model, different colour) | Useless for buyers | Exclude by default the same `PROD_REF` and the same `THEME` + colour variants |
| Sparse attributes that only apply to one category | Noisy similarity across categories | Search within one category, and encode "not applicable" explicitly |
| LLM explanations make things up | Buyers lose trust | Explanations are computed. The LLM only rephrases them |
