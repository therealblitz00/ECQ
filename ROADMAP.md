# Roadmap: Parfois Product Similarity System

**Goal:** for each product colourway (`PROD_CLR_EQUIV`), return at least 4 similar products and explain why each one matches (visual and categorical reasons), so commercial buyers can check the result.

**Team:** 5 members plus AI coding agents. Agents do the work in small tasks, each with a clear owner, and a human reviews and merges every task. See [§ How agents should work](#how-agents-should-work).

**Repository:** branch [`parfois`](https://github.com/therealblitz00/ECQ/tree/parfois) of `therealblitz00/ECQ`. Setup for Windows and Mac is in `README.md`.

**Related docs:** `problem_description.md` (the brief), `data_description.md` (data dictionary and data-quality notes), `SPRINT1_GUIDE.md` (how to do the Sprint 1 review).

---

## Status (last updated 2026-10-09)

Progress log with evidence: **[PROGRESS.md](PROGRESS.md)**.

| Sprint | Roadmap phases | Status |
|---|---|---|
| **Sprint 1: data cleaning and validation** | 0, 1, 1b | 🟡 **In progress**: automatic checks ✅, image pre-screen ✅, review app ✅. Manual review by the 5 members under way, then merge |
| Sprint 2: features | 1 (rest), 2a, 2b | ⚪ Not started (image embeddings already computed in 1b) |
| Sprint 3: similarity and evaluation | 3, 4 | ⚪ Not started |
| Sprint 4: explanations and delivery | 5, 6 | ⚪ Not started |

| Phase | Status | Done | Still to do |
|---|---|---|---|
| 0. Setup | 🟡 | Repo, environments, README, double-click scripts, `CLAUDE.md` | Tests/CI |
| 1. Data foundation | 🟡 | Loader, colourway table, sales join, image link, size-conflict check, reports | Merge the review, dates, placeholders, casing, drop empty columns, sales features |
| 1b. Image audit | 🟡 | Masks, CLIP embeddings, type/colour/duplicate/quality flags, review app | Human review; measure the flags' precision afterwards |
| 2a. Tabular/text features | ⚪ | – | Everything |
| 2b. Visual features | 🟡 | Image embeddings (`data/embeddings/image_clip.npy`) | Colour palettes, optional tags |
| 3–6 | ⚪ | – | Everything |

**Done in Sprint 1 so far:**
- **Automatic checks** (`src/sprint1_preprocess.py check`, results in `outputs/sprint1/check_report.md`):
  - no duplicate keys, and every sales row joins to a product;
  - 10,555 colourways: 29 high, 578 medium and 351 low priority, the rest without issues.
- **Image pre-screen (Phase 1b,** `src/phase1b_image_audit.py`, results in `outputs/phase1b/`**):**
  - masks and CLIP embeddings for all 9,457 photos;
  - 24 photos showing another item type, 132 another colour, 54 the same photo for another colour.
- **Review workflow:**
  - deterministic split into 5 batches: 1 André, 2 Pedro Correia, 3 Pedro Meireles, 4 Manuel, 5 Zé;
  - browser review app (`src/review_app.py`) with buttons, autosave and one-click submit;
  - `merge` applies the decisions and logs every change.
- **Fixed after Pedro Meireles's audit** (`SPRINT1_REVIEW_AND_ROADMAP.md`): BUG-001, BUG-002, BUG-003.
- **Repository** with data, docs and double-click scripts for Windows and Mac.

**Next:**
1. Team calibration: all 5 members review the same ~20 flagged products and agree on borderline cases.
2. Each member reviews their batch in the app and clicks **Submit**. Then one person runs `merge`, which produces `data/processed/items_clean.parquet`, the Sprint 1 deliverable.
3. Sprint 2: Phase 2a (tabular and text features), then Phase 3 (similarity engine).

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

## Compute strategy: encode once, reuse everywhere

**Problem:** the team has laptops without GPUs, and vision models are heavy. A 640×640 photo is about 1.2 million numbers, and comparing photos pixel by pixel is slow and doesn't capture similarity anyway.

**Decision:** every image (and every product description) is converted **once** into an **embedding**, a short vector of about 512 numbers that summarises what it shows. A vision transformer splits the image into patch *tokens* and summarises them into this vector, which is what "converting an image into a token" refers to. Everything after that works on these small vectors.

```
photo ──[mask: keep only the target item]──[CLIP/SigLIP, once, ~0.1 s per image]──► 512 numbers
description ──[text encoder, once]──► 384–512 numbers
```

| | Without embeddings | With embeddings |
|---|---|---|
| Heavy model on the processor | every comparison | **once per image** (≈10–20 min for all 9.5k on a laptop) |
| Size | 278 MB of photos | ≈**20 MB** (9.5k × 512 × float32) |
| Top-k similar items for one product | slow | **milliseconds** |
| All-pairs similarity (≈90 M pairs) | not feasible | ≈**1 s** with NumPy |

**Rules:**
1. **One machine computes, everyone reuses.** The heavy steps (masks, image and text embeddings) run once on one computer. The results are committed to `data/embeddings/` (tracked in git, unlike `data/processed/`). Teammates only load the files: they need no `torch` and no waiting.
2. **Order:** mask first (Phase 1b), then embed the masked crop (Phase 2b). The same embeddings serve the CLIP pre-screen, similarity, clustering and evaluation, so they are never recomputed per task.
3. **Reproducible:** every embedding file gets a sidecar `.json` with the model name and version, the input size, mask settings, date, and the row order (`PROD_CLR_EQUIV` list). Recompute only when the model or the masks change.
4. **Keep it light:** a base-size model (e.g. CLIP ViT-B/32 or SigLIP base), 224 px input, processed in batches. Store float16 if size matters (≈10 MB).

**Files:**

| File | Content | Produced in |
|---|---|---|
| `data/embeddings/masks.parquet` | bounding box, mask method (threshold / detector) and `target_found` per item | Phase 1b |
| `data/embeddings/image_clip.npy` + `.json` | one image vector per colourway (masked crop) | Phase 1b/2b |
| `data/embeddings/text.npy` + `.json` | one text vector per colourway (description built from attributes) | Phase 2a |

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
  data/embeddings/   # masks + image/text embeddings, computed once, committed to git
  src/               # sprint1_preprocess.py now; later a package: features/, retrieval/, explain/, eval/
  notebooks/         # exploration only, never imported
  tests/
  outputs/
  ```
- [x] Vision packages for the computing machine (`requirements-vision.txt`: torch CPU, transformers, scipy).
- [ ] Add packages as later phases need them: scikit-learn, sentence-transformers, faiss-cpu (optional), pytest.
- [x] `CLAUDE.md` with the project conventions agents must follow: data contracts, "never edit `data/csv` or `data/images`", how to run the tests, code style.
- [ ] CI that runs `pytest` and a linter.

**Done when:** `pytest` passes on an empty test suite, and an agent can read `CLAUDE.md` and work out where to put new code.

## Phase 1: Data foundation

- [x] Loader: UTF-8, whitespace and non-breaking-space (`\xa0`) cleanup, `COMPOSITION` split into `;`-separated materials (`sprint1_preprocess.py`). *The `�` characters reported earlier were a terminal display issue, not a data problem.*
- [ ] Date parsing for all three date formats.
- [ ] Drop the empty, constant and duplicate columns (list in `data_description.md` §3.3).
- [ ] Treat placeholders (`UNDEFINED`, `Not Applicable`, `Without Block`, `Others`) as missing values.
- [x] Tokenise `COMPOSITION` into a canonical, sorted list of materials.
- [ ] Normalise casing (`THEME`, `L1_DES`…).
- [x] **Collapse to colourway level** (`PROD_CLR_EQUIV`), with the list of sizes and a size-free description (`PROD_DES_BASE`).
- [x] Check that the attributes are constant within each colourway (`SKU_ATTR_CONFLICT`, `sku_conflicts`). COMPOSITION is put in a canonical order first.
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
- [x] Review app (`2_review_…` → `src/review_app.py`).
- [ ] Human review of all images in 5 batches (`SPRINT1_GUIDE.md`) → merge.

**Done 2026-10-09 (vision model, `src/phase1b_image_audit.py`, run on one machine in 86 min):**
- [x] **Vision stack:** `torch` (CPU), `transformers`, `scipy` (`requirements-vision.txt`, only for the computing machine).
- [x] **Item masks:** 9,096 packshots cropped by white-background threshold, 272 non-packshot photos and 89 white-background photos with a person (e.g. bag carried by a model) cropped with the OWLv2 detector, prompted with the CSV item type. SAM outlines were not needed for cropping, so they were left out.
- [x] **Image embeddings:** CLIP ViT-B/32 on the masked crop, 512 numbers per image → `data/embeddings/image_clip.npy` (9,457 × 512, float16, ≈10 MB) + `.json` (model, settings, row order).
- [x] **Type check:** zero-shot over 15 broad item types (earrings, necklace, ring, bracelet, keychain, phone case, bag, wallet, shoes, top, coat, dress, trousers, skirt, swimwear) plus agreement of the 10 nearest images. Confusions between small jewellery items (no sense of scale) are only doubts.
- [x] **Colour check:** zero-shot over 16 basic colours, with neighbouring colours tolerated. Strong only when very confident or when the photo is shared with another colour.
- [x] **Near-duplicates:** same model, other colour, same photo (perceptual hash + CLIP + product pixels).
- [x] **Quality:** not-a-packshot photos (model, lifestyle, amateur sample photos).
- [x] **Calibration on contact sheets** (`outputs/phase1b/*.jpg`): thresholds tuned after visual inspection. Flags went from 77 to 20 on a 300-item trial, and from 1,400 to about 700 on the full set.
- [x] **Flags merged into Sprint 1:** `check` adds them to `issues`, and the review app reads `data/embeddings/image_audit.parquet` directly. No batch files changed.

| Flag | Items | Severity |
|---|---|---|
| `VIS_TYPE_MISMATCH` | 24 | high |
| `VIS_COLOUR_MISMATCH` | 132 | medium |
| `IMG_NEAR_DUPLICATE` | 54 | medium |
| `VIS_TYPE_DOUBT` | 189 | low |
| `VIS_TARGET_NOT_FOUND` | 46 | low |
| `IMG_NOT_PACKSHOT` | 272 | low |

**Still to do:**
- [ ] Measure the precision of each flag from the reviewers' decisions after the merge (target ≥ 80% for high, ≥ 50% for medium), and adjust the thresholds with `python src/phase1b_image_audit.py --reflag` (seconds, no models needed).
- [ ] Optional: SAM outlines, if the colour check needs product-only pixels on model photos.

**Output:** `data/embeddings/{image_clip.npy, image_clip.json, masks.parquet, image_audit.parquet}` (committed) and `outputs/phase1b/image_audit_report.md` + contact sheets. The final status per item comes from the human review (`review_status` in `items_clean.parquet`).
**Done when:** every colourway has an audit status, and the manual precision of the flags is at least 80% on the review sample.
**Use downstream:** Phase 2b only embeds `ok` images (and `colour_unreliable` images, for shape only). Phase 3 falls back to tabular data for everything else. Phase 6 delivers the flag list to the business as a data-quality result in its own right.

## Phase 2a: Tabular and text features

- [ ] **Categorical block:** hierarchy (CAT › GFA › GFS, `CATEGORY_MATRIX`), colour (`CLR_DES`, `CLR_TYPE`), `FINISHING`, `MATERIAL`, `PRINT_TYPE`, `DIMENSION`, `OUTFIT`, `THEME`, `FASHIONTYPE`, `PROD_SEG`, `NUMBER_OF_UNITS`. Use one-hot or weighted encodings, and give the hierarchy levels larger weights.
- [ ] **Composition block:** multi-hot vector of material tokens.
- [ ] **Numeric block:** price band, launch season and date.
- [ ] **Text block:** sentence embedding of a description assembled from the attributes, e.g. *"Hoop earring, golden finish, pearl and zinc, Golden Delicates theme"*.
- [ ] Keep each block as a separate matrix with the same row order, so the score can later be broken down by block.

**Output:** `data/processed/features_{block}.npy` and `feature_meta.json`. The text embedding goes to `data/embeddings/text.npy` (committed, computed once).

## Phase 2b: Visual features *(only images that pass the 1b audit)*

- [x] Image embeddings with a pretrained vision model (CLIP ViT-B/32), computed on the **masked crop of the target item** (Phase 1b), so a model's face, clothes or other jewellery don't drive the similarity → `data/embeddings/image_clip.npy`.
- [ ] Optional: compare a DINOv2 embedding (closer to shape and texture) with CLIP (closer to meaning) in the Phase 4 ablations.
- [ ] Dominant colour palette for each image (k-means in LAB colour space, on the masked pixels only), so explanations can say things like *"same colour palette"*.
- [ ] Optional: zero-shot CLIP tags (shape, style, motif) to fill sparse attributes such as `SHAPE` and `PRODUCT_DETAILS`.

**Output:** `data/embeddings/image_clip.npy` + `.json` (committed), `palettes.parquet`, `image_tags.parquet`.

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
