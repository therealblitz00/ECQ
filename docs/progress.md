# Progress Log: Parfois Similarity Project

What has been done so far, with evidence, in date order. **Status and next steps are only in [roadmap.md](roadmap.md).**

---

## At a glance

| | |
|---|---|
| Products (SKUs / colourways) | 17,125 SKUs → **10,555 colourways** (model × colour), the unit of analysis |
| Photos | 9,496 files, **9,491 colourways (89.9%)** with a photo (33 wrong paths repaired) |
| Automatic checks | 21 checks on codes, text, images and sizes. **29 high, 578 medium, 351 low priority** products |
| Image AI | 9,490 photos masked and embedded (512 numbers each): 9,457 in 86 min on one laptop CPU, plus 33 repaired paths in 98 s |
| Image flags | **24** photos of another item type, **135** of another colour, **54** same photo for another colour |
| Review | 5 balanced batches (2,111 products each), browser app, autosave, one-click submit |
| Review merged | 10,290 ok, **147 fixed**, **118 wrong photos** unlinked, 0 discarded → `items_clean.parquet` (10,555 colourways) |
| Image flag precision | Colour flag **70%**, type mismatch 17%; image flags caught 47% of the problems in the fully reviewed batch |
| Features (Phase 2a) | 4 blocks: categorical (1,619 columns), composition (95 materials), numeric (price, launch, season), text (384-number sentence embedding) |
| Visual features (Phase 2b) | CLIP image embeddings + colour palettes for the 9,449 photos kept by the review |

---

## 1. Understanding the data (2026-10-07)

- **Data dictionary** for all 141 product columns and the sales file: [data.md](data.md).
- **Key findings:**
  - one row per SKU (size), so the work happens per colourway (`PROD_CLR_EQUIV`);
  - sales join with no orphans;
  - 33 columns are empty or constant;
  - hidden non-breaking spaces in the text.
- **Architecture decision:** a deterministic retrieval pipeline, **not a multi-agent system**: [roadmap.md § Architecture](roadmap.md#architecture-decision-no-multi-agent-system-mas-in-the-product).

## 2. Automatic checks (Phase 1, 2026-10-07 → 09)

- **Script:** `src/phase1_checks.py check`. It takes about 15 s on the full dataset.
- **Report:** [outputs/phase1/check_report.md](../outputs/phase1/check_report.md).
- **What it checks:**
  - **Keys and duplicates:** none found.
  - **Missing values:** a full table of empty and constant columns.
  - **Image paths and files:** 1 invalid path, 1 corrupt file, 1,100 colourways without a photo (1,064 after the path repair, see §9).
  - **Image file name vs product code, colour and category:** 252 generic photos, 100 shared photos.
  - **Colour code ↔ colour name ↔ description, and description type ↔ family.**
  - **Sizes of the same colourway that disagree:** 165 colourways, e.g. a different price or theme per size. The value most sizes share is used, and the product is flagged.
- **Audit by a teammate:** Pedro Meireles's AI audit ([archive/2026-10-09_phase1_audit.md](archive/2026-10-09_phase1_audit.md)) found 3 bugs. All were fixed on 2026-10-09.

## 3. Image pre-screen with AI (Phase 1b, 2026-10-09)

The professor warned that some photos don't match their row (e.g. the row says necklace, the photo shows earrings) and suggested **masks** for photos where a model wears several items.

**How:** `src/phase1b_image_audit.py`, run once on one laptop. Results are committed to `data/embeddings/`, so nobody else has to run it.

1. **Mask:**
   - white-background photos (97%) are cropped to the product;
   - photos of models, lifestyle shots and white-background photos with a person are cropped with an object detector (OWLv2), prompted with the CSV item type (e.g. "earrings").
2. **Embed:** each cropped product goes through CLIP and becomes **512 numbers** ("an image as a token"). Comparing products is then simple maths: all 90 million pairs take about 1 second.
3. **Check:**
   - item type (15 broad types) vs the CSV family;
   - colour vs `CLR_DES`;
   - the same photo used for another colour;
   - not a studio photo.
4. **Calibrate:** every flag type was inspected on contact sheets and the thresholds tuned. Flags fell from about 1,400 to about 700.

**Report:** [outputs/phase1b/image_audit_report.md](../outputs/phase1b/image_audit_report.md).

**Photos that look like another item type** (24 high-priority flags). Examples: a "bath suit" photo showing an outfit, a "top" that looks like a skirt, ballerinas photographed in a swimming pool:

![Type mismatches](../outputs/phase1b/vis_type_mismatch.jpg)

**Same photo for another colour** (54). Examples: the white and the blue sweatshirt, the blue and the silver skirt, the black and the taupe bag share one photo:

![Same photo for another colour](../outputs/phase1b/img_near_duplicate.jpg)

**Photo shows another colour** (132): [contact sheet](../outputs/phase1b/vis_colour_mismatch.jpg).

**Not a studio photo** (272): [contact sheet](../outputs/phase1b/img_not_packshot.jpg).

These are **suggestions for the reviewers**, not verdicts. After the review, the decisions will measure each flag's real precision.

## 4. Review workflow for the team (2026-10-08 → 09)

- **Setup:** double-click `1_setup_windows.bat` / `1_setup_mac.command`. See [README.md](../README.md).
- **Review app:** double-click `2_review_…`, type your number, and the browser opens:
  - each product as a card with its photo, data and flags in plain words;
  - buttons: Looks right / Wrong photo / Fix data / Not sure / Discard;
  - every click saved automatically;
  - **Submit** sends your file to GitHub.

| # | Member | Batch |
|---|---|---|
| 1 | André | `batch_01_of_05` |
| 2 | Pedro Correia | `batch_02_of_05` |
| 3 | Pedro Meireles | `batch_03_of_05` |
| 4 | Manuel | `batch_04_of_05` |
| 5 | Zé | `batch_05_of_05` |

- **Merge:** `3_merge_…` applies all decisions and logs every change. It produces the clean dataset `data/processed/items_clean.parquet`, the Sprint 1 deliverable. Products still unresolved stay out until the team decides.

## 5. Compute strategy (2026-10-09)

**Encode once, reuse everywhere.**
- Heavy models run on one machine only.
- The results (≈10 MB of embeddings) are committed to the repository.
- Everyone else just loads small vectors: no GPU, no `torch`.

Details: [roadmap.md § Compute strategy](roadmap.md#compute-strategy-encode-once-reuse-everywhere).

## 6. Tests and CI (Phase 0, 2026-10-10)

- **`CLAUDE.md`:** conventions for AI agents (data rules, layout, commands, code style).
- **42 tests** in `tests/`, about 2 s, no vision packages needed:
  - **automatic checks** on a toy dataset (6 SKUs, 4 colourways): size collapsing, image, colour, type and sales flags, priorities;
  - **review workflow:** the batch split, never overwriting a reviewer's decisions, `merge` (fixes, discard, drop image, pending), and the review app's save rules;
  - **data contracts** on the committed files: embeddings vs their sidecar, keys of `masks`/`image_audit`, and the 5 review batches (valid decisions, every colourway in exactly one batch).
- **CI:** GitHub Actions runs `ruff` and `pytest` on Python 3.10 and 3.14 for every push to `parfois`. A teammate's batch with an invalid decision now turns the CI red before the merge.

## 7. Phase 1 cleaning and preview table (2026-10-10)

The human review hasn't happened yet, so everything that doesn't depend on it is done now.

- **Script:** `src/phase1_clean.py` (`finalise_items`). It takes about 1 s. `merge` runs the same function after applying the review decisions, so the official table and the preview have the same columns.
- **Preview:** `data/processed/items_preview.parquet`, 10,555 colourways × 109 columns. Later phases read it through `load_items()`, which switches to `items_clean.parquet` once the review is merged. Report and schema: [outputs/phase1/clean_report.md](../outputs/phase1/clean_report.md).
- **What it cleans:**
  - 61 columns dropped (empty, constant, duplicates, size-level);
  - 10 date columns in 3 formats parsed, none lost;
  - case variants merged (`THEME`: 153 values, e.g. `GOLDEN BASICS` → `Golden Basics`);
  - placeholders set to missing **only where they mean "no value"** (`DIMENSION`, `DISTRIBUTION_BLOCK`, `CATEGORY_MATRIX`). `PRINT_TYPE = Others` stays: it means a print outside the list, which is different from `No Print`;
  - sales features: `log_sales_qty`, `realised_price` (empty for the 9 items with 0 units), percentile within the category.
- **Photos, before any human review:**
  - `img_trusted`: the photo shows this product. **9,433 of 9,458** photos. The 25 left out are flagged as another item type, another model or a damaged file;
  - `img_colour_trusted`: its colour can also be trusted. **9,151**. Shared, generic or colour-mismatched photos are left out;
  - after `merge`, "Looks right" or "Fix data" accepts a photo and "Wrong photo" removes it. Items without a trusted photo use the tabular and text blocks only.
- **Tests:** 62 in total (20 new), including Phase 1's "done when" rule on the real data: unique key, all 10,185 sales rows joined, 10,555 rows.

## 8. Repository reorganised (2026-10-10)

The project was hard to follow: two numberings (sprints and phases) mixed in file names, 9 documents in the root and the status repeated in 3 places.

- **One numbering:** code and outputs use phases. `sprint1_preprocess.py` → `phase1_checks.py`, `outputs/sprint1/` → `outputs/phase1/`. Sprints remain only as the calendar in the roadmap.
- **Shared settings** (paths, item key, issue codes, team) moved to `src/config.py`.
- **Docs in `docs/`:** `roadmap.md` (the only place with the status), `progress.md` (this log), `review_guide.md`, `data.md`, `problem.md`. The README is now a short entry point with a documentation map.
- **Archived** in `docs/archive/`: the teammate audit of 2026-10-09 and the prompt used to produce it. A note at its top gives the status of each finding: 6 fixed, 1 partly, 7 still open. The open ones are now in the roadmap's Phase 1 checklist, so they aren't lost in the archive.
- The review batches moved unchanged (no decisions had been recorded yet).
- Fixed on the way: the review app showed raw image-check codes next to their plain-language note if a batch was regenerated after Phase 1b.

## 9. Image counts verified and review fixes normalised (2026-10-10)

**Are the "missing" photos real?** Checked from scratch:
- The CSV is read correctly: 17,125 rows = 17,125 lines in the file, 10,555 colourways, every row has an image path. 9,496 image files, all with different names.
- **33 image paths in the CSV are wrong:** they drop the colour code (`212983_2`) while the file has it (`212983_HM_1.jpg`). Each file matches exactly one colourway and nothing else used it. `check` now links them and flags them `IMG_PATH_REPAIRED` (medium) so a reviewer confirms the photo. *(Corrected 2026-10-12: 18 files are still unused, all photos of the other sizes of colourways whose sizes list different photos, mostly re-coded items; each colourway uses the photo most of its sizes list, and all are flagged `SKU_ATTR_CONFLICT`.)*
- **1,064 colourways really have no photo.** 91% are from the latest season (`242`), so the photos were not delivered. Coverage: 9,458 → **9,491 colourways (89.9%)**.
- **Phase 1b on the 33 repaired photos:** new mode `phase1b_image_audit.py --add-missing` embeds only the photos missing from `data/embeddings/` and keeps the stored vectors; the comparisons across all photos (neighbours, near-duplicates, flags) are recomputed on the full set. It took 98 s on CPU. Before using it, the full-set comparisons were recomputed on the stored data: identical flags on all 9,457 photos. Result: 3 new colour flags (e.g. `193076_PU`: CSV says purple, the earrings are teal; checked by eye), no flag changed on existing photos. 9,490 photos are embedded; only `216726_DM` (damaged file) is not.
- Photo trust after this: 9,466 photos trusted, 9,181 also for colour.

**Review fixes use fixed lists** (`src/phase1_vocab.py`), so 5 reviewers can't create 5 spellings of the same colour:
- a fix can only use values that already exist in the data: 100 colours, 416 category › family › sub-family combinations, finishing and material per category, 95 materials;
- the review app shows lists instead of free text, and the family list depends on the category; the description is no longer editable;
- if the right value isn't listed, the reviewer uses **Not sure** with a note;
- the app and `merge` both validate, so a CSV edited by hand can't bring in new values; `merge` stores them in the data's spelling (`yellow` → `Yellow`);
- the allowed values are in `outputs/phase1/review_vocabulary.csv`.
- The review batches were regenerated (no decisions had been recorded): same products per member, now with the repaired photos and the Phase 1b flags written in.
- Tests: 79 (17 new).

## 10. Phase 2a started: categorical block (2026-10-10)

Phase 2a starts on the preview table, without waiting for the photo review: it uses the CSV attributes, not the photos. `phase1_clean.load_items()` will return the reviewed table after `merge`, so the same script is simply rerun then.

- **`src/phase2a_features.py`** → `data/processed/features_categorical.npy` (10,555 × 1,620, float32) and `feature_meta.json` (row order, columns, weight and position of each attribute).
- **15 attributes**, one-hot with a weight each: family and sub-family 3, colour 2, category matrix, finishing and theme 1.5, category, material, print, dimension and outfit 1, colour type, fashion type, segment and units 0.5. These are starting values; Phase 4 tunes them.
- **Family and sub-family are encoded under their category** (`Jewellery › Others` ≠ `Apparel › Others`): 416 sub-family paths, the same list the review uses.
- **A missing value never counts as a match** (finishing is 48% filled, material 13%).
- **Explainable by construction:** rows are normalised, so the dot product of two rows is their cosine similarity, and each attribute's slice gives its share of the score (`shared_attributes`). Phase 5 can say "same sub-family, same colour, same theme" with the exact contribution of each.
- Sanity check in `outputs/phase2a/features_report.md`: the neighbours of 3 random items share sub-family and mostly colour and theme (e.g. silver pendant `217529_SV` → 4 silver pendants of the same theme, score 0.83–0.87).
- `NUMBER_OF_UNITS` added to the Phase 1 contract columns (it is used here).
- Tests: 86 (7 new).

## 11. Review merged: Phase 1 done (2026-10-11)

All 5 batches were submitted. Before the merge, the decisions were made consistent with the team rule agreed on 2026-10-10:
- **If the photo shows this product in another colour → Fix data with the photo's colour** (more labelled examples for the model). Wrong photo only when the photo shows another article, a sketch, a swatch or a damaged file, or the colour of a sibling colourway of the same model.
- 48 doubtful decisions in batches 2, 3 and 5 were re-checked on the photos: 22 became colour fixes, 8 became Looks right (flag was a false alarm), the others stayed Wrong photo. Every changed row has a note (`[2026-10-11, Manuel: X -> Y, team rule]`); the original reviewer is kept.
- All 7 Not sure items and the 1 Discard were resolved, so nothing is left out of the clean table.
- No colour fix creates two colourways of one model with the same colour, except 4 models in batch 5 that the team kept on purpose.

**`merge` result** (`python src/phase1_checks.py merge`):

| | Items |
|---|---|
| Looks right | 10,290 |
| Fix data | 147 (141 colours, 10 family/sub-family, 1 finishing) |
| Wrong photo (image unlinked) | 118 (76 of them had no photo file anyway) |
| Discarded / unresolved | 0 / 0 |

- `data/processed/items_clean.parquet`: 10,555 colourways, 9,449 with a trusted photo. Every change is in `outputs/phase1/changes_log.csv` (270 rows, with old value, new value and reviewer); report in `outputs/phase1/clean_report.md`.
- `load_items()` now returns the merged table. Phase 2a was rebuilt on it: categorical block 10,555 × 1,619 (one colour value fewer after the fixes).

## 12. Image flag precision (Phase 1b, 2026-10-11)

`python src/phase1b_precision.py` → `outputs/phase1b/flag_precision.md`. A flag is confirmed when the reviewer dropped the photo or fixed what it points at.

| Flag | Flagged | Precision | Target |
|---|---|---|---|
| `VIS_TYPE_MISMATCH` (high) | 24 | 17% | 80% ❌ |
| `VIS_COLOUR_MISMATCH` (medium) | 135 | **70%** | 50% ✅ |
| `IMG_NEAR_DUPLICATE` (medium) | 54 | 33% | 50% (about half expected: the shared photo is right for one colourway) |
| `VIS_TARGET_NOT_FOUND` (low) | 46 | 50% | – |

- **Colour flag: useful.** On batch 4 (every photo looked at), a looser cut (0.02 instead of 0.005) keeps precision at 51% and raises recall from 36% to 48%.
- **Type flags: mostly false alarms.** The 20 unconfirmed ones were checked on a contact sheet: feathers, fringes, beads, macramé and lifestyle photos that CLIP misreads. The reviewers were right.
- **Recall:** any image flag caught 47% of the 58 problems found in batch 4; the rest were found by eye (mostly close colours, e.g. blue vs turquoise).
- Thresholds were **not** changed: new flags on items already reviewed would be left pending by a new `merge`. The report gives the settings to use on a future dataset.

## 13. Phase 2a: composition and numeric blocks (2026-10-11)

`python src/phase2a_features.py` now builds 3 blocks in about 1 s (report: `outputs/phase2a/features_report.md`):

| Block | Shape | What it encodes |
|---|---|---|
| categorical | 10,555 × 1,619 | 15 product attributes, weighted one-hot |
| composition | 10,555 × 95 | materials, weighted by rarity (IDF): sharing "Pearl" counts more than sharing "Polyester" |
| numeric | 10,555 × 79 | price percentile **within the category** (weight 2), launch month (1), season (0.5) |

- Price and launch date are soft bands (Gaussian bumps): 10 price percentile points apart → similarity ≈ 0.78; 1 month apart → ≈ 0.89, 3 months → ≈ 0.37.
- Same contract for every block: unit-length rows (dot product = cosine) and named attribute slices that add up to the score, so Phase 5 can say e.g. "shares Zinc and Pearl, same price band".
- A data contract test now builds every block on the real data and checks it; it caught one block stored as float64 instead of float32.
- Tests: 93 (7 new).

## 14. Phase 2a: text block, Phase 2a done (2026-10-11)

- **`src/phase2a_text.py`** builds one English sentence per item from the reviewed attributes, e.g. *"Cocktail (Rings, Jewellery). Colour: Pastel Multicolor. Finish: Worn Gold. Made of Stone, Zinc. Theme: Nile. Fashion, Woman."*, and embeds it with `sentence-transformers/all-MiniLM-L6-v2` (384 numbers, mean pooling, unit length).
- Run once on the vision machine (54 s on CPU) → `data/embeddings/text.npy` (10,555 × 384, float16, 8 MB) + `text.json` (model, commit, template, row order). Committed, so nobody else needs torch.
- `phase2a_features.py` loads it as the 4th block (`text`), in the same row order as the others; it is skipped with a message if the file is missing.
- Why it helps on top of the categorical block: it sees that close labels are close (e.g. "Printed Fake Nylon" and "Fake Nylon", or theme names written differently), where one-hot only matches identical labels.
- Tests: 96 (3 new), including a data contract for `text.npy` vs its sidecar and the Phase 1 table.

## 15. Phase 2b: colour palettes, Phase 2b done (2026-10-11)

- **`src/phase2b_palettes.py`**: for each photo kept by the review, k-means (k = 4, fixed seed) in CIELAB on the product pixels: inside the Phase 1b item box, without the near-white background. Clusters under 5% are dropped.
- 9,449 photos in 2.5 min on the light environment (no torch) → `data/embeddings/palettes.parquet` + `.json` (committed). Per item: colours as hex and LAB, their shares, the nearest of 24 colour names, and a `palette_vector` (shares per name, unit length) whose dot product is a palette similarity that can be explained ("both mostly black and white").
- Checked by eye on `outputs/phase2b/palettes_sample.jpg`: the swatches match the photos. 9,197 palettes come from packshots (`reliable`); the others (model / lifestyle photos) keep some background.
- **Limitation:** metals look like light neutrals (a gold item's main cluster is its highlights: L 86, b 15, like beige or ecru), so a colour name can't tell gold from beige. The metal comes from `FINISHING` in the CSV; palette similarity is unaffected.
- Tests: 101 (5 new), including a data contract for `palettes.parquet`.

**Every input of Phase 3 is now ready:** 4 tabular/text blocks (`data/processed/features_*.npy`), CLIP image embeddings and colour palettes (`data/embeddings/`), and the photo trust columns of `items_clean.parquet`.

## 16. Missing attributes filled where reliable (Phase 2a, 2026-10-11)

Only items **with a trusted photo** are filled; items without a photo are left out. Fields are filled only in the categories where they exist: `FINISHING` is empty for apparel and footwear because it doesn't apply to them, and `MATERIAL` only exists for apparel.

| Gap (items with a photo) | How | Measured precision | Filled |
|---|---|---|---|
| `MATERIAL`, apparel (778) | 1. another colourway of the same model has it · 2. the composition is a single material · 3. vote of the 15 most similar items (vote share ≥ 0.8) | 99.7% · 98% · 92–95% | **581** (75%) |
| `FINISH_FAMILY`, jewellery (417) | vote of the 15 most similar items, accepted only if the photo agrees (gold must look yellowish, silver must not) | 97% (gold 97%, silver 91%) | **266** (64%) |
| Detailed `FINISHING`, jewellery | – | ≈ 88% on only 30% | not filled |
| `FINISHING`, hand bags | – | ≈ 77% | not filled |

- Precision is measured by 5-fold cross-validation **grouped by model**, so a colourway never learns from its own siblings. A value is only filled if its own precision is ≥ 90% (e.g. Linen 81% and Polyamide 71% are not filled by the neighbours).
- Checked by eye on a sample sheet: two silver items first filled as gold led to the photo check (gold is 79% of the jewellery, so the vote leaned to gold).
- `FINISH_FAMILY` is also a new categorical attribute (weight 1): two items with "Golden" and "Worn Gold" now get partial credit.
- Nothing in the CSV or in `items_clean.parquet` changes: the values live in `data/processed/attributes_inferred.parquet` with method and confidence, and `phase2a_features.py` uses them where the CSV is empty.
- Tests: 108 (7 new), including a data contract (never replaces a value, only items with a photo, only known values).

## 17. Every photo checked (Phase 1, 2026-10-12)

Before, only flagged photos had been looked at in batches 1, 2, 3 and 5 (6,912 photos never seen). Batch 4 had shown that about 3% of unflagged photos have a problem, so all 6,912 were checked one by one on contact sheets of 12.

- **Rule used:** fix only clear colour-family changes (e.g. CSV "Blue", photo turquoise; CSV "Pink", photo fuchsia; CSV "Midnight Blue", photo light pink stone). Close shades (ecru / beige / white, taupe / brown / camel, grey shades) were left as they are, so the fixes stay reliable.
- **Result:** 314 colour fixes and 4 wrong photos (3 show a sibling colourway's colour, 1 a different article); the rest are ok. Every changed row has a note `[2026-10-12, Manuel: full photo check, team rule]`.
- Most common errors: "Blue" for turquoise or teal items (jewellery and wallets), "Pink" for fuchsia, "Green" for lime or teal, and pastel names for strong colours.
- **Merge now:** 9,972 ok, **461 fixed**, 122 wrong photos, 0 unresolved; 588 changes in `changes_log.csv` (455 colours). 9,445 photos trusted.
- **What it says about the automatic flags:** they caught only 28% of the 506 photo problems found by eye (`outputs/phase1b/flag_precision.md`). A review of flagged items alone would have missed most of them.
- Phase 2a features, inferred attributes and Phase 2b palettes were rebuilt on the new table. Tests: 108.

## 18. Similarity engine (Phase 3, 2026-10-12)

`src/phase3_engine.py build` returns the 10 most similar colourways for each of the 10,555 colourways in about 7 seconds (light environment, no torch).

- **Score:** weighted sum of the cosine similarity of six blocks: image 0.30, categorical 0.30, text, composition, palette and numeric 0.10 each (`src/phase3_weights.json`, starting values for Phase 4). A pair is scored on the blocks both items have, with the weights rescaled, so an item without a trusted photo is matched on its tabular blocks and is not penalised.
- **Candidates:** same category, never the item itself and never another colour of the same model (`PROD_REF` or `PROD_REF_EQUIV`).
- **Coverage:** every colourway has at least 4 neighbours (0 with fewer).
- **First sanity check (top 4):** same family 91–99%, same sub-family 76–91%, same colour 42–66%, median price ratio 1.00–1.08 (`outputs/phase3/engine_report.md`). Items without a photo get similar agreement (sub-family 88%).
- **By eye:** `outputs/phase3/examples.jpg` (12 random queries): matches share the sub-family, shape and finish; colour varies more in apparel.
- **Finding:** 72 queries have a best match scoring ≥ 0.99, mostly the same design under another model code. They are kept; Phase 4 should check whether buyers see them as duplicates.
- Each neighbour row stores `sim_<block>` and `contrib_<block>` (the contributions add up to the score), ready for the Phase 5 explanations.
- Tests: 116 (7 unit tests on toy blocks + a data contract on `neighbours.parquet`).

## 19. Evaluation and weight tuning (Phase 4, 2026-10-12)

`src/phase4_eval.py run` writes `outputs/phase4/eval_report.md` in about 3 minutes. Weights are tuned on half of the models (fixed split by `PROD_REF`) and every number is measured on the other half (5,292 queries).

| Top 4, test half | Random | Starting weights | Tuned weights |
|---|---|---|---|
| `hit@4` (a known similar item: same sub-family, theme and colour, other model) | 1% | 87% | **91%** |
| Same family / sub-family | 25% / 8% | 96% / 85% | **98% / 90%** |
| Same colour / finishing / material | 9% / 37% / 18% | 57% / 80% / 75% | **58% / 84% / 74%** |
| Among the query's 20 nearest photos (`visual@4`) | 1% | 37% | **39%** |
| Sales correlation (Spearman) | 0.00 | 0.62 | 0.62 |

- **Tuned weights:** image 0.36, categorical 0.36, text 0.08, composition 0.08, palette 0.04, numeric 0.08 (were 0.30 / 0.30 / 0.10 / 0.10 / 0.10 / 0.10). Applied to `src/phase3_weights.json` and `neighbours.parquet` rebuilt.
- **Why the tuning is constrained:** every proxy is circular for one block. A first search on the mean of the three families put 88% of the weight on the image: visual agreement rose, but `hit@4` fell to 76% and material agreement from 75% to 45%. So a change is now kept only if no single metric drops by more than 1 point.
- **Ablations:** the categorical block matters most (without it `hit@4` falls to 60%); text alone is as good as categorical alone on `hit@4`; palette alone is close to random; numeric drives the sales correlation (without it 0.55).
- **Image weight sweep (0 to 0.8):** more image weight buys visual agreement (28% → 76%) and costs catalogue agreement (81% → 71%). The proxies cannot pick the point, so the human review does.
- **Human review sheet:** `outputs/phase4/human_review_<n>.csv` + `.jpg`, one per member (Excel in Portuguese saves with `;`, which the reader accepts): 50 queries (10 per category, 2 without a photo), candidates from image weights 0, 0.3 and 0.6 pooled and shown in key order without scores (blind), 319 pairs. Once the team fills `verdict`, `run --apply` picks the image weight with the best human precision.
- Tests: 123 (6 new unit tests + a contract on the review sheet).

## 20. Open audit findings closed (Phase 1, 2026-10-12)

The six findings left open from the 2026-10-09 audit are done (`src/phase1_checks.py`). Every new flag is `info`, so no reviewed item went back to pending: the merge still gives 9,972 ok, 461 fixed, 122 wrong photos, 0 pending, and features, neighbours and the Phase 4 numbers are unchanged.

| Finding | What was done | Result on the data |
|---|---|---|
| GAP-001 prices and sales | `check_numerical`: price, outlet, landed cost, negative or inconsistent sales, realised vs list price | No invalid price; 175 without landed cost, 10 cost above price, 1 amount without units, 21 odd realised prices |
| INC-002 stale flags | `merge` re-runs the text checks on fixed rows and drops photo flags of unlinked photos (`issues_checked` keeps the originals) | 2 flags cleared, 3 confirmed, 249 photo flags removed |
| WEA-001 image size and mode | width, height and mode stored per item; `IMG_SMALL`, `IMG_ODD_SHAPE`, `IMG_NOT_RGB` | All RGB; 45 small, 15 odd proportions |
| WEA-002 singular forms | "Dress" no longer becomes "dres" | Same 2 `TYPE_CONFLICT_DESC` items |
| RED-001 `CLR_NOT_IN_DESC` | Severity medium → low, based on the review | 8 of 25 flags confirmed (32%) |
| OPP-001 sales profile | `outputs/phase1/sales_profile.csv` + report section | 122 low-sales outliers (none high); realised price ≈ 0.85 × list price |

- After a colour fix, the description and the colour code still name the old colour in 898 cases (e.g. CSV "Blue", photo turquoise). These flags are counted, not added: the CSV text is what the reviewer corrected.
- Tests: 137 (13 new).

## 21. Interactive feedback: the weights learn from the team's sorting (Phase 4, 2026-10-12)

The CSV sheet asked for one verdict per pair. The new app (`src/phase4_feedback.py`) asks reviewers to **sort**: each
candidate goes to *Similar* (most similar first) or *Not similar*. Every answer becomes pairwise preferences, and the
six block weights are fitted to them (Bradley-Terry, the logistic model of RankNet and of RLHF reward models).

| Design choice | Why |
|---|---|
| Pools built once and committed (`feedback_pool.json`): top 5 at image weights 0, 0.3, 0.6 + 2 random same-category items | The learned weights never change what a reviewer sees, so every setting is judged on the same items; the random items let the fit see weights outside the three settings |
| Candidates shuffled, no scores, no source | No position or engine bias |
| 20 of the 100 queries never fitted | Honest before/after check; `fit --apply` refuses weights that do worse there |
| 10 queries sorted by all 5 members | Measures reviewer agreement, the ceiling for any weights |
| Prior on the log-weights (sd 0.5) towards the current weights | A few odd answers cannot swing the weights |
| Refit after every answer (~0.4 s) | The panel shows what the engine is learning while you sort |

- **Pool:** 100 queries (16 with a photo + 4 without, per category), 847 candidates (7–11 per query), 28 queries per member.
- **Check on a simulated reviewer** who scores with image weight 0.6 plus noise: the fit moves image 0.36 → 0.61 and
  categorical 0.36 → 0.20, and the held-out preferences ordered right rise from 82% to 85%.
- Launchers `4_feedback_windows.bat` / `4_feedback_mac.command`; the submit button commits only the member's file
  (`git_submit`, shared with the Phase 1 review app).
- Tests: 147 (10 new: 9 unit tests, incl. gradient vs finite differences and engine-score equivalence, + a contract on the pool and the answers).

## 22. Colour names measured on the photos (Phase 2a, 2026-10-12)

`CLR_DES` has 99 names and many are shades of one colour. As a one-hot, "Navy" vs "Midnight Blue" scores 0, like "Navy" vs "Red". `src/phase2a_colour.py` gives each name a reference colour measured on the photos instead of a hand-made colour grouping.

- **Reference colour:** median CIELAB of the dominant palette colour (Phase 2b) of the trusted packshots with that name. A refinement that picked, in each photo, the palette colour nearest the reference drifted pale colours to white (Aquamarine, Pastel Blue) and was dropped.
- **Similarity of two names:** exp(−d² / 2τ²), d = LAB distance with lightness counted half (CMC 2:1), τ = 8. Examples: Navy–Midnight Blue 0.87, Ecru–White 0.87, Blue–Light Blue 0.33, Navy–Blue 0.10, Red–Burgundy 0.
- **Coverage:** 64 of 99 names measured (63% of the items). Multicolour names (one colour means nothing for a print), metallic names (gold measures as light beige on a photo, and gold vs silver is a key distinction) and names with under 5 photos only match themselves.
- **Automatic colour families** (k-means on the reference colours) were tried for wording like "both blues" and dropped: Grey and Lilac fell into the blues, and the groups changed with the number of families.

**Engine test** (current weights, test half; τ chosen on the tune half):

| Categorical colour encoding | hit@4 | Sub-family | Colour (exact name) | Material | Close colour (sim ≥ 0.5) | Objective |
|---|---|---|---|---|---|---|
| Before: exact name, weight 2 | 90.9% | 89.9% | 58.4% | 74.5% | 60.9% | 0.601 |
| Exact name 1 + measured tone 1 (τ 6 / 8 / 10) | 90.5–90.6% | 89.9–90.0% | 57.2–57.8% | 74.4–74.8% | 61.3–61.5% | 0.600–0.601 |
| Exact name 1 + tone 2 | 92.0% | 88.0% | 63.9% | 73.3% | 68.4% | 0.606 |
| Exact name 1 + tone 3 | 93.2% | 86.0% | 69.1% | 71.5% | 73.9% | 0.607 |

- At the same colour weight the measured tone changes nothing (objective ±0.001): the photo-based blocks (image, palette) and the text already carry most of this signal.
- More colour weight raises colour agreement but costs sub-family (up to −3.9 points) and material: a trade-off the Phase 4 rule refuses (no metric may drop more than 1 point). Whether buyers want colour first or type first is a question for the team's sorting, not for the proxies.
- **Decision:** the engine is unchanged (`features_categorical.npy` is identical). The reference table is kept for Phase 5, which can say "close colours: Navy ≈ Midnight Blue" instead of "different colour".
- **For the team (optional, 2 min):** check `outputs/phase2a/colour_reference.jpg`. Two names look off: "Blue" is a greyish blue (denim and neutral parts in the photos) and "Coffee" is grey (5 photos only).
- Tests: 155 (8 new: 7 unit tests on hand-made palettes, incl. the exact factorisation of the similarity, + a contract on `colour_reference.parquet`).

## 22. Weights learned from the team's sorting (Phase 4, 2026-10-21)

All members sorted their queries in the feedback app: 138 of 140 (André 26/28), 3,330 pairwise preferences.

| Block | Before | Learned | Applied (25% blend) |
|---|---|---|---|
| image | 0.36 | 0.62 | **0.42** |
| categorical | 0.36 | 0.25 | **0.33** |
| text | 0.08 | 0.07 | 0.08 |
| composition | 0.08 | 0.03 | 0.07 |
| palette | 0.04 | 0.02 | 0.03 |
| numeric | 0.08 | 0.02 | 0.07 |

- **The team wants the photo to count more.** On the 20 held-out queries the learned weights order 75.3% of the preferences right (72.8% before).
- **Reviewer agreement:** 70% of the pairs on the 10 shared queries, the ceiling for any weights.
- **Cost of the full learned weights** (proxy metrics, test half): visual agreement +17 points, but material agreement −9 (74.5% → 65.6%), sales correlation 0.62 → 0.59, price ratio 1.00 → 1.08.
- **Applied: a 25% blend.** Held-out 73.8%, material 73.9%, sales 0.62, price ratio 1.00, `hit@4` 91.5%, visual agreement 41%. The gap between 73.8% and 75.3% is small for 699 held-out pairs; the material drop is not. Table of the blends in `outputs/phase4/feedback_report.md`.
- `neighbours.parquet` rebuilt; `outputs/phase3/` and `outputs/phase4/eval_report.md` regenerated.


## 23. Sprint 2 notebook and a fix in the attribute imputation (2026-10-21)

- `notebooks/phase2_features.ipynb`: the Sprint 2 features step by step, self-contained (hard rule 8). The categorical, composition and numeric blocks it builds are identical to `features_*.npy` (max difference 0.0); the filled attributes are the same 846 values; the palettes match the committed file (200 photos checked, differences only from rounding). The text and CLIP embeddings are described and loaded.
- Fix in `src/phase2a_impute.py`: when guessing the metal (`FINISH_FAMILY`), the neighbour vote also used `FINISH_FAMILY` itself, so the cross-validation partly saw the answer. Now every guessed attribute is left out. The filled values are unchanged; the measured precision drops slightly (FINISH_FAMILY at a vote share ≥ 0.8: 97% → 96.5%, `outputs/phase2a/impute_report.md`). New regression test.

## 24. Modelling added to the Sprint 2 notebook (2026-10-21)

Sprint 2 covers features **and** modelling, so the sprint table in the roadmap now puts Phases 3 and 4 in Sprint 2 (explanations and delivery become Sprint 3).

- `notebooks/phase2_features.ipynb`, new sections 11–16, self-contained (hard rule 8):
  - the model (weighted mean of the block similarities, like-for-like candidates, other colours of the same model excluded);
  - the evaluation signals on the test half of the models, against a random baseline;
  - weights step 1, the guarded coordinate search, rerun in the notebook;
  - weights step 2, the Bradley–Terry fit on the team's sorting, loaded from `outputs/phase4/feedback_*.jsonl`, with the held-out check, reviewer agreement and the 25% blend;
  - the final model, the ablations and the example of the brief with its block contributions.
- **Same results as the pipeline:**
  - step 1 gives the same weights as the Phase 4 tuning (0.36 / 0.36 / 0.08 / 0.08 / 0.04 / 0.08);
  - the learned weights are the same (image 0.618), and so is the held-out accuracy (72.8% → 75.3%);
  - the final top 10 is identical to `neighbours.parquet` (105,550 of 105,550 rows);
  - the test-half metrics match `outputs/phase4/eval_report.md` (`hit@4` 92%, material 74%, `sales_corr` 0.618).
- Running time: about 5.5 minutes.

## 25. Sprint 3: baseline frozen, improvements measured against it (branch `s3/andre`, 2026-10-26 →)

Rule for this sprint: the Sprint 2 engine is the **baseline**. Every change is measured against it on the same data, and
kept only if it wins on signals it was not tuned on and no guarded metric drops by more than 1 point (the Phase 4
rule). Results that did not win are recorded too.

**Where a change is measured** (the strongest signal first):
1. **The team's judgements**: 138 sorted answers (the latest of each member) on **99 distinct queries** (9 sorted by all five members, 1 by four), 4,029 pairwise preferences (`outputs/phase4/feedback_*.jsonl`). Two ways:
   current weights on all answers, and 5-fold cross-validation over queries with the weights refitted (same
   Bradley–Terry fit as Phase 4). Confidence intervals by paired bootstrap over queries (a query's pairs are not
   independent).
2. **Labels the change never saw**: for an image encoder, the sub-family, material and finish of each photo's 4 nearest
   photos (the image block alone, test half of the models).
3. **The proxy metrics** of `phase4_eval.py` (test half, guarded) and **the brief's own example** (where the engine ranks
   the 4 products Parfois gave for `167718_BU`; one example, so only a sanity check).

**Baseline reproduced.** Rebuilt from the committed files on another machine (check → merge → Phase 2a → Phase 3 →
Phase 4): the "current weights" row of `eval_metrics.csv` is identical (`hit@4` 91.5%, sub-family 90.2%, material 73.9%,
`visual@4` 41.4%, sales 0.618); the ablation rows differ only at the 5th decimal (float summation order). A ceiling to keep
in mind: the baseline orders 72.8% of the team's preferences right (cross-validated), and two team members agree with each
other on 70% (Phase 4). Re-weighting the same blocks has little room left; gains must come from better representations.

### 25.1 Fusion: per-category standardisation of the block similarities — rejected

The image similarities are compressed (random pair ≈ 0.75–0.84, best 4 matches ≈ 0.91–0.95, per category) while the
categorical one spans 0–1, so the weights do not mean the same thing for every block. Tested: each block's similarity
standardised by its mean and SD over random pairs of the same category, before the weighted sum.

| Fusion | Team preferences, current weights | Team preferences, 5-fold CV refit |
|---|---|---|
| Raw cosine (baseline) | 70.5% | **72.8%** |
| Standardised per category | 71.6% (+1.1, CI −1.4 to +3.6) | 70.3% (−2.5, CI −4.9 to −0.2) |

On the 20% held-out queries alone the standardised version looked much better (78.8% vs 73.8%), but on all answers with
cross-validation it is worse: the held-out set (≈20 queries) was too small. The Bradley–Terry fit already compensates the
compressed image scale (it learned image 0.62). Literature agrees: a tuned weighted combination beats rank fusion when
validation data exists (Bruch et al., *An analysis of fusion functions for hybrid retrieval*). **Not adopted.**

### 25.2 Colour and theme weights inside the categorical block — rejected

The brief's four answers for `167718_BU` have other colours and another theme, and the engine ranks them 452, 44, 66 and
659 of 4,638. Hypothesis: colour (weight 2) and theme (1.5) weigh too much.

| Variant | Team (current w) | Team (CV) | Brief ranks | Sub-family | Material | Sales corr. |
|---|---|---|---|---|---|---|
| Baseline | **70.5%** | 72.8% | 452 / 44 / 66 / 659 | 90.2% | 73.9% | **0.618** |
| Colour ×0.5 | 69.7% | 71.7% | 472 / 53 / 79 / 683 | 92.0% | 74.5% | 0.616 |
| Colour 0 | 68.7% | 70.8% | 494 / 60 / 91 / 699 | 93.2% | 76.0% | 0.613 |
| Theme 0 | 70.1% | 73.7% | 484 / 56 / 83 / 690 | 92.3% | 76.4% | 0.578 |
| Colour 0 + theme 0 | 67.4% | 70.7% | 525 / 80 / 113 / 729 | 94.9% | 81.7% | 0.569 |

The team values colour (removing it loses up to 3 points), the brief's answers get *worse*, and theme 0 costs 4 points of
sales correlation for a team gain inside the noise (±2.5 points). **Not adopted.** (The colours of the photo palette do not
separate the team's similar / not similar candidates at all: Cohen's d −0.06, §25.4.)

### 25.3 Image encoder: FashionSigLIP adopted, DINOv2 rejected

Same item crops (Phase 1b), same trust mask, three challengers to CLIP ViT-B/32 (the FashionSigLIP row includes the 6
re-crops of §25.7; the effect of 6 photos out of 9,445 is negligible). Computed once on a laptop CPU (CLIP
full photos, FashionCLIP and DINOv2 in 40 min; FashionSigLIP in about 1 h).

| | CLIP B/32 (baseline) | FashionCLIP | DINOv2-small | **FashionSigLIP** |
|---|---|---|---|---|
| Team, current weights | 70.5% | 72.0% | 70.8% | **72.1%** |
| Team, 5-fold CV refit | 72.8% | 73.6% | 69.9% | **74.0%** |
| Photo alone: 4 nearest, same sub-family | 57.7% | 65.0% | 63.9% | **68.1%** |
| Photo alone: same material | 30.3% | 35.7% | 36.0% | **39.1%** |
| Photo alone: same finish | 58.5% | 60.2% | 54.4% | **61.9%** |
| Photo alone: same colour | 43.3% | 42.9% | 25.4% | **43.4%** |
| Brief's answers, photo alone | 5 / 10 / 39 / 214 | 3 / 7 / 49 / 225 | 6 / 50 / 19 / 53 | **2 / 12 / 33 / 68** |
| Engine `hit@4` / sales corr. (guards) | 91.5% / 0.618 | 89.8% / 0.611 | 80.1% / 0.571 | 91.0% / 0.614 |

- **DINOv2** loses colour (25%) and the team's agreement: rejected, as the fashion product-matching literature found
  (CLIP-type encoders beat DINOv2 on fashion images).
- **FashionSigLIP** (Marqo, trained on fashion products) wins on every label the encoders never saw (+10.4 points of
  sub-family, +8.8 of material from the photo alone, on the 4,716 test-half products with a trusted photo) and keeps every
  guard within 1 point. On the team's preferences the gain is +1.5 points (fixed weights, CI −0.4 to +3.4, positive in 94%
  of the resamples) and +1.2 (CV, CI −0.8 to +3.0, 87%): **not significant at 95%** with 99 sorted queries.
  Caveats on this evidence: the encoder was picked as the best of three on the same team metric; the photo-only labels
  (category, material, colour) are close to what a fashion encoder was trained on, so that test favours it; the team's
  candidate pools were built with the CLIP engine and the fixed weights were fitted on CLIP similarities, which, if
  anything, favours the baseline. **Adopted** for the improved version, with
  that caveat; the CLIP baseline stays for comparison. New: `src/phase2b_image_encoders.py embed`, and the engine reads
  the embedding named in `phase3_weights.json` (`image_embedding`, CLIP by default).
- **Reproducible:** the FashionCLIP and DINOv2 embeddings are committed (`data/embeddings/image_*_nopersoncrops.*`), and
  `python src/phase4_compare.py encoders image_clip image_fashionclip_nopersoncrops image_dinov2_nopersoncrops
  image_fashionsiglip` regenerates `outputs/phase2b/image_encoders.csv` (same numbers as the first, exploratory run).

### 25.3b Diversity: at most 1 or 2 colours of one model in the top 5 — rejected

The brief's example has three colours of one model (`210347`) in its top 5, which a buyer counts almost as one suggestion.
Tested by re-ranking the top 50:

| | Distinct models in the top 5 | Same material | Same sub-family | Team: share of the judged top-5 called similar |
|---|---|---|---|---|
| Baseline | 4.59 | 73.9% | 90.2% | 88.4% |
| At most 2 colours per model | 4.66 | 73.1% | 90.1% | 88.3% |
| At most 1 colour per model | 5.00 | 71.5% | 89.4% | 88.3% |

The problem is rare (4.6 distinct models out of 5 already). One colour per model costs 2.4 points of material (beyond the
1-point guard); two colours change almost nothing. **Not adopted.**

### 25.4 Explanations for every match (Phase 5)

`src/phase5_explain.py build` → `data/processed/explanations.parquet`, `outputs/phase5/` (report, the brief's format for
every product in `similar_products.csv`, the brief's example in `brief_example.json`). Numbers on the Sprint 3 engine.

- **Method:** the score is a weighted mean of block similarities, and each block is a dot product of unit rows split into
  named slices, so the explanation is an **exact decomposition** (no surrogate, no sampling, no LLM): share of each block,
  shared attributes with their points, shared materials (rare ones weigh more), price/launch parts, differences (price,
  colour with the measured closeness of the two names, finish, sub-family) and the shared dominant photo colours.
- **Completeness:** the parts add up to the score (largest gap 6.8e-07 on 2,000 pairs: float32 rounding).
- **Sufficiency** (best match of 300 random queries, scores rounded like the engine, ties counted against the match):
  keeping only the two blocks the explanation names first, the match stays in the top 5 for 86% of the queries; two random
  blocks 53%, the two named last 10%. **But a control — the two blocks with the largest global weights — gives the same
  86%:** the two blocks named first *are* the two heaviest (photo + attributes) in 292 of 300 matches, and in the 8 others
  the difference (75% vs 62%) rests on too few cases to mean anything. So this test shows that the heavy blocks recover
  the match, not that the per-match ranking of blocks adds information. What is specific to each match is inside the
  blocks — which attributes and materials are shared, and the differences — and that part is exact by construction
  (completeness), not something a test can make more or less true.
- **Deletion** (for an additive score this mostly confirms the arithmetic): removing the block named first unseats the
  match in 50% of the queries (mean rank 5.7), a random block 32%, the block named last 19%.
- **Agreement with the team** (Phase 4 feedback, only the candidates the engine proposed: 769 called similar, 134 not;
  Cohen's d with the pooled within-group SD): photo 0.96, description 0.49, catalogue attributes 0.44, materials 0.42,
  price/launch 0.28, photo colours −0.06. A first version also counted the 2 random candidates per query (easy to reject on
  any block) and divided by the total SD; it put the catalogue attributes first, an artefact of the easy cases.
- **Cost:** 52,775 matches (10,555 products × top 5) explained in about 85 s (1.6 ms each) on a laptop.

### 25.5 Price and sales estimates (Phase 5b, new)

`src/phase5b_forecast.py run` → `data/processed/forecast.parquet`, `outputs/phase5b/forecast_report.md`,
`leakage_checks.csv`, `estimates.csv` (one readable row per product); `select` → `feature_selection.md`;
`src/phase5b_buyer.py run` → `buyer_report.md`. The first version (test on 242, a "time on sale" feature, price without
the landed cost: sales off by a factor of about 3, price 3.41 €) is **superseded by `4f694ad`**, because 242 sales are
cut by the extraction.

- **Data checks:** no list price missing, at 0 € or negative (3.99 € to 250 €, 40 price points). Missing: sales for 370
  colourways (no sales row) and 0 units for 9; landed cost for 175. Products without sales are unknown, not zero: out of
  training and evaluation, but they get an estimate.
- **Sales are whole-life units.** Sales are accumulated until the extraction (about January 2025). Seasons 232 and 241
  had finished selling (98% of 241 went to the outlet); 242 had not (median units by launch month fall from 3,341 in
  May 2024 to 143 in October). So the units are learned and scored on whole lives only: **train 232 → the new models of
  241**, split at random by model into two halves, one for the choices (validation), one reported once (test). 242
  only checks the ranking among products of a category launched in the same month (equally cut), with a model trained
  on 232 + 241. No "time on sale" feature: the days to the extraction describe the extraction, not the product.
- **Price = landed cost × markup.** The model learns log(price / landed cost); the cost carries the rise of prices
  between seasons, which trees cannot extrapolate. Products without a cost use the model without it. Validation
  232 → 241, test 232 + 241 → 242 (list prices are not cut). Snapped to the nearest list price (`price_point_estimate`).
- **Features, by a rule declared before the choice:** a block leaves only if it helps in neither check (the 241 test
  and the 242 ranking). Sales: attributes + photo components (the photo is neutral on 241, +0.005 to +0.052 same-month
  ranking on 242); price: attributes only. The description components, the materials and the launch month help in
  neither. The similar products enter as a feature only if the 95% interval of their validation gain excludes 0: not for
  sales (−0.026 to +0.026), yes for price (+0.06 to +0.16 €).
- **Left out:** `PUR_BET` (purchase depth) and `TYPOLOGY` (store tiers), the buyer's own expectation; they are the
  benchmark instead.

| Sales, test: the other half of the new 241 models (1,436 products) | Same launch month | Error factor | Within ×2 | Top-20% |
|---|---|---|---|---|
| Category median | 0.00 | ×2.09 | 67% | 22% |
| Family + price band median | 0.29 | ×2.05 | 70% | 29% |
| Similar past products (engine, 10 nearest) | 0.33 | ×2.03 | 66% | 32% |
| **Gradient boosting (chosen)** | **0.50** | **×1.69** | **76%** | **37%** |
| Reference: + buyer decisions (not used) | 0.61 | ×1.63 | 79% | 51% |

Ranking check on 242 (3,393 products, cut sales, same launch month): similar products 0.28, gradient boosting 0.41.

| Price, test: new models of 242 (3,728 products) | Mean error | Exact list price | Right or next | Bias |
|---|---|---|---|---|
| Category median | 5.76 € | 23% | 45% | −6.1% |
| Sub-family median | 4.16 € | 33% | 60% | −6.5% |
| Similar past products (engine, no price block) | 3.77 € | 31% | 68% | −5.6% |
| Gradient boosting without the cost (the first version) | 3.39 € | 34% | 72% | −4.0% |
| **Gradient boosting, cost × markup (chosen)** | **2.68 €** | **38%** | **78%** | −4.5% |

**Gradient boosting against the similar past products** (paired, same products, 95% CI by model): same-month ranking **+0.10 to +0.22**, units error saved **+0.14 to +0.23** log units (sales, 241 test half); price error saved **+0.90 to +1.33 €** (242). Both intervals exclude 0, so the evidence that it errs less is strong; the similar products stay as the explanation a buyer can check, listed with every estimate.

- **No leakage found:** with the training target shuffled at random (20 runs) the ranking falls to zero (0.004 ± 0.040
  same month).
- **The planning classes carry the sales:** without fashion type and collection type the error grows from 0.52 to 0.73
  log units (within ×2: 76% → 62%). Their shares shift between seasons ("Commercial Online" 1.9% / 2.5% / 7.1% in
  232 / 241 / 242, collection "Online" 0% / 0.6% / 3.3%); when Parfois assigns them is a question for Parfois.
- **Biases** (`forecast_report.md` § Biases): sales −6%, mostly jewellery (−15%) while apparel is over-estimated (+7%):
  not one season growth, and with one whole-season transition no correction can be tested, so none is applied. Price
  −4.5%: markups rise between seasons (jewellery 8.5 → 12.1 from 232 to 241); a season index and more weight on 241
  were tried on half of the new 242 models, none gains, none applied.
- **Against the buyer** (`buyer_report.md`, test half): the model ranks like the buyer (same month −0.03 to +0.12) and
  errs less in units (+0.07 to +0.18 log saved); model + buyer beats both (0.61); when they disagree the buyer is right in
  55–63% of the pairs; on the 26% of pairs the buyer ties, the model orders 65% right.
- **What drives the estimates** (mean |SHAP|, log units): sales — fashion type 0.49, collection type 0.20, the photo 0.18,
  planning category 0.16, supplier 0.13; price (the markup) — the landed cost 0.65, the similar products 0.11, weight
  0.05.

**Delivered estimates are never in-sample:** products with a known target get a cross-validated estimate (5 folds
grouped by model, PROD_REF and PROD_REF_EQUIV joined); 242 products (cut sales) and products never sold get a forecast.
Each carries an **80% range** from the test residuals (sales ×0.50 to ×2.33 of the estimate, price ×0.89 to ×1.23). **Stock for a sales period is not delivered
yet:** the data has whole-life units, no dates and no stores; `ITEM_LIFE_CYCLE_WEEK` (4–12 weeks, median 8) is not the
selling time (241 stayed a median of 48 weeks at full price, Spearman −0.15 with it; the outlet dates are 6 season-wide
dates). The definition is a question to the professor.

### 25.6 Delivery (Phase 6)

`src/phase6_deliver.py query KEY` (similar products, why, sales and price estimates with their 80% range), `report` →
`outputs/phase6/report.html` (self-contained: the brief's example, 2 products per category and one without a photo) and
`brief` → `outputs/phase6/brief_example.md` (§25.10). The final report for the repository-only delivery:
`docs/final_report.md`.

### 25.7 Photos of a person wearing the item: 6 re-cropped after a check by eye

Phase 1b crops a photo on a white background to its non-white pixels; when a model wears the item, that crop is the whole
person (the swimsuit `211451_BM` was embedded as white trousers). Two stages (`src/phase2b_image_encoders.py persons`, run
once, results in `data/embeddings/person_crops.parquet` + `.json`):

1. **CLIP on the whole photo** ("a person", "a face", "a hand wearing jewellery"… against "a product on a white
   background"…): a cheap filter. A first version that scored the stored *crop* embeddings was poor (about 7 real people in
   its top 48: the crop hides the context). On the whole photo the top is mostly people; a random sample of 96 photos below
   0.3 had 1 person, so the filter keeps everything ≥ 0.15: **2,454 of 9,445 trusted photos**.
2. **OWLv2** on those (CPU, 16.6 h, far more than estimated: about 20–40 CPU-seconds per photo): a "person / face / hand /
   legs" score and the box of the expected item. Calibrated on contact sheets: ≥ 0.16 is almost always a person (about 41 of
   the top 48), 0.10–0.16 about two thirds (60 of 89), 0.05–0.10 rarely (4 of 48).

**Finding: the problem is much smaller than feared.** Of the 137 photos with a person (score ≥ 0.10), **88 had already been
cropped to the item by the detector in Phase 1b** (they were not packshots): the new box is identical. Only models on a
white background were cropped by threshold. Automatic rule — person ≥ 0.10, item ≥ 0.10, item box < 70% of the current
crop — selects 13 photos; checked before/after by eye, **6 are better** (swimsuit, 4 dresses and trousers worn by a model)
and **7 worse** (packshots of t-shirts with printed faces, zooms into prints, jewellery sets cut to one piece): precision
46%, too low to apply blindly. **Only the 6 confirmed by eye are re-cropped** (the machine suggests, a person decides, as in
the Phase 1 review); the file keeps both the automatic decision and the confirmed one.

Effect: 6 products of 9,445, so no aggregate metric can move; for those products the photo now shows the item (the
swimsuit's 5 nearest photos: trousers and jumpsuits before, dresses, tops and a kimono after; cosine between its old and new
vector 0.74).

### 25.8 The improved engine against the baseline

`src/phase4_compare.py run` → `outputs/phase4/compare_report.md`. Same weights; only the image embedding differs (CLIP →
FashionSigLIP with the 6 re-crops, `image_embedding` in `src/phase3_weights.json`).

| | Baseline (Sprint 2) | Sprint 3 |
|---|---|---|
| Team preferences, fixed weights | 70.5% | **72.1%** (+1.5, CI −0.4 to +3.4, positive in 94%) |
| Team preferences, 5-fold CV refit | 72.8% | **74.0%** (+1.2, CI −0.8 to +3.0, positive in 87%) |
| Photo alone: same sub-family / material / finish | 57.7% / 30.3% / 58.5% | **68.1% / 39.1% / 61.9%** |
| `hit@4` / same sub-family / same material | 91.5% / 90.2% / 73.9% | 91.0% / 89.7% / 74.5% |
| Sales correlation | 0.618 | 0.614 |
| Brief's four answers (engine rank) | 452 / 44 / 66 / 659 | 269 / 30 / 56 / 575 |

All guards within 1 point. The gain on the team's judgements is positive but within the noise of 99 sorted queries; the
gain on what the photos show (labels never seen by the encoder) is large. The brief's own answers move up but stay far:
the catalogue attributes (other colour, theme, finish) keep them down, and the team's preferences say those attributes
matter (§25.2).

Downstream, everything is rebuilt on the Sprint 3 engine: neighbours, explanations (§25.4) and estimates (§25.5; the
image components of the estimates now come from FashionSigLIP). The checked before/after sheet of the 13 automatic
re-crops: `outputs/phase2b/person_recrops_checked.jpg`.

### 25.9 Independent review and the fixes (2026-10-27)

After the main tasks a separate agent reviewed the branch (read-only: code, outputs, docs, rules; `pytest` and `ruff`
passed). It found no error in the engine or the explanations' arithmetic (no negative contribution in 105,550 neighbour
rows, completeness ≤ 1e-6, every product ≥ 4 matches, no colour of the query's own model in any top 5), and these problems,
all fixed:

| Finding | Fix |
|---|---|
| **Delivered estimates were in-sample** (refitted on every product and shown next to its real sales: error 0.33 instead of 1.13) | Cross-validated estimates grouped by model; each carries its kind and the new-season typical error (§25.5); CLI and HTML report say so |
| **Person re-crops not reproducible** (thresholds in the code ≠ the ones used; rerunning would overwrite the human decision) | Real thresholds in the code; the human decision in `data/embeddings/person_crops_reviewed.json`, never regenerated; a test checks the code reproduces the 13 suggestions and 6 re-crops |
| **Evidence without committed code** (leakage table, encoder comparison) | `phase5b_forecast.py` writes `leakage_checks.csv`; `phase4_compare.py encoders` + committed candidate embeddings |
| Forecast choices made on the test season; final (revised) store dates | Choices on 232 → 241, test reported once; planned store dates |
| Planning classes said to be stable across seasons (false) | Measured and reported as a shift risk (§25.5) |
| "The photo adds little" (the favourable metric) | Both metrics reported: −0.16 within category, −0.03 same month |
| Explanation checks weak (deletion is near-tautological; team check pooled easy random candidates) | Sufficiency test added (and, after the second review, a largest-weights control that shows its limit, §25.4); Cohen's d on the engine's candidates (the ordering changed) |
| Reference rows compared with the wrong model | Built from the chosen model |
| Stale numbers from the CLIP run; "138 queries" | Regenerated from the reports; 99 distinct queries (138 answers) |
| Notebook explained the baseline | Sections 2–4 on the Sprint 3 engine, with the baseline beside it |
| Second review (same day): sufficiency confounded by the block weights; the notebook's Cohen's d on CLIP similarities; the model choice written in prose, not computed; stale notebook cells; answer counts; roadmap without Phase 5b and contracts; folds grouped by PROD_REF only; one shuffled run | Largest-weights control and the rows where the explanation differs from it; Cohen's d on FashionSigLIP; the choice is a rule in the code (validation interval); notebook cells fixed; 138 answers / 99 queries; roadmap Phase 5b + output contracts + contract tests; folds and bootstrap by joined model codes; 20 shuffled runs; provenance notes in hand-written sidecars |
| Minor: `open_clip_torch` missing from `requirements-vision.txt`, no library versions in the sidecar, excluded products listed as analogues, inconsistent weighting of the rank metrics, ties counted in the match's favour | Fixed (regression tests for the analogue keys and the re-crop rule) |

### 25.10 Team review (Pedro, 2026-10-27): what was checked and done (2026-10-28)

Every claim of the review was checked against the PDF, the data and the branch. Most held; three did not, and they are
corrected here before anything goes to the professor:

| Review claim | Checked |
|---|---|
| The engine's top 5 for `167718_BU` are "of the same theme" | **No:** 1 of 5 is STARRY NIGHT, 2 are DEGRADÊ (the brief's theme). What they share with the query is the type (beaded dangle), the golden finish and the price |
| The brief's 4 answers all have finish *Other* | **No:** 3 *Other*, 1 Golden (`177781_LB`, a stud) |
| Stock per week = whole-life units / `WEEKS_TARGET` | **Not supported:** `WEEKS_TARGET` = `ITEM_LIFE_CYCLE_WEEK` (4–12), but 241 stayed a median of 48 weeks at full price, unrelated to it; the rate would be about 6× too high |
| The PDF has 6 slides | 7 (title, 5 content slides, closing); the quotes are right |
| Sales and price are required (D1) | Not in the PDF: from the professor in class, to confirm in writing |

**The brief's example** (`python src/phase6_deliver.py brief` → `outputs/phase6/brief_example.md` + `.jpg`):

| Matches the query | Brief's 4 answers | Engine's top 5 |
|---|---|---|
| Same sub-family (Dangle) | 3 | 5 |
| Same theme (STARRY NIGHT) | 0 | 1 |
| Same finish (Golden) | 1 | 5 |
| Distinct models | 3 | 3 |

Both lists are beaded earrings; the engine ranks the brief's answers 30, 56, 269 and 575 of 4,638 candidates. A
difference of criterion, not an error; one example cannot evaluate the engine. `117781_LB` does not exist; `177781_LB`
(its photo is the one on the slide) is used.

**Is `sales_corr` biased by the cut 242 sales?** (`python src/phase4_compare.py salescheck` →
`outputs/phase4/sales_corr_check.md`; report only, no weight changed)

| | (a) as now: within category | (b) within category × season | (c) whole lives only |
|---|---|---|---|
| Baseline (Sprint 2, CLIP) | 0.618 | 0.561 | 0.552 |
| Current (FashionSigLIP) | 0.614 | 0.561 | 0.541 |
| Current, numeric block off (diagnostic) | 0.532 | 0.506 | 0.508 |
| Control: random, same category | −0.001 | −0.009 | −0.016 |
| Control: random, same category and launch month | 0.363 | 0.236 | 0.144 |

Yes, by about 0.07: with all seasons pooled, the launch month alone is worth 0.363, because products launched together
share the cut. Most of the engine's correlation survives on whole lives (0.541), and the two engines stay within 0.011 of
each other in every column, so the Phase 4 guard decision does not depend on it. A future tuning should use (c).

**Also done:** the documents agree with the code (this section, the roadmap, the notebook's section 4); a readable file
per product (`outputs/phase5b/estimates.csv`, with an 80% range and a contract test against `forecast.parquet`); the
model against the similar past products as a paired interval; `docs/final_report.md` and a "Start here" in the README.
Waiting: the professor's answers (the brief's criterion, whether sales and price are required, the sales period).

### 25.11 External audit: two-stage retrieval and the other proposals tested — rejected (2026-10-28)

An external audit (`phase3_audit_and_improvements.md`, not in the repo) proposed a cap of one colour per model, lower
colour / finish / theme weights, and a two-stage retrieval (the N nearest photos of the same sub-family first, then
re-ranked). The first two had been rejected already (§25.2, §25.3b). Its result table came from a script that is not in
the repo, and two of its numbers are impossible (the cap leaves the brief's ranks unchanged; a sub-family first stage
brings the stud `177781_LB` to 137). All variants re-measured on the usual signals with `python src/phase4_twostage.py`
→ `outputs/phase4/twostage_check.md` + `.csv` (the baseline row reproduces §25.8 exactly):

| Variant | Team pairs right (Δ, 95% CI) | hit@4 | Sub-family | Colour | Finish | Material | Sales corr. | Brief ranks |
|---|---|---|---|---|---|---|---|---|
| Baseline (current engine) | 72.1% | 91.0% | 89.7% | 58.4% | 83.0% | 74.5% | 0.614 | 269 / 30 / 56 / 575 |
| Cap 1 colour per model | −1.5 (−2.5 to −0.5) | 91.1% | 88.8% | 60.4% | 82.6% | 71.5% | 0.613 | 269 / 25 / 180 / 575 |
| Soft colour/finish/theme + cap | −2.1 (−3.5 to −0.5) | 80.5% | 93.4% | 45.5% | 74.8% | 76.2% | 0.584 | 293 / 28 / 187 / 722 |
| Two-stage 50, re-rank without the photo | −2.5 (−4.9 to −0.1) | 88.0% | 96.0% | 53.9% | 79.4% | 70.2% | 0.583 | 48 / 20 / 19 / 575 |
| Two-stage 50, re-rank with the full score | +0.7 (−1.4 to +2.9) | 88.1% | 96.0% | 53.4% | 78.6% | 67.8% | 0.570 | 43 / 15 / 21 / 575 |
| Two-stage 20, re-rank without the photo | −0.3 (−2.4 to +1.9) | 82.6% | 96.0% | 51.7% | 76.2% | 64.0% | 0.550 | 19 / 10 / 66 / 575 |
| Two-stage 50 + soft + cap (the audit's best) | −3.2 (−5.6 to −1.1) | 79.3% | 96.1% | 43.8% | 72.5% | 72.1% | 0.569 | 184 / 177 / 15 / 722 |

The best variant (two-stage 50 with the full score) brings three of the brief's answers forward and raises same
sub-family to 96%, but the team's preferences do not move (inside the noise) and five guards fall by more than 1 point
(colour −5.0, finish −4.4, material −6.8, sales −0.044). The team's pools come from the current engine, so the variants'
top 5 are less covered by the team's judgements (72–83% against 88%). **Not adopted.** The audit's sales proposals were
done already (242 out of the units model) or do not apply (`PUR_BET` is the buyer's decision, the benchmark; classes
instead of units hide the uncertainty, the top-20% precision is reported).
