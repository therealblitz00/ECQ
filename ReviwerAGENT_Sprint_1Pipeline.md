# AGENT INSTRUCTIONS — Sprint 1 Pipeline Review

## 1. Role

Act as a **Senior Data Analyst / Senior Data Scientist with strong Data Engineering and Data Quality expertise**.

Your task is to perform a **technical and data-science review of the existing Sprint 1 preprocessing pipeline** of this academic project.

You are acting as an **auditor/reviewer**, not as an implementation agent.

The project already contains work developed by other team members. Your responsibility is to understand what has already been implemented, evaluate it against the project requirements and previously defined preprocessing roadmap, identify gaps and inconsistencies, and produce a detailed roadmap for improvement.

---

# 2. CRITICAL RESTRICTION — DO NOT MODIFY THE PROJECT

**Do NOT implement any changes.**

During this task you must NOT:

* modify existing source code;
* create or delete project files;
* refactor code;
* rewrite functions;
* change datasets;
* generate corrected datasets;
* automatically fix detected problems;
* change configuration files;
* install packages;
* commit changes;
* create pull requests;
* silently apply improvements.

You may inspect, analyse and explain code.

Your output must be a **review and improvement roadmap**, not an implementation.

If you identify something that should be changed, describe:

1. what is currently happening;
2. why it may be a problem;
3. what should conceptually be changed;
4. why the change matters;
5. priority;
6. dependencies or prerequisites.

Do **not** implement the change.

---

# 3. Main Objective

Review the existing implementation under:

`src/sprint1_preprocess/`

and determine how well it covers the preprocessing pipeline previously defined for the project.

The review must answer:

### A. What is already implemented?

Understand exactly what the current code does.

### B. What is missing?

Identify requirements or pipeline stages that are not implemented or are insufficiently covered.

### C. What is implemented unnecessarily?

Identify functionality that does not appear to be required for the current Sprint 1 objectives, or that belongs to a later stage.

### D. What is inconsistent?

Identify inconsistencies between:

* project documentation;
* expected pipeline;
* individual scripts;
* functions;
* input/output datasets;
* intermediate results;
* naming conventions;
* assumptions;
* data transformations.

### E. Is the pipeline coherent?

Determine whether the individual scripts form a logically consistent end-to-end preprocessing pipeline.

### F. What should be improved?

Identify technical, methodological, data-quality and maintainability improvements.

### G. What should be done next?

Produce a prioritized **roadmap of recommended actions**, without implementing them.

---

# 4. Project Context — Read These First

Before analysing the code, read and understand the following project documents:

* `SPRINT1_GUIDE.md`
* `ROADMAP.md`
* `data_description.md`
* `problem_description.md`
* `README.md`

These documents define the project context and must be treated as the primary reference for the review.

Do not assume that the existing code is correct simply because it exists.

Likewise, do not assume that the previously defined roadmap is correct without checking it against the project documentation.

---

# 5. Reconstruct the Expected Sprint 1 Pipeline

Before judging the implementation, reconstruct what the Sprint 1 preprocessing pipeline is expected to accomplish.

Use the project documentation and the previously established preprocessing roadmap.

The previously defined pipeline includes, where applicable:

1. Data inventory
2. Dataset profiling
3. Identification of unit of observation / granularity
4. Identification and validation of keys
5. Identification of product IDs / SKU relationships
6. Missing-value analysis
7. Duplicate analysis
8. Key-level duplicate/conflict analysis
9. Categorical consistency analysis
10. Numerical validation and anomaly detection
11. Cross-dataset validation
12. Orphan-key detection
13. Product master / product-level integration
14. Image inventory
15. Image ↔ product mapping
16. Image quality analysis
17. Exact image duplicate detection
18. Perceptual image duplicate detection
19. Raw-data integrity
20. Reproducibility and traceability
21. Output validation
22. Issue logging / data-quality reporting

Do not assume that every item above must necessarily be implemented in Sprint 1.

Determine from the project documentation:

* which items are explicitly required;
* which are implied;
* which belong to later stages;
* which are optional recommendations.

Clearly distinguish these categories.

---

# 6. Inspect the Existing Implementation

Inspect the complete contents of:

`src/sprint1_preprocess/`

Do not review files individually in isolation.

Understand the complete pipeline.

For each relevant script/module/function determine:

* purpose;
* inputs;
* outputs;
* dependencies;
* transformations;
* validation performed;
* assumptions;
* error handling;
* relationship with other scripts;
* whether its output is consumed by another stage;
* whether the functionality is actually executed;
* whether it is duplicated elsewhere.

Also inspect, when relevant:

* configuration files;
* requirements/dependencies;
* README documentation;
* scripts that call the preprocessing modules;
* generated output structures;
* naming conventions.

The goal is to reconstruct **how the current code actually works**, not how it appears to be intended to work.

---

# 7. Do Not Judge Only by Code Presence

A pipeline stage should NOT be considered implemented merely because a function or script with a relevant name exists.

Check whether the functionality is:

* actually executed;
* correctly connected to the pipeline;
* applied to the correct dataset;
* applied at the correct granularity;
* producing meaningful outputs;
* validated;
* documented;
* consumed by subsequent stages.

For example:

A function called `check_duplicates()` does not automatically mean that duplicate analysis is adequately implemented.

Evaluate what it actually checks.

---

# 8. Required Comparison Matrix

Create a comprehensive comparison between the expected pipeline and the current implementation.

Use a structure equivalent to:

| Pipeline Stage | Requirement Source | Expected Behaviour | Current Implementation | Status | Evidence | Gap / Issue | Priority |
| -------------- | ------------------ | ------------------ | ---------------------- | ------ | -------- | ----------- | -------- |

Use statuses such as:

* **COMPLETE**
* **PARTIALLY IMPLEMENTED**
* **MISSING**
* **IMPLEMENTED BUT INCONSISTENT**
* **IMPLEMENTED BUT WEAK**
* **NOT REQUIRED FOR SPRINT 1**
* **UNCLEAR / NEEDS CONFIRMATION**

The "Evidence" column must refer to the actual code, file, function or documentation that supports the assessment.

Do not make unsupported claims.

---

# 9. Review Categories

Evaluate the existing pipeline across the following dimensions.

## 9.1 Data Inventory

Check whether the code correctly identifies:

* available datasets;
* files;
* columns;
* data types;
* dimensions;
* image assets;
* expected relationships.

Evaluate whether the inventory is reproducible and sufficiently informative.

---

## 9.2 Dataset Profiling

Check:

* row counts;
* column counts;
* data types;
* unique values;
* missing values;
* basic statistics;
* categorical distributions;
* suspicious values.

Evaluate whether profiling is performed consistently across datasets.

---

## 9.3 Granularity and Keys

Determine whether the pipeline explicitly establishes:

* unit of observation;
* product-level granularity;
* image-level granularity;
* SKU/product identifiers;
* composite keys where applicable.

Check whether uniqueness assumptions are actually validated.

---

## 9.4 Missing Values

Review:

* detection;
* classification;
* reporting;
* treatment;
* distinction between legitimate missing values and errors.

Check whether missing-value handling is performed too early or without sufficient evidence.

---

## 9.5 Duplicates

Evaluate:

* exact row duplicates;
* duplicate product IDs;
* duplicate SKUs;
* conflicting records sharing a key;
* duplicate images;
* duplicate relationships.

Distinguish:

**duplicate record**

from

**multiple valid observations referring to the same product**.

---

## 9.6 Categorical Consistency

Check whether categorical variables are evaluated for:

* spelling variants;
* capitalization;
* whitespace;
* accents;
* synonyms;
* inconsistent encodings;
* unexpected categories.

Determine whether the current code silently standardizes values or merely detects inconsistencies.

---

## 9.7 Numerical Validation

Review validation of:

* ranges;
* impossible values;
* negative values where inappropriate;
* outliers;
* suspicious distributions;
* units;
* precision;
* data-type conversions.

Distinguish statistical anomalies from confirmed data errors.

---

## 9.8 Cross-Dataset Validation

Evaluate whether relationships between datasets are explicitly checked.

Examples:

* products ↔ images;
* products ↔ attributes;
* product IDs across CSVs;
* missing references;
* orphan records;
* multiple images per product;
* products without images;
* images without valid product references.

---

## 9.9 Product Master / Integration

Assess whether the current pipeline creates or should create a coherent product-level representation.

Review:

* key selection;
* joins;
* relationship cardinality;
* conflicting attributes;
* duplicated information;
* preservation of source provenance.

---

## 9.10 Image Pipeline

If image data is part of Sprint 1, review:

### Inventory

* image count;
* file formats;
* dimensions;
* file sizes;
* missing files.

### Mapping

* image → product;
* product → images;
* missing mappings;
* orphan images.

### Quality

* corrupted images;
* unreadable files;
* unusual dimensions;
* blank images;
* very small images;
* inconsistent formats.

### Duplicates

Determine whether the implementation checks:

* exact duplicates;
* perceptual duplicates.

Do not confuse these with image similarity for the final recommendation system.

---

## 9.11 Raw Data Integrity

Check whether the implementation preserves the principle:

> RAW data should remain unchanged.

Determine whether transformations are:

* traceable;
* reproducible;
* separated from raw data;
* documented.

---

## 9.12 Reproducibility

Review:

* deterministic execution;
* paths;
* configuration;
* hard-coded values;
* dependency assumptions;
* execution order;
* random seeds where relevant;
* output locations.

---

## 9.13 Output Validation

Check whether generated outputs are:

* complete;
* internally consistent;
* usable by downstream stages;
* documented;
* reproducible.

---

## 9.14 Error Handling

Evaluate whether the code handles:

* missing files;
* malformed files;
* invalid values;
* unexpected schema changes;
* corrupted images;
* empty datasets;
* failed joins.

Also check whether errors are:

* explicitly raised;
* logged;
* silently ignored.

---

## 9.15 Code Quality

Review:

* modularity;
* naming;
* readability;
* duplication;
* function responsibilities;
* comments/docstrings;
* hard-coded values;
* unnecessary complexity;
* dead code;
* unused imports;
* duplicated logic.

This is a review, not a refactoring exercise.

---

# 10. Evaluate Pipeline Consistency

This is a particularly important part of the review.

Do not only ask:

> "Does each script work?"

Also ask:

> "Does the complete pipeline make sense?"

Look for problems such as:

* one script assuming a different schema from another;
* inconsistent product identifiers;
* transformations applied in incompatible orders;
* outputs not consumed downstream;
* duplicated processing;
* contradictory cleaning rules;
* inconsistent naming;
* inconsistent missing-value treatment;
* different definitions of the same concept;
* analysis performed at different granularities;
* image/product mappings becoming inconsistent;
* information being lost between stages.

Identify dependencies between issues.

---

# 11. Distinguish Four Types of Conclusions

Every important conclusion should be classified as one of:

### DOCUMENTED REQUIREMENT

Explicitly required by project documentation.

### IMPLEMENTATION OBSERVATION

Directly observed in the existing code.

### DATA-SCIENCE RECOMMENDATION

A methodological recommendation based on good practice.

### INFERENCE

A conclusion inferred from the available evidence.

Do not present recommendations as project requirements.

---

# 12. Issue Classification

Classify each identified issue as one or more of:

* **GAP** — required functionality is missing.
* **INCONSISTENCY** — different parts of the project behave differently or contradict each other.
* **BUG / POTENTIAL BUG** — implementation may produce incorrect results.
* **WEAKNESS** — functionality exists but is insufficient.
* **REDUNDANCY** — duplicated or unnecessary functionality.
* **RISK** — potential future problem.
* **OPPORTUNITY** — improvement that could significantly improve quality.
* **DOCUMENTATION GAP** — behaviour exists but is insufficiently documented.

---

# 13. Priority

Assign a priority to each finding.

### P0 — Critical

Could invalidate the preprocessing results, break the pipeline, corrupt data, or make downstream analysis unreliable.

### P1 — High

Important issue that should be resolved before considering Sprint 1 preprocessing complete.

### P2 — Medium

Meaningful improvement, but does not prevent the pipeline from functioning.

### P3 — Low

Nice-to-have, maintainability or future improvement.

---

# 14. Identify Things That Are "Too Much"

The review must also identify functionality that may be beyond the current Sprint 1 scope.

For each such item determine whether it is:

* useful but premature;
* unnecessary;
* duplicated;
* better postponed to a later stage;
* potentially introducing unnecessary complexity.

Do not recommend removing functionality simply because it is not in the roadmap.

Explain why it may be premature or unnecessary.

---

# 15. Identify Missing vs Future Work

Clearly separate:

### Missing from Sprint 1

Functionality that should already exist according to the project requirements.

### Recommended improvement to Sprint 1

Functionality that would strengthen the current pipeline.

### Future work

Functionality that should probably be handled later, for example during:

* feature engineering;
* image embedding;
* similarity modelling;
* recommendation generation;
* model evaluation.

This distinction is essential.

---

# 16. Final Deliverable — ROADMAP

The main output must be a **detailed review roadmap**.

Structure it approximately as follows:

# Sprint 1 Preprocessing — Review & Improvement Roadmap

## 1. Executive Summary

Briefly explain:

* overall quality of the current implementation;
* how complete the pipeline is;
* most important strengths;
* most important weaknesses;
* whether the current pipeline appears ready for the next project stage.

---

## 2. Current Pipeline Overview

Describe how the existing code currently works.

Include:

* scripts;
* execution flow;
* main transformations;
* inputs;
* outputs;
* dependencies.

---

## 3. Pipeline Coverage

Provide the expected-vs-current comparison matrix.

---

## 4. Critical Findings

List all P0/P1 findings.

For each:

* issue;
* evidence;
* impact;
* recommended direction;
* dependencies.

---

## 5. Detailed Findings

Group findings by:

1. Data inventory
2. Profiling
3. Granularity and keys
4. Missing values
5. Duplicates
6. Categorical consistency
7. Numerical validation
8. Cross-dataset validation
9. Product integration
10. Image pipeline
11. Raw data integrity
12. Reproducibility
13. Outputs
14. Error handling
15. Code quality

---

## 6. Existing Strengths

Explicitly identify what the team has already done well.

Do not make the review purely negative.

---

## 7. Missing Components

Provide a consolidated list of genuinely missing components.

---

## 8. Inconsistencies

Provide a consolidated list of inconsistencies across:

* documentation;
* scripts;
* datasets;
* transformations;
* outputs.

---

## 9. Excess / Premature Components

Identify functionality that appears to be:

* beyond Sprint 1;
* redundant;
* unnecessarily complex;
* better postponed.

---

## 10. Opportunities for Improvement

Identify methodological and engineering improvements that would strengthen the pipeline.

---

# 17. Recommended Roadmap Table

The final roadmap should include a table such as:

| ID | Priority | Category | Area | Finding | Evidence | Impact | Recommended Action | Dependencies | Sprint |
| -- | -------- | -------- | ---- | ------- | -------- | ------ | ------------------ | ------------ | ------ |

Example:

| ID      | Priority | Category      | Area                     | Finding                                                    | Evidence                      | Impact                                           | Recommended Action                        | Dependencies           | Sprint   |
| ------- | -------- | ------------- | ------------------------ | ---------------------------------------------------------- | ----------------------------- | ------------------------------------------------ | ----------------------------------------- | ---------------------- | -------- |
| GAP-001 | P1       | GAP           | Cross-dataset validation | Product IDs are not systematically checked across datasets | `module_x.py`, function `...` | Orphan records may remain undetected             | Add explicit cross-dataset key validation | Product key definition | Sprint 1 |
| INC-001 | P1       | INCONSISTENCY | Missing values           | Two modules use different missing-value rules              | `module_a.py`, `module_b.py`  | Different datasets may be treated inconsistently | Define a common missing-value policy      | Data dictionary        | Sprint 1 |
| OPP-001 | P2       | OPPORTUNITY   | Reporting                | Quality checks exist but results are not consolidated      | `...`                         | Difficult to audit results                       | Create consolidated quality report        | Output specification   | Sprint 1 |

These are examples of structure only. Do not invent findings.

---

# 18. Dependency-Aware Roadmap

Do not simply create a flat list of tasks.

Identify dependencies.

For example:

```text
Define product key
        ↓
Validate key uniqueness
        ↓
Validate cross-dataset relationships
        ↓
Build product-level master
        ↓
Validate product-image mapping
        ↓
Image quality analysis
        ↓
Similarity features
```

If a later task depends on an unresolved earlier issue, explicitly state that dependency.

---

# 19. Recommended Order of Action

At the end, provide a proposed order for addressing the findings:

### Phase 1 — Critical correctness

P0/P1 issues that can affect data validity.

### Phase 2 — Pipeline consistency

Resolve inconsistent assumptions and interfaces.

### Phase 3 — Data-quality completeness

Complete missing profiling, validation and cross-dataset checks.

### Phase 4 — Reporting and reproducibility

Improve traceability and quality reporting.

### Phase 5 — Maintainability

Address code-quality and engineering improvements.

### Phase 6 — Future stages

Identify what should deliberately be postponed to later parts of the project.

---

# 20. Important Analytical Principles

Follow these principles throughout the review.

### Do not assume an anomaly is an error.

An unusual value is not automatically wrong.

### Do not silently "correct" data.

The objective is to identify and document issues.

### Do not confuse detection with treatment.

A pipeline may detect an issue without needing to automatically correct it.

### Do not confuse image duplicate detection with image similarity.

Duplicate detection belongs to data-quality/preprocessing.

Similarity modelling belongs to a later stage.

### Do not confuse recommendations with requirements.

Clearly distinguish project requirements from good-practice recommendations.

### Do not penalize the existing implementation for not implementing future stages.

The review must respect the project's intended scope.

### Do not reward complexity for its own sake.

A simpler pipeline that correctly satisfies the requirements is preferable to an unnecessarily complex pipeline.

---

# 21. Review Philosophy

The goal is **not** to produce the most sophisticated pipeline possible.

The goal is to determine:

> **Is the current Sprint 1 preprocessing pipeline correct, complete, coherent, reproducible and appropriate for the project's objectives?**

The review should therefore prioritize:

1. correctness;
2. data integrity;
3. completeness;
4. consistency;
5. reproducibility;
6. traceability;
7. maintainability;
8. only then additional sophistication.

---

# 22. Final Rule

The agent must finish with a **review roadmap only**.

It must NOT:

* modify code;
* generate replacement code;
* automatically fix issues;
* refactor the repository;
* create a new preprocessing pipeline;
* overwrite existing files.

If implementation would be useful, describe **what should be implemented and why**, but leave implementation for a separate subsequent task.

The purpose of this agent is:

> **REVIEW → DIAGNOSE → DOCUMENT → PRIORITIZE → ROADMAP**

not:

> **REVIEW → MODIFY**.
