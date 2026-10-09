"""Sprint 1: data preprocessing and alignment checks (Parfois similarity project).

Every team member runs this same script. The checks and the batch split are
deterministic, so everyone gets identical results without exchanging files.

Usage (from the project root):
    python src/sprint1_preprocess.py check                       # full automatic checks + report
    python src/sprint1_preprocess.py batch --members 5 --id 2    # my review batch (CSV + HTML sheet)
    python src/sprint1_preprocess.py batch --members 5           # all batches
    python src/sprint1_preprocess.py merge                       # merge reviewed batches, apply fixes

Unit of analysis: one colourway (PROD_CLR_EQUIV = model + colour). Sizes are collapsed,
because the description, colour, category and image are the same for every size.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"
IMG_DIR = ROOT / "data" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "sprint1"
BATCH_DIR = OUT_DIR / "batches"

KEY = "PROD_CLR_EQUIV"
# Sprint 1 review: batch number -> team member (used with --members 5).
TEAM = {1: "André", 2: "Pedro Correia", 3: "Pedro Meireles", 4: "Manuel", 5: "Zé"}
# Codes must stay strings (leading zeros, no float conversion).
STR_DTYPES = {c: str for c in ["PROD_REF", "PROD_REF_EQUIV", "BAR_COD", "SEA_COD", "CAT_COD",
                               "GFA_COD", "GFS_COD", "SUP_COD", "TARIFF_COD"]}

# Issue code -> severity. "high" = sources contradict each other, must be reviewed.
ISSUES = {
    "IMG_PATH_INVALID": "high",        # PROG_IMAGE does not follow /<season>/<cat>/<file>
    "IMG_FILE_MISSING": "info",        # no image file -> item will use tabular data only
    "IMG_UNREADABLE": "high",          # file exists but is corrupt
    "IMG_REF_MISMATCH": "high",        # image file name belongs to another model
    "IMG_CAT_MISMATCH": "medium",      # category folder in path != CAT_COD (often a reclassified item)
    "IMG_COLOUR_MISMATCH": "high",     # colour code in file name != CLR_COD
    "IMG_GENERIC": "medium",           # file name has no colour -> colour cannot be confirmed
    "IMG_SHARED": "medium",            # same image used by another colour or model
    "TAB_KEY_INCONSISTENT": "high",    # PROD_CLR != PROD_REF + CLR_COD
    "CLR_CODE_NAME_INCONSISTENT": "high",  # CLR_COD letters map to a different CLR_DES elsewhere
    "CLR_CONFLICT_DESC": "high",       # PROD_DES names a different colour than CLR_DES
    "CLR_NOT_IN_DESC": "medium",       # PROD_DES does not mention the colour at all
    "TYPE_CONFLICT_DESC": "high",      # PROD_DES item type disagrees with GFA_DES_EN
    "SALES_MISSING": "info",           # no row in df_sales
    "SALES_ZERO_QTY": "low",           # SALES_QTY == 0
    "SKU_ATTR_CONFLICT": "medium",     # sizes of the same colourway disagree on an attribute (see sku_conflicts)
}
# Attributes that must be identical for every size of a colourway (they feed the similarity model).
SKU_INVARIANT_COLS = ["CLR_COD", "CLR_DES", "CLR_TYPE", "CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN",
                      "CATEGORY_MATRIX", "COMPOSITION", "MATERIAL", "FINISHING", "PRINT_TYPE", "OUTFIT",
                      "DIMENSION", "NUMBER_OF_UNITS", "THEME", "FASHIONTYPE", "PROD_SEG",
                      "PRICE_BASE_W_VAT", "PROG_IMAGE"]
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}

REVIEW_STATUSES = {"ok", "fix", "drop_image", "discard", "team_review"}
REVIEW_COLS = ["review_status", "fixes", "reviewer", "notes"]
BATCH_COLS = [KEY, "PROD_REF", "PROD_DES_BASE", "CLR_COD", "CLR_DES", "CAT_DES_EN", "GFA_DES_EN",
              "GFS_DES_EN", "COMPOSITION", "FINISHING", "MATERIAL", "PRICE_BASE_W_VAT", "PROG_IMAGE",
              "img_file", "priority", "issues", "sku_conflicts"]


# --------------------------------------------------------------------------- loading
def _normalise_text(s: pd.Series) -> pd.Series:
    s = s.str.replace("\xa0", " ", regex=False).str.strip()
    return s.str.replace(r"\s+", " ", regex=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs, normalise whitespace and put COMPOSITION in a canonical form."""
    prod = pd.read_csv(CSV_DIR / "df_product.csv", dtype=STR_DTYPES, encoding="utf-8", low_memory=False)
    sales = pd.read_csv(CSV_DIR / "df_sales.csv", encoding="utf-8")

    # Materials in COMPOSITION are separated by double spaces: turn them into "; " first.
    prod["COMPOSITION"] = prod["COMPOSITION"].str.strip().str.replace(r"\s{2,}", "; ", regex=True)
    for col in prod.columns:
        if pd.api.types.is_string_dtype(prod[col]):
            prod[col] = _normalise_text(prod[col])
    # Sizes list the same materials in different orders ("Zinc; Enamel" vs "Enamel; Zinc"): sort them.
    prod["COMPOSITION"] = prod["COMPOSITION"].map(
        lambda s: "; ".join(sorted(set(s.split("; ")))) if isinstance(s, str) else s)
    sales[KEY] = sales[KEY].str.strip()
    return prod, sales


# --------------------------------------------------------------------------- dataset-level checks
def missing_report(prod: pd.DataFrame) -> pd.DataFrame:
    n = len(prod)
    rep = pd.DataFrame({
        "n_missing": prod.isna().sum(),
        "pct_missing": (prod.isna().mean() * 100).round(2),
        "n_unique": prod.nunique(),
    })
    rep["status"] = "ok"
    rep.loc[rep["pct_missing"] >= 50, "status"] = "sparse (>=50% missing)"
    rep.loc[rep["n_unique"] <= 1, "status"] = "constant"
    rep.loc[rep["n_missing"] >= n - n * 0.001, "status"] = "empty"
    return rep.sort_values("pct_missing", ascending=False)


def duplicate_report(prod: pd.DataFrame, sales: pd.DataFrame) -> dict[str, int]:
    prod_keys, sales_keys = set(prod[KEY]), set(sales[KEY])
    return {
        "product: fully duplicated rows": int(prod.duplicated().sum()),
        "product: duplicated PROD_SPK": int(prod["PROD_SPK"].duplicated().sum()),
        "product: duplicated PROD_COD (SKU)": int(prod["PROD_COD"].duplicated().sum()),
        "product: duplicated BAR_COD": int(prod["BAR_COD"].duplicated().sum()),
        "product: duplicated PROD_COD_EQUIV (re-coded items, expected)": int(prod["PROD_COD_EQUIV"].duplicated().sum()),
        "sales: duplicated keys": int(sales[KEY].duplicated().sum()),
        "sales: keys not in product table": len(sales_keys - prod_keys),
        "product: colourways without sales": len(prod_keys - sales_keys),
    }


# --------------------------------------------------------------------------- colourway table
def build_items(prod: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Collapse SKUs (sizes) into one row per colourway and attach sales."""
    prod = prod.copy()
    # PROD_DES ends with the size ("... Gold U"): remove it so the text is size independent.
    prod["PROD_DES_BASE"] = [
        d[: -len(sz) - 1] if isinstance(d, str) and isinstance(sz, str) and d.endswith(" " + sz) else d
        for d, sz in zip(prod["PROD_DES"], prod["SZ_DES"])
    ]
    grouped = prod.groupby(KEY, sort=True)
    items = grouped.first()

    # Sizes must agree on the attributes the model uses. Where they don't, keep the most
    # common value (instead of whichever size comes first) and record the conflict for review.
    attrs = [c for c in SKU_INVARIANT_COLS if c in prod.columns]
    varies = grouped[attrs].nunique(dropna=False) > 1
    items["sku_conflicts"] = varies.apply(lambda r: "|".join(r.index[r]), axis=1)
    conflicted = varies.index[varies.any(axis=1)]
    if len(conflicted):
        modes = (prod[prod[KEY].isin(conflicted)].groupby(KEY)[attrs]
                 .agg(lambda s: s.mode(dropna=False).iat[0]))
        items.loc[modes.index, attrs] = modes

    # Re-coded items merge SKUs of an old and a new model code: keep every code the colourway owns.
    refs = pd.concat([prod[[KEY, c]].set_axis([KEY, "ref"], axis=1) for c in ["PROD_REF", "PROD_REF_EQUIV"]])
    items["refs_all"] = refs.dropna().drop_duplicates().groupby(KEY)["ref"].agg(lambda s: "|".join(sorted(s)))
    items["n_skus"] = grouped.size()
    items["sizes"] = grouped["SZ_DES"].agg(lambda s: "|".join(s.dropna().unique()))
    items["n_images_listed"] = grouped["PROG_IMAGE"].nunique()
    items = items.reset_index()

    items = items.merge(sales, on=KEY, how="left")
    return items


# --------------------------------------------------------------------------- item-level checks
def _add_issue(items: pd.DataFrame, mask: pd.Series, code: str) -> None:
    items.loc[mask.fillna(False).astype(bool), "issues"] += code + ";"


def check_images(items: pd.DataFrame, verify_files: bool = True) -> None:
    """Validate PROG_IMAGE paths and their link to the image files."""
    parts = items["PROG_IMAGE"].str.extract(r"^/(?P<season>[^/]+)/(?P<cat>\d{2})/(?P<stem>[^/]+)$")
    items["img_stem"] = parts["stem"].str.rstrip(".")
    path_ok = parts["stem"].notna() & parts["season"].str.fullmatch(r"\d{3}").fillna(False)
    _add_issue(items, ~path_ok & items["PROG_IMAGE"].notna(), "IMG_PATH_INVALID")

    files = {os.path.splitext(f)[0]: f for f in os.listdir(IMG_DIR)} if IMG_DIR.exists() else {}
    items["img_file"] = items["img_stem"].map(files)
    items["has_image"] = items["img_file"].notna()
    _add_issue(items, ~items["has_image"], "IMG_FILE_MISSING")

    _add_issue(items, parts["cat"].notna() & (parts["cat"] != items["CAT_COD"]), "IMG_CAT_MISMATCH")
    ref_ok = pd.Series([isinstance(s, str) and any(s.startswith(r) for r in refs.split("|"))
                        for s, refs in zip(items["img_stem"], items["refs_all"])], index=items.index)
    _add_issue(items, items["img_stem"].notna() & ~ref_ok, "IMG_REF_MISMATCH")

    # File name pattern: <ref:6 digits><colour code: "_BK", "1BK", "DNC" or "CT">_<n>; generic images have no colour.
    img_clr = items["img_stem"].str.extract(r"^\d{6}([_\dA-Z]?[A-Z]{2})_")[0]
    items["img_colour_code"] = img_clr
    same = img_clr.str.lstrip("_") == items["CLR_COD"].str.lstrip("_")  # "_BK" == "BK", "1BK" != "BK"
    _add_issue(items, img_clr.notna() & ~same, "IMG_COLOUR_MISMATCH")
    _add_issue(items, items["img_stem"].notna() & img_clr.isna(), "IMG_GENERIC")

    # Content hash catches identical pictures saved under different names.
    items["img_md5"] = None
    items["img_readable"] = pd.NA
    if verify_files:
        from PIL import Image  # imported lazily: only needed for file checks

        md5s, readable = {}, {}
        for idx, f in items["img_file"].dropna().items():
            path = IMG_DIR / f
            md5s[idx] = hashlib.md5(path.read_bytes()).hexdigest()
            try:
                with Image.open(path) as im:
                    im.verify()
                readable[idx] = True
            except Exception:
                readable[idx] = False
        items["img_md5"] = pd.Series(md5s, dtype="object")
        items["img_readable"] = pd.Series(readable, dtype="boolean")
        _add_issue(items, items["img_readable"] == False, "IMG_UNREADABLE")  # noqa: E712

    # Same image (by path or by content) used by colourways with a different colour or model.
    img_key = items["img_md5"].fillna(items["PROG_IMAGE"])
    n_clr = items.groupby(img_key)["CLR_COD"].transform("nunique")
    n_ref = items.groupby(img_key)["PROD_REF"].transform("nunique")
    items["img_shared_with"] = items.groupby(img_key)[KEY].transform(lambda s: "|".join(s) if len(s) > 1 else "")
    _add_issue(items, img_key.notna() & ((n_clr > 1) | (n_ref > 1)), "IMG_SHARED")


def _colour_vocabulary(items: pd.DataFrame) -> list[str]:
    return sorted(items["CLR_DES"].dropna().unique(), key=len, reverse=True)


def check_colour_consistency(items: pd.DataFrame) -> None:
    """CLR_COD <-> CLR_DES <-> PROD_DES."""
    # 1) the code is part of the key: PROD_CLR = PROD_REF + CLR_COD
    _add_issue(items, items["PROD_CLR"] != items["PROD_REF"] + items["CLR_COD"], "TAB_KEY_INCONSISTENT")

    # 2) the letters of CLR_COD ("_BK", "1BK" -> "BK"; "DNC" stays) must always mean the same colour name
    letters = items["CLR_COD"].str.replace(r"^[_\d]", "", regex=True)
    majority = items.groupby(letters)["CLR_DES"].agg(lambda s: s.mode().iat[0])
    items["clr_expected_name"] = letters.map(majority)
    _add_issue(items, items["CLR_DES"] != items["clr_expected_name"], "CLR_CODE_NAME_INCONSISTENT")

    # 3) PROD_DES ("Earring GLDN DEL Gold") should mention CLR_DES, normally at the end.
    vocab = _colour_vocabulary(items)
    status = []
    for desc, clr in zip(items["PROD_DES_BASE"], items["CLR_DES"]):
        if not isinstance(desc, str) or not isinstance(clr, str):
            status.append("missing_data")
        elif clr.lower() in desc.lower():
            status.append("ok")
        else:
            # Colour words are Title Case; theme names are upper case ("SILVER BASICS"), so match case-sensitively.
            other = next((c for c in vocab if desc.endswith(" " + c)), None)
            status.append(f"conflict:{other}" if other else "not_found")
    items["desc_colour_status"] = status
    s = items["desc_colour_status"]
    _add_issue(items, s.str.startswith("conflict"), "CLR_CONFLICT_DESC")
    _add_issue(items, s == "not_found", "CLR_NOT_IN_DESC")


def _singular(word: str) -> str:
    w = word.lower().strip()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w  # Bracelets -> bracelet


def _type_word(desc: str) -> str | None:
    if not isinstance(desc, str) or not desc:
        return None
    tokens = desc.split()
    if tokens[0].lower() == "set":
        tokens = tokens[2:] if len(tokens) > 1 and tokens[1].lower() == "of" else tokens[1:]
    if not tokens:
        return None
    return _singular(tokens[0])


def check_type_consistency(items: pd.DataFrame, min_support: int = 20, max_share: float = 0.05) -> None:
    """First word of PROD_DES (item type) vs GFA_DES_EN, learned from the data itself.

    For each type word with enough items, a family that covers less than `max_share`
    of them is treated as a contradiction (e.g. "Necklace ..." labelled Earrings).
    """
    items["desc_type_word"] = items["PROD_DES_BASE"].map(_type_word)
    counts = items.groupby(["desc_type_word", "GFA_DES_EN"]).size()
    totals = counts.groupby(level=0).transform("sum")
    share = (counts / totals).rename("share")
    support = totals.rename("support")
    stats = pd.concat([share, support], axis=1).reset_index()
    items_stats = items[["desc_type_word", "GFA_DES_EN"]].merge(stats, how="left", on=["desc_type_word", "GFA_DES_EN"])
    items["desc_type_share"] = items_stats["share"].round(3).values
    # The item's own family named in the text wins ("Shoulder Strap for Bags" -> Shoulder Strap).
    own_named = pd.Series([isinstance(d, str) and isinstance(g, str) and _singular(g) in d.lower()
                           for d, g in zip(items["PROD_DES_BASE"], items["GFA_DES_EN"])])
    conflict = (items_stats["support"] >= min_support) & (items_stats["share"] < max_share) & ~own_named
    _add_issue(items, pd.Series(conflict.values, index=items.index), "TYPE_CONFLICT_DESC")


def check_sales(items: pd.DataFrame) -> None:
    _add_issue(items, items["SALES_QTY"].isna(), "SALES_MISSING")
    _add_issue(items, items["SALES_QTY"] == 0, "SALES_ZERO_QTY")


def run_checks(verify_files: bool = True) -> tuple[pd.DataFrame, dict]:
    prod, sales = load_data()
    items = build_items(prod, sales)
    items["issues"] = ""
    check_images(items, verify_files)
    check_colour_consistency(items)
    check_type_consistency(items)
    check_sales(items)
    _add_issue(items, items["sku_conflicts"] != "", "SKU_ATTR_CONFLICT")

    items["issues"] = items["issues"].str.rstrip(";")
    sev = items["issues"].map(
        lambda s: max((SEVERITY_RANK[ISSUES[c]] for c in s.split(";") if c), default=-1))
    items["priority"] = sev.map({3: "high", 2: "medium", 1: "low", 0: "info", -1: "none"})
    context = {"prod": prod, "sales": sales,
               "missing": missing_report(prod), "duplicates": duplicate_report(prod, sales)}
    return items, context


# --------------------------------------------------------------------------- reporting
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_check_outputs(items: pd.DataFrame, ctx: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    items.to_parquet(PROCESSED_DIR / "items_checked.parquet", index=False)
    items.to_csv(OUT_DIR / "items_checked.csv", index=False, encoding="utf-8-sig")
    ctx["missing"].to_csv(OUT_DIR / "missing_values.csv", encoding="utf-8-sig")

    issue_counts = (items["issues"].str.split(";").explode().loc[lambda s: s != ""]
                    .value_counts().rename_axis("issue").reset_index(name="n_items"))
    issue_counts["severity"] = issue_counts["issue"].map(ISSUES)
    issue_counts["pct_items"] = (issue_counts["n_items"] / len(items) * 100).round(2)
    by_cat = pd.crosstab(items["CAT_DES_EN"], items["priority"]).reset_index()
    miss = ctx["missing"]
    dropped = miss[miss["status"].isin(["empty", "constant"])].index.tolist()

    examples = []
    for code in issue_counts["issue"]:
        if ISSUES[code] in ("high", "medium"):
            ex = items.loc[items["issues"].str.contains(code, regex=False),
                           [KEY, "PROD_DES_BASE", "CLR_DES", "GFA_DES_EN", "PROG_IMAGE"]].head(5)
            examples.append(f"**{code}**\n\n{_md_table(ex)}\n")

    report = [
        "# Sprint 1: automatic check report",
        "",
        f"- SKU rows: {len(ctx['prod']):,} · colourways: {len(items):,} · with image file: {int(items['has_image'].sum()):,}",
        f"- Colourways needing review: high {int((items['priority'] == 'high').sum())}, "
        f"medium {int((items['priority'] == 'medium').sum())}",
        "",
        "## Duplicates and keys",
        "",
        _md_table(pd.DataFrame(list(ctx["duplicates"].items()), columns=["check", "count"])),
        "",
        "## Issues per colourway",
        "",
        _md_table(issue_counts[["issue", "severity", "n_items", "pct_items"]]),
        "",
        "## Priority by category",
        "",
        _md_table(by_cat),
        "",
        "## Missing values",
        "",
        f"Full table: `missing_values.csv`. Empty or constant columns ({len(dropped)}): {', '.join(dropped)}.",
        "",
        "## Examples",
        "",
        *examples,
        "## What these checks cannot see",
        "",
        "They compare codes and text. They cannot see what an image actually shows (e.g. a necklace "
        "row with an earring photo). That needs human review (`batch` command) or a vision model.",
    ]
    (OUT_DIR / "check_report.md").write_text("\n".join(report), encoding="utf-8")


# --------------------------------------------------------------------------- batch split
def split_batches(items: pd.DataFrame, n_members: int) -> pd.Series:
    """Deterministic, balanced split. Returns batch id (1..n) per item.

    All colours of the same model (PROD_REF) go to the same person, so shared images
    and colour variants are reviewed together. Within each category, models are
    assigned greedily to the batch with the fewest items, so every batch gets a
    similar size and category mix.
    """
    groups = (items.groupby(["CAT_DES_EN", "PROD_REF"]).size().reset_index(name="n")
              .sort_values(["CAT_DES_EN", "n", "PROD_REF"], ascending=[True, False, True]))
    load = [0] * n_members
    assignment = {}
    for cat, block in groups.groupby("CAT_DES_EN", sort=True):
        cat_load = [0] * n_members
        for ref, n in zip(block["PROD_REF"], block["n"]):
            b = min(range(n_members), key=lambda i: (cat_load[i], load[i], i))
            assignment[(cat, ref)] = b + 1
            cat_load[b] += n
            load[b] += n
    return pd.Series([assignment[(c, r)] for c, r in zip(items["CAT_DES_EN"], items["PROD_REF"])],
                     index=items.index, name="batch")


def _review_sheet(batch: pd.DataFrame, title: str, html_path: Path) -> None:
    """A lightweight HTML contact sheet: image + labels + flags, flagged items first."""
    cards = []
    for r in batch.itertuples(index=False):
        src = Path(os.path.relpath(IMG_DIR / r.img_file, html_path.parent)).as_posix() if isinstance(r.img_file, str) else ""
        img = f'<img loading="lazy" src="{html.escape(src)}">' if src else '<div class="noimg">no image</div>'
        cards.append(
            f'<div class="card {r.priority}">{img}<b>{html.escape(r.PROD_CLR_EQUIV)}</b>'
            f"<span>{html.escape(str(r.PROD_DES_BASE))}</span>"
            f"<span>{html.escape(str(r.CAT_DES_EN))} › {html.escape(str(r.GFA_DES_EN))} · {html.escape(str(r.CLR_DES))}</span>"
            f'<i>{html.escape(r.issues.replace(";", " · "))}</i></div>')
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:13px system-ui,sans-serif;margin:16px;background:#fff;color:#111}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}}
.card{{border:1px solid #ddd;border-radius:6px;padding:6px;display:flex;flex-direction:column;gap:2px}}
.card img,.noimg{{width:100%;aspect-ratio:1;object-fit:contain;background:#f6f6f6}}.noimg{{display:grid;place-items:center;color:#888}}
.high{{border-color:#d33;border-width:2px}}.medium{{border-color:#e90}}i{{color:#b22;font-size:11px}}</style>
<h1>{html.escape(title)}</h1><p>{len(batch)} items, flagged first. Record decisions in the matching CSV
(column <code>review_status</code>: ok | fix | drop_image | discard | team_review).</p>
<div class="grid">{''.join(cards)}</div>"""
    html_path.write_text(page, encoding="utf-8")


def _has_review_work(csv_path: Path) -> bool:
    """True if a batch CSV already has any decision filled in (never overwrite a reviewer's work)."""
    done = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    cols = [c for c in REVIEW_COLS if c in done.columns]
    return bool((done[cols] != "").to_numpy().any())


def write_batches(items: pd.DataFrame, n_members: int, only_id: int | None) -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    items = items.assign(batch=split_batches(items, n_members))
    order = items["priority"].map({"high": 0, "medium": 1, "low": 2, "info": 3, "none": 4})
    items = items.assign(_o=order).sort_values(["batch", "_o", "CAT_DES_EN", "PROD_REF", KEY])
    ids = [only_id] if only_id else range(1, n_members + 1)
    for b in ids:
        batch = items.loc[items["batch"] == b, BATCH_COLS].copy()
        for c in REVIEW_COLS:
            batch[c] = ""
        name = f"batch_{b:02d}_of_{n_members:02d}"
        csv_path = BATCH_DIR / f"{name}.csv"
        if csv_path.exists() and _has_review_work(csv_path):
            print(f"{name}.csv already contains review decisions: kept as is (only the HTML sheet was rebuilt).")
        else:
            batch.to_csv(csv_path, index=False, encoding="utf-8-sig")
        owner = f" ({TEAM[b]})" if n_members == len(TEAM) else ""
        _review_sheet(batch, f"Sprint 1 review – {name}{owner}", BATCH_DIR / f"{name}.html")
        print(f"{name}{owner}: {len(batch)} items ({int((batch['priority'] == 'high').sum())} high priority) "
              f"-> {BATCH_DIR / name}.csv / .html")


# --------------------------------------------------------------------------- merge
_FIX_SEPARATOR = re.compile(r";\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)")


def _parse_fixes(text: str) -> list[tuple[str, str]]:
    """'CLR_DES=Black; COMPOSITION=Pearl; Zinc' -> [('CLR_DES', 'Black'), ('COMPOSITION', 'Pearl; Zinc')]

    A ';' only starts a new fix when it is followed by 'COLUMN=', so values may contain ';'.
    """
    out = []
    for part in _FIX_SEPARATOR.split(str(text)):
        if "=" in part:
            col, val = part.split("=", 1)
            out.append((col.strip(), val.strip()))
    return out


def merge_batches(input_dir: Path, include_unresolved: bool = False) -> None:
    items = pd.read_parquet(PROCESSED_DIR / "items_checked.parquet")
    files = sorted(input_dir.glob("batch_*_of_*.csv"))
    if not files:
        raise SystemExit(f"No batch_*_of_*.csv files in {input_dir}")
    reviews = pd.concat([pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False)
                         .assign(source_file=f.name) for f in files], ignore_index=True)
    reviews = reviews[[KEY, *REVIEW_COLS, "source_file"]]
    reviews["review_status"] = reviews["review_status"].str.strip().str.lower()

    errors = []
    dup = reviews[KEY].duplicated(keep=False)
    if dup.any():
        errors.append(f"{reviews.loc[dup, KEY].nunique()} items appear in more than one batch file")
    unknown = set(reviews[KEY]) - set(items[KEY])
    if unknown:
        errors.append(f"{len(unknown)} unknown keys, e.g. {sorted(unknown)[:3]}")
    bad = ~reviews["review_status"].isin(REVIEW_STATUSES | {""})
    if bad.any():
        errors.append(f"invalid review_status values: {sorted(reviews.loc[bad, 'review_status'].unique())}")
    if errors:
        raise SystemExit("Merge aborted:\n- " + "\n- ".join(errors))

    items = items.merge(reviews, on=KEY, how="left")
    not_in_batches = items["source_file"].isna()
    # Unreviewed items without issues are accepted; unreviewed flagged items stay pending.
    auto_ok = (items["review_status"].fillna("") == "") & items["priority"].isin(["none", "info"])
    items.loc[auto_ok, "review_status"] = "ok"
    items["review_status"] = items["review_status"].replace("", pd.NA).fillna("pending")

    log = []
    for idx, r in items[items["review_status"] == "fix"].iterrows():
        for col, val in _parse_fixes(r["fixes"]):
            if col not in items.columns:
                errors.append(f"{r[KEY]}: unknown column in fixes '{col}'")
                continue
            log.append({KEY: r[KEY], "column": col, "old": items.at[idx, col], "new": val,
                        "reviewer": r["reviewer"], "source_file": r["source_file"]})
            items.at[idx, col] = val
    if errors:
        raise SystemExit("Merge aborted:\n- " + "\n- ".join(errors))

    drop_img = items["review_status"] == "drop_image"
    for idx in items.index[drop_img]:
        log.append({KEY: items.at[idx, KEY], "column": "img_file", "old": items.at[idx, "img_file"], "new": None,
                    "reviewer": items.at[idx, "reviewer"], "source_file": items.at[idx, "source_file"]})
    items.loc[drop_img, ["img_file", "PROG_IMAGE"]] = None
    items.loc[drop_img, "has_image"] = False

    discarded = items[items["review_status"] == "discard"]
    team = items[items["review_status"].isin(["team_review", "pending"])]
    # Unresolved items (flagged but not reviewed, or sent to team review) stay out of the clean
    # table unless explicitly requested, so nothing doubtful reaches the features.
    excluded = ["discard"] if include_unresolved else ["discard", "team_review", "pending"]
    clean = items[~items["review_status"].isin(excluded)].drop(columns=["source_file"])

    clean.to_parquet(PROCESSED_DIR / "items_clean.parquet", index=False)
    clean.to_csv(OUT_DIR / "items_clean.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(log, columns=[KEY, "column", "old", "new", "reviewer", "source_file"]).to_csv(
        OUT_DIR / "changes_log.csv", index=False, encoding="utf-8-sig")
    discarded.to_csv(OUT_DIR / "discarded.csv", index=False, encoding="utf-8-sig")
    team.to_csv(OUT_DIR / "team_review.csv", index=False, encoding="utf-8-sig")

    summary = items["review_status"].value_counts()
    print(f"Merged {len(files)} batch files. Status counts:\n{summary.to_string()}")
    if not_in_batches.any():
        print(f"WARNING: {int(not_in_batches.sum())} items were not in any batch file.")
    print(f"Fixes applied: {len(log)} -> {OUT_DIR / 'changes_log.csv'}")
    print(f"Clean table: {PROCESSED_DIR / 'items_clean.parquet'} ({len(clean)} items)")
    if len(team) and not include_unresolved:
        print(f"Left out of the clean table until resolved: {len(team)} items (pending / team_review) "
              f"-> {OUT_DIR / 'team_review.csv'}. Use --include-unresolved to keep them.")


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run all automatic checks on the full dataset")
    c.add_argument("--skip-image-verify", action="store_true", help="skip opening/hashing image files (faster)")
    b = sub.add_parser("batch", help="write review batch(es) for team members")
    b.add_argument("--members", type=int, required=True)
    b.add_argument("--id", type=int, help="only write this batch (1..members)")
    b.add_argument("--skip-image-verify", action="store_true")
    m = sub.add_parser("merge", help="merge reviewed batches and apply fixes")
    m.add_argument("--input-dir", type=Path, default=BATCH_DIR)
    m.add_argument("--include-unresolved", action="store_true",
                   help="keep pending / team_review items in the clean table")
    args = ap.parse_args()

    if args.cmd == "merge":
        merge_batches(args.input_dir, args.include_unresolved)
        return
    if args.cmd == "batch" and args.id and not 1 <= args.id <= args.members:
        raise SystemExit("--id must be between 1 and --members")
    items, ctx = run_checks(verify_files=not args.skip_image_verify)
    write_check_outputs(items, ctx)
    print(f"Checked {len(items):,} colourways. Report: {OUT_DIR / 'check_report.md'}")
    print(items["priority"].value_counts().to_string())
    if args.cmd == "batch":
        write_batches(items, args.members, args.id)


if __name__ == "__main__":
    main()
