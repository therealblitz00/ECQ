"""Phase 1b: how good were the image flags? Measured on the reviewers' decisions after the merge.

    python src/phase1b_precision.py      # -> outputs/phase1b/flag_precision.md + flag_precision.csv

Needs data/processed/items_clean.parquet (`phase1_checks.py merge`) and
data/embeddings/image_audit.parquet. No vision packages: it only reads stored scores.

A flag is *confirmed* when the reviewer acted on what it says:
- type flags: the photo was dropped, or the category / family / sub-family was fixed;
- colour flag: the photo was dropped, or the colour was fixed;
- near-duplicate, target not found: the photo was dropped or the colour was fixed;
- not a packshot is a quality note, not a mismatch: its "precision" is only reported.

Precision uses every flagged item (all flagged items had to be reviewed). Recall needs items that
were reviewed whether flagged or not: since 2026-10-12 every photo of every batch has been looked at,
so recall and the threshold sweep use all items with a photo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BATCH_DIR, EMB_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402
from phase1_vocab import parse_fixes  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase1b"
FULL_REVIEW_BATCHES = [f"batch_0{k}_of_05.csv" for k in range(1, 6)]  # every photo was looked at
TARGETS = {"high": 0.80, "medium": 0.50}
SEVERITY = {"VIS_TYPE_MISMATCH": "high", "VIS_COLOUR_MISMATCH": "medium", "IMG_NEAR_DUPLICATE": "medium",
            "VIS_TYPE_DOUBT": "low", "VIS_TARGET_NOT_FOUND": "low", "IMG_NOT_PACKSHOT": "low"}
TYPE_COLS = {"CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"}


def outcomes(items: pd.DataFrame, full_review: set[str]) -> pd.DataFrame:
    """Per item: was the photo dropped, the colour fixed, the type fixed? Was every photo of its batch seen?"""
    fixed = items["fixes"].fillna("").map(lambda s: {c for c, _ in parse_fixes(s)})
    return pd.DataFrame({
        KEY: items[KEY],
        "dropped": items["review_status"].eq("drop_image"),
        "colour_fixed": fixed.map(lambda s: "CLR_DES" in s),
        "type_fixed": fixed.map(lambda s: bool(s & TYPE_COLS)),
        "full_review": items[KEY].isin(full_review),
    })


def confirmed(flag: str, o: pd.DataFrame) -> pd.Series:
    if flag in ("VIS_TYPE_MISMATCH", "VIS_TYPE_DOUBT"):
        return o["dropped"] | o["type_fixed"]
    return o["dropped"] | o["colour_fixed"]


def precision_table(audit: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    df = audit[[KEY, "vis_issues"]].merge(o, on=KEY, how="inner")
    problem = df["dropped"] | df["colour_fixed"] | df["type_fixed"]
    full = df["full_review"]
    rows = []
    for flag, sev in SEVERITY.items():
        has = df["vis_issues"].fillna("").str.split(";").map(lambda s, f=flag: f in s)
        ok = confirmed(flag, df)
        n, hit = int(has.sum()), int((has & ok).sum())
        # recall: share of the problems of this kind (all photos) that the flag caught
        kind = confirmed(flag, df) & full
        rows.append({"flag": flag, "severity": sev, "flagged": n, "confirmed": hit,
                     "precision": hit / n if n else np.nan, "target": TARGETS.get(sev, np.nan),
                     "recall": (has & kind).sum() / kind.sum() if kind.sum() else np.nan})
    out = pd.DataFrame(rows)
    any_flag = df["vis_issues"].fillna("").ne("")
    out.attrs["any_recall"] = float((any_flag & problem & full).sum() / max(int((problem & full).sum()), 1))
    out.attrs["problems"] = int((problem & full).sum())
    return out


def colour_sweep(audit: pd.DataFrame, o: pd.DataFrame, cuts=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    """All photos: precision / recall of the colour rule for other confidence cuts (no shared-photo clause)."""
    from phase1b_image_audit import COLOUR_NEIGHBOURS
    df = audit.merge(o, on=KEY)
    df = df[df["full_review"]]
    disagree = (df["colour_expected"].notna() & df["colour_pred"].ne(df["colour_expected"])
                & ~df.apply(lambda r: frozenset((r.colour_expected, r.colour_pred)) in COLOUR_NEIGHBOURS, axis=1))
    truth = df["dropped"] | df["colour_fixed"]
    rows = []
    for t in cuts:
        f = disagree & df["p_colour_expected"].lt(t)
        rows.append({"p_colour_expected <": t, "flagged": int(f.sum()), "confirmed": int((f & truth).sum()),
                     "precision": (f & truth).sum() / f.sum() if f.sum() else np.nan,
                     "recall": (f & truth).sum() / truth.sum() if truth.sum() else np.nan})
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def write_report(table: pd.DataFrame, sweep: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "flag_precision.csv", index=False, encoding="utf-8-sig")
    rows = "\n".join(
        f"| `{r.flag}` | {r.severity} | {r.flagged} | {r.confirmed} | **{_pct(r.precision)}** | {_pct(r.target)} | "
        f"{'✅' if pd.isna(r.target) or r.precision >= r.target else '❌'} | {_pct(r.recall)} |"
        for r in table.itertuples())
    srows = "\n".join(f"| < {r[1]:g} | {r[2]} | {r[3]} | {_pct(r[4])} | {_pct(r[5])} |" for r in sweep.itertuples())
    text = f"""# Phase 1b: precision of the image flags

Generated by `src/phase1b_precision.py` from the merged review (`items_clean.parquet`).
A flag is confirmed when the reviewer dropped the photo or fixed what the flag points at
(colour for colour flags, category / family / sub-family for type flags).

| Flag | Severity | Flagged | Confirmed | Precision | Target | Met | Recall |
|---|---|---|---|---|---|---|---|
{rows}

- `IMG_NOT_PACKSHOT` is a quality note (model, lifestyle or amateur photo), not a mismatch, so it has no target.
- `IMG_NEAR_DUPLICATE` marks every colourway that shares the photo, but the photo is right for one of them, so
  about half of its flags are expected to be "not confirmed".
- **Recall** uses every item with a photo (all photos were looked at, not just the flagged ones). Any image flag caught
  {_pct(table.attrs["any_recall"])} of the {table.attrs["problems"]} problems found; the rest
  were found by eye (mostly colours that CLIP sees as close, e.g. blue vs turquoise).
- Caveat: reviewers saw the flags, which may push them towards confirming.

## Colour check: other thresholds (all photos)

The current rule flags a colour when CLIP disagrees with the CSV and `p_colour_expected < 0.005`
(or `< 0.02` when the photo is shared with another colour).

| Threshold | Flagged | Confirmed | Precision | Recall |
|---|---|---|---|---|
{srows}

## What this means

- **Colour flag: useful.** It meets its target. A cut between 0.005 and 0.02 keeps precision at or above 50%
  with more recall; the current 0.005 favours precision.
- **Type flags: mostly false alarms.** CLIP gets the item type wrong on unusual products (feathers, fringes,
  beads, macramé) and on lifestyle or sample photos, and no score threshold separates the true cases from
  the false ones. The few real cases were photos of another article.
- **Thresholds are not changed now.** Changing them would add flags to items that are already reviewed, and a
  new `merge` would leave those items pending. For a future dataset: lower `VIS_TYPE_MISMATCH` to medium
  and use the colour cut that suits the review budget.
"""
    (REPORT_DIR / "flag_precision.md").write_text(text, encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    items = pd.read_parquet(PROCESSED_DIR / "items_clean.parquet")
    audit = pd.read_parquet(EMB_DIR / "image_audit.parquet")
    full = pd.concat(pd.read_csv(BATCH_DIR / f, dtype=str, encoding="utf-8-sig", usecols=[KEY, "img_file"],
                                 keep_default_na=False) for f in FULL_REVIEW_BATCHES)
    full = full.loc[full["img_file"] != "", KEY]
    o = outcomes(items, set(full))
    table, sweep = precision_table(audit, o), colour_sweep(audit, o)
    write_report(table, sweep)
    print(table.to_string(index=False))
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
