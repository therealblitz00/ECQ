"""Sprint 1 review workflow: batch split, batch files, merge and the review app's Batch class."""
from __future__ import annotations

import pandas as pd
import pytest

import review_app
import sprint1_preprocess as sp


# --------------------------------------------------------------------------- batch split
def make_items(n_refs: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_refs):
        cat = ["Apparel", "Hand Bag", "Jewellery"][i % 3]
        for c in range(1 + i % 4):  # 1 to 4 colours per model
            rows.append({sp.KEY: f"{200000 + i}_C{c}", "PROD_REF": str(200000 + i), "CAT_DES_EN": cat})
    return pd.DataFrame(rows)


def test_split_batches_is_deterministic_balanced_and_keeps_models_together():
    items = make_items()
    batch = sp.split_batches(items, 5)
    assert batch.equals(sp.split_batches(items.sample(frac=1, random_state=0).sort_index(), 5))
    assert set(batch) == {1, 2, 3, 4, 5}
    assert items.assign(b=batch).groupby("PROD_REF")["b"].nunique().eq(1).all()
    sizes = batch.value_counts()
    assert sizes.max() - sizes.min() <= 4  # at most one model (max 4 colours) apart


def test_write_batches_never_overwrites_review_decisions(checked_items):
    items = checked_items.reset_index()
    sp.write_batches(items, n_members=2, only_id=None)
    files = sorted(p.name for p in sp.BATCH_DIR.glob("batch_*.csv"))
    assert files == ["batch_01_of_02.csv", "batch_02_of_02.csv"]

    path = sp.BATCH_DIR / "batch_01_of_02.csv"
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    assert list(df.columns) == sp.BATCH_COLS + sp.REVIEW_COLS
    assert sp._has_review_work(path) is False
    df.loc[0, "review_status"] = "ok"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    sp.write_batches(items, n_members=2, only_id=1)
    again = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    assert again.loc[0, "review_status"] == "ok"


# --------------------------------------------------------------------------- merge
@pytest.mark.parametrize("text, expected", [
    ("CLR_DES=Black", [("CLR_DES", "Black")]),
    ("CLR_DES=Black; COMPOSITION=Pearl; Zinc", [("CLR_DES", "Black"), ("COMPOSITION", "Pearl; Zinc")]),
    (" GFA_DES_EN = Rings ;FINISHING=Golden", [("GFA_DES_EN", "Rings"), ("FINISHING", "Golden")]),
    ("", []),
])
def test_parse_fixes(text, expected):
    assert sp._parse_fixes(text) == expected


def write_merge_inputs(reviews: list[dict]) -> None:
    items = pd.DataFrame({
        sp.KEY: ["A_1", "B_1", "C_1", "D_1", "E_1"],
        "priority": ["high", "medium", "medium", "medium", "none"],
        "CLR_DES": ["Navy", "Gold", "Silver", "Black", "Ecru"],
        "PROG_IMAGE": ["/241/52/A_1_1"] * 5,
        "img_file": ["A_1_1.jpg"] * 5,
        "has_image": [True] * 5,
    })
    items.to_parquet(sp.PROCESSED_DIR / "items_checked.parquet", index=False)
    batch = pd.DataFrame(reviews, columns=[sp.KEY, *sp.REVIEW_COLS]).fillna("")
    batch.to_csv(sp.BATCH_DIR / "batch_01_of_01.csv", index=False, encoding="utf-8-sig")


def test_merge_applies_decisions_and_logs_changes(toy_dirs):
    write_merge_inputs([
        {sp.KEY: "A_1", "review_status": "fix", "fixes": "CLR_DES=Black", "reviewer": "Manuel"},
        {sp.KEY: "B_1", "review_status": "discard", "reviewer": "Manuel"},
        {sp.KEY: "C_1", "review_status": "drop_image", "reviewer": "Manuel"},
        {sp.KEY: "D_1"},  # flagged but not reviewed -> pending
        {sp.KEY: "E_1"},  # no issues -> accepted automatically
    ])
    sp.merge_batches(sp.BATCH_DIR)

    clean = pd.read_parquet(sp.PROCESSED_DIR / "items_clean.parquet").set_index(sp.KEY)
    assert sorted(clean.index) == ["A_1", "C_1", "E_1"]
    assert clean.at["A_1", "CLR_DES"] == "Black"
    assert pd.isna(clean.at["C_1", "img_file"]) and not clean.at["C_1", "has_image"]
    assert clean.at["E_1", "review_status"] == "ok"

    log = pd.read_csv(sp.OUT_DIR / "changes_log.csv", encoding="utf-8-sig")
    assert set(zip(log[sp.KEY], log["column"])) == {("A_1", "CLR_DES"), ("C_1", "img_file")}
    assert pd.read_csv(sp.OUT_DIR / "discarded.csv", encoding="utf-8-sig")[sp.KEY].tolist() == ["B_1"]
    assert pd.read_csv(sp.OUT_DIR / "team_review.csv", encoding="utf-8-sig")[sp.KEY].tolist() == ["D_1"]


def test_merge_can_keep_unresolved_items(toy_dirs):
    write_merge_inputs([{sp.KEY: "D_1", "review_status": "team_review"}])
    sp.merge_batches(sp.BATCH_DIR, include_unresolved=True)
    clean = pd.read_parquet(sp.PROCESSED_DIR / "items_clean.parquet")
    assert len(clean) == 5


@pytest.mark.parametrize("reviews", [
    [{sp.KEY: "A_1", "review_status": "maybe"}],  # invalid status
    [{sp.KEY: "Z_9", "review_status": "ok"}],  # unknown key
    [{sp.KEY: "A_1", "review_status": "ok"}, {sp.KEY: "A_1", "review_status": "ok"}],  # duplicated key
    [{sp.KEY: "A_1", "review_status": "fix", "fixes": "NOT_A_COLUMN=1"}],  # unknown column
])
def test_merge_aborts_on_bad_input(toy_dirs, reviews):
    write_merge_inputs(reviews)
    with pytest.raises(SystemExit, match="Merge aborted"):
        sp.merge_batches(sp.BATCH_DIR)
    assert not (sp.PROCESSED_DIR / "items_clean.parquet").exists()


# --------------------------------------------------------------------------- review app
@pytest.fixture
def batch(checked_items, monkeypatch):
    for name in ["BATCH_DIR", "EMB_DIR", "PROCESSED_DIR"]:
        monkeypatch.setattr(review_app, name, getattr(sp, name))
    sp.write_batches(checked_items.reset_index(), n_members=review_app.N_MEMBERS, only_id=4)
    return review_app.Batch(4)


def test_issue_labels_cover_every_text_check():
    text_checks = [c for c in sp.ISSUES if not c.startswith(("VIS_", "IMG_NEAR", "IMG_NOT"))]
    assert set(text_checks) <= set(review_app.ISSUE_LABELS)


def test_review_app_saves_decisions_in_the_merge_format(batch):
    key = batch.df.at[0, sp.KEY]
    batch.save(key, "fix", "CLR_DES=Black", "colour was wrong")
    df = pd.read_csv(batch.path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    row = df.set_index(sp.KEY).loc[key]
    assert (row["review_status"], row["fixes"], row["reviewer"]) == ("fix", "CLR_DES=Black", sp.TEAM[4])

    batch.save(key, "ok", "CLR_DES=Black", "")  # fixes only kept for "fix"
    assert batch.df.at[0, "fixes"] == ""

    batch.save(key, "", "", "")  # clearing a decision also clears the reviewer
    assert batch.df.at[0, "reviewer"] == ""


def test_review_app_rejects_bad_input(batch):
    with pytest.raises(ValueError, match="invalid status"):
        batch.save(batch.df.at[0, sp.KEY], "maybe", "", "")
    with pytest.raises(ValueError, match="unknown product"):
        batch.save("999999_XX", "ok", "", "")


def test_review_app_raises_priority_with_image_flags(checked_items, monkeypatch):
    for name in ["BATCH_DIR", "EMB_DIR", "PROCESSED_DIR"]:
        monkeypatch.setattr(review_app, name, getattr(sp, name))
    sp.write_batches(checked_items.reset_index(), n_members=1, only_id=1)
    monkeypatch.setattr(review_app, "N_MEMBERS", 1)
    pd.DataFrame({sp.KEY: ["100001_BK"], "vis_issues": ["VIS_TYPE_MISMATCH"],
                  "vis_note": ["photo looks like a bag"]}).to_parquet(sp.EMB_DIR / "image_audit.parquet")

    items = {r[sp.KEY]: r for r in review_app.Batch(1).items()}
    assert items["100001_BK"]["priority"] == "high"
    assert "Photo looks like a bag" in items["100001_BK"]["issue_labels"]
