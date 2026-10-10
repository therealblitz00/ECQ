"""Phase 1 automatic checks (src/phase1_checks.py check) on the toy dataset."""
from __future__ import annotations

import pandas as pd
import pytest

import phase1_checks as checks


def issues(items: pd.DataFrame, key: str) -> set[str]:
    return set(filter(None, items.at[key, "issues"].split(";")))


# --------------------------------------------------------------------------- helpers
def test_normalise_text_removes_nbsp_and_extra_spaces():
    s = pd.Series(["Raincoat\xa0", "  Earring   GLDN  ", None])
    assert checks._normalise_text(s).tolist()[:2] == ["Raincoat", "Earring GLDN"]


@pytest.mark.parametrize("desc, expected", [
    ("Earrings GLDN DEL Gold", "earring"),
    ("Set of 3 Rings Gold", "3"),
    ("Set Bracelets Silver", "bracelet"),
    ("Bag Black", "bag"),
    ("Set", None),
    ("", None),
    (None, None),
])
def test_type_word(desc, expected):
    assert checks._type_word(desc) == expected


def test_every_issue_code_has_a_known_severity():
    assert set(checks.ISSUES.values()) <= set(checks.SEVERITY_RANK)


# --------------------------------------------------------------------------- loading and colourway table
def test_load_data_normalises_text_and_composition(toy_data):
    prod, sales = checks.load_data()
    assert prod["PROD_REF"].map(type).eq(str).all()  # codes stay strings
    assert prod["CAT_COD"].eq("52").all()
    assert "Earrings" in set(prod["GFA_DES_EN"])  # trailing \xa0 removed
    assert set(prod.loc[prod["PROD_CLR_EQUIV"] == "100001_BK", "COMPOSITION"]) == {"Enamel; Zinc"}
    assert len(sales) == 4


def test_build_items_collapses_sizes_and_keeps_the_most_common_value(toy_data):
    prod, sales = checks.load_data()
    items = checks.build_items(prod, sales).set_index(checks.KEY)

    assert list(items.index) == ["100001_BK", "100002_GD", "100003_SV", "100004_BK", "100005_GN"]
    a = items.loc["100001_BK"]
    assert a["n_skus"] == 3
    assert a["sizes"] == "S|M|L"
    assert a["PROD_DES_BASE"] == "Earring ESSENTIALS Black"  # size removed
    assert a["PRICE_BASE_W_VAT"] == 9.99  # 2 of 3 sizes
    assert a["sku_conflicts"] == "PRICE_BASE_W_VAT"
    assert items.loc["100002_GD", "sku_conflicts"] == ""
    assert pd.isna(items.loc["100003_SV", "SALES_QTY"])  # left join keeps items without sales


# --------------------------------------------------------------------------- item-level checks
def test_clean_item_only_has_the_size_conflict(checked_items):
    assert issues(checked_items, "100001_BK") == {"SKU_ATTR_CONFLICT"}
    assert checked_items.at["100001_BK", "priority"] == "medium"
    assert bool(checked_items.at["100001_BK", "has_image"])


def test_corrupt_image_and_zero_sales(checked_items):
    assert issues(checked_items, "100002_GD") == {"IMG_UNREADABLE", "SALES_ZERO_QTY"}
    assert checked_items.at["100002_GD", "priority"] == "high"


def test_missing_image_wrong_colour_in_description_and_no_sales(checked_items):
    assert issues(checked_items, "100003_SV") == {"IMG_FILE_MISSING", "CLR_CONFLICT_DESC", "SALES_MISSING"}
    assert checked_items.at["100003_SV", "desc_colour_status"] == "conflict:Gold"
    assert checked_items.at["100003_SV", "priority"] == "high"


def test_image_of_another_model(checked_items):
    assert issues(checked_items, "100004_BK") == {"IMG_REF_MISMATCH"}


def test_wrong_image_path_is_repaired_from_the_unused_file(checked_items):
    e = checked_items.loc["100005_GN"]
    assert e["img_file"] == "100005_GN_1.jpg" and bool(e["has_image"])
    assert issues(checked_items, "100005_GN") == {"IMG_PATH_REPAIRED"}
    assert e["priority"] == "medium"  # a reviewer confirms the photo


def test_image_repair_needs_one_unambiguous_file(toy_dirs):
    items = pd.DataFrame({checks.KEY: ["100001_BK", "100002_GD", "100003_SV"],
                          "PROD_CLR": ["100001_BK", "100002_GD", "100003_SV"],
                          "img_stem": ["100001_1", "100002_1", "100003_SV_1"]})
    files = {s: s + ".jpg" for s in ["100001_BK_1", "100001_BK_2", "100002_GD_1", "100003_SV_1"]}
    items["img_file"] = items["img_stem"].map(files)
    repaired = checks._repair_image_links(items, files)
    assert repaired.tolist() == [False, True, False]  # two candidate files for 100001_BK: left alone
    assert pd.isna(items.at[0, "img_file"])
    assert items.at[1, "img_file"] == "100002_GD_1.jpg"
    assert items.at[2, "img_file"] == "100003_SV_1.jpg"  # already linked: untouched


def test_invalid_path_generic_and_shared_images(toy_dirs):
    items = pd.DataFrame({
        checks.KEY: ["100001_BK", "100001_WT", "100002_GD"],
        "PROG_IMAGE": ["/241/52/100001_1", "/241/52/100001_1", "241-52-100002_GD_1"],
        "CAT_COD": ["52", "52", "52"],
        "CLR_COD": ["_BK", "_WT", "_GD"],
        "PROD_REF": ["100001", "100001", "100002"],
        "refs_all": ["100001", "100001", "100002"],
    })
    items["issues"] = ""
    checks.check_images(items, verify_files=False)
    assert set(items.at[0, "issues"].rstrip(";").split(";")) == {"IMG_FILE_MISSING", "IMG_GENERIC", "IMG_SHARED"}
    assert items.at[0, "img_shared_with"] == "100001_BK|100001_WT"
    assert "IMG_PATH_INVALID" in items.at[2, "issues"]


def test_colour_code_must_always_mean_the_same_name():
    items = pd.DataFrame({
        "PROD_CLR": ["1_BK", "2_BK", "3_BK", "41BK"],
        "PROD_REF": ["1", "2", "3", "4"],
        "CLR_COD": ["_BK", "_BK", "_BK", "1BK"],
        "CLR_DES": ["Black", "Black", "Black", "Navy"],
        "PROD_DES_BASE": ["Bag Black", "Bag Black", "Bag Black", "Bag Navy"],
    })
    items["issues"] = ""
    checks.check_colour_consistency(items)
    assert items["issues"].tolist() == ["", "", "", "CLR_CODE_NAME_INCONSISTENT;"]


def test_type_conflict_is_learned_from_the_data():
    descs = ["Necklace GLDN Gold"] * 30 + ["Necklace GLDN Gold", "Necklace Shoulder Strap Black"]
    families = ["Necklaces"] * 30 + ["Earrings", "Shoulder Strap"]
    items = pd.DataFrame({"PROD_DES_BASE": descs, "GFA_DES_EN": families, "issues": ""})
    checks.check_type_consistency(items)
    flagged = items.index[items["issues"].str.contains("TYPE_CONFLICT_DESC")].tolist()
    assert flagged == [30]  # the strap names its own family, so it is not a conflict


def test_image_audit_flags_are_merged(checked_items, toy_data):
    audit = pd.DataFrame({checks.KEY: ["100001_BK"], "vis_issues": ["VIS_TYPE_MISMATCH"],
                          "vis_note": ["photo looks like a bag"]})
    audit.to_parquet(checks.EMB_DIR / "image_audit.parquet", index=False)
    items, _ = checks.run_checks(verify_files=False)
    items = items.set_index(checks.KEY)
    assert "VIS_TYPE_MISMATCH" in issues(items, "100001_BK")
    assert items.at["100001_BK", "priority"] == "high"
    assert items.at["100001_BK", "vis_note"] == "photo looks like a bag"


def test_write_check_outputs(toy_data):
    items, ctx = checks.run_checks(verify_files=False)
    checks.write_check_outputs(items, ctx)
    assert (checks.PROCESSED_DIR / "items_checked.parquet").exists()
    report = (checks.OUT_DIR / "check_report.md").read_text(encoding="utf-8")
    assert "colourways: 5" in report
    vocab = pd.read_csv(checks.OUT_DIR / "review_vocabulary.csv", encoding="utf-8-sig")
    assert {"CLR_DES", "COMPOSITION (one material)"} <= set(vocab["column"])
    assert "CLR_CONFLICT_DESC" in report
