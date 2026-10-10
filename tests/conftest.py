"""Shared fixtures: a toy copy of the dataset in a temporary folder.

Every path constant of phase1_checks, phase1_clean and review_app is redirected to tmp_path, so the
tests never read or write the real data/ and outputs/ folders.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import phase1_clean as p1
import phase1_checks as checks


def sku(ref: str, clr_cod: str, clr_des: str, size: str, spk: int, *, desc: str | None = None,
        gfa: str = "Earrings", price: float = 9.99, comp: str = "Zinc", img: str | None = None) -> dict:
    """One df_product row (SKU) with the columns the Phase 1 checks use."""
    clr = f"{ref}{clr_cod}"
    return {
        "PROD_SPK": spk, "PROD_COD": f"{clr}{size}", "PROD_COD_EQUIV": f"{clr}{size}",
        "PROD_CLR": clr, "PROD_CLR_EQUIV": clr, "PROD_REF": ref, "PROD_REF_EQUIV": ref,
        "BAR_COD": f"560000000{spk:04d}", "PROD_DES": f"{desc or 'Earring ESSENTIALS ' + clr_des} {size}",
        "SZ_DES": size, "CLR_COD": clr_cod, "CLR_DES": clr_des, "CAT_COD": "52", "CAT_DES_EN": "Jewellery",
        "GFA_DES_EN": gfa, "GFS_DES_EN": "Plain", "COMPOSITION": comp, "FINISHING": "Golden", "MATERIAL": None,
        "PRICE_BASE_W_VAT": price, "PROG_IMAGE": img or f"/241/52/{clr}_1",
    }


# A: three sizes, one with another price (SKU conflict); materials listed in different orders.
# B: corrupt image file and zero sales.
# C: description names another colour, no image file, no sales row.
# D: image file name belongs to another model.
TOY_PRODUCTS = [
    sku("100001", "_BK", "Black", "S", 1, comp=" Zinc  Enamel", gfa="Earrings\xa0"),
    sku("100001", "_BK", "Black", "M", 2, comp="Enamel  Zinc"),
    sku("100001", "_BK", "Black", "L", 3, comp="Zinc  Enamel", price=12.99),
    sku("100002", "_GD", "Gold", "U", 4, desc="Necklace ESSENTIALS Gold", gfa="Necklaces"),
    sku("100003", "_SV", "Silver", "U", 5, desc="Ring ESSENTIALS Gold", gfa="Rings"),
    sku("100004", "_BK", "Black", "U", 6, img="/241/52/100009_BK_1"),
]
TOY_SALES = [("100001_BK", 100.0, 999.0), ("100002_GD", 0.0, 10.0), ("100004_BK", 50.0, 499.5)]
TOY_IMAGES = {"100001_BK_1.jpg": (20, 20, 20), "100004_BK_1.jpg": (200, 30, 30), "100009_BK_1.jpg": (30, 30, 200)}


@pytest.fixture
def toy_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty data/output folders under tmp_path, with phase1_checks pointed at them."""
    dirs = {
        "CSV_DIR": tmp_path / "data" / "csv",
        "IMG_DIR": tmp_path / "data" / "images",
        "PROCESSED_DIR": tmp_path / "data" / "processed",
        "EMB_DIR": tmp_path / "data" / "embeddings",
        "OUT_DIR": tmp_path / "outputs" / "phase1",
        "BATCH_DIR": tmp_path / "outputs" / "phase1" / "batches",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(checks, name, path)
    monkeypatch.setattr(p1, "PROCESSED_DIR", dirs["PROCESSED_DIR"])
    monkeypatch.setattr(p1, "REPORT_DIR", dirs["OUT_DIR"])
    return tmp_path


@pytest.fixture
def toy_data(toy_dirs: Path) -> Path:
    """The toy dataset above written as CSVs and image files."""
    pd.DataFrame(TOY_PRODUCTS).to_csv(checks.CSV_DIR / "df_product.csv", index=False, encoding="utf-8")
    pd.DataFrame(TOY_SALES, columns=["PROD_CLR_EQUIV", "SALES_QTY", "SALES_AMT_FX_RATE"]).to_csv(
        checks.CSV_DIR / "df_sales.csv", index=False, encoding="utf-8")
    for name, colour in TOY_IMAGES.items():
        Image.new("RGB", (16, 16), colour).save(checks.IMG_DIR / name)
    (checks.IMG_DIR / "100002_GD_1.jpg").write_bytes(b"not an image")
    return toy_dirs


@pytest.fixture
def checked_items(toy_data: Path) -> pd.DataFrame:
    items, _ = checks.run_checks(verify_files=True)
    return items.set_index(checks.KEY)
