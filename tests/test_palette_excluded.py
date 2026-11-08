"""The palette exclusion list: read from a CSV, empty when the file is missing."""
from __future__ import annotations

import config


def test_palette_excluded_reads_keys(tmp_path, monkeypatch):
    path = tmp_path / "palette_excluded.csv"
    path.write_text("PROD_CLR_EQUIV,reason\n167657_LB,not a packshot\n166363_PK,other item in frame\n",
                    encoding="utf-8-sig")
    monkeypatch.setattr(config, "PALETTE_EXCLUDED_FILE", path)
    assert config.palette_excluded() == {"167657_LB", "166363_PK"}


def test_palette_excluded_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PALETTE_EXCLUDED_FILE", tmp_path / "none.csv")
    assert config.palette_excluded() == set()
