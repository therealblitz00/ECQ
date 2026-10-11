"""Phase 2b colour palettes (src/phase2b_palettes.py) on synthetic images."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import phase2b_palettes as pb


def test_rgb_to_lab_reference_points():
    lab = pb.rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.1)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.1)
    assert lab[2, 1] > 70  # red has a large positive a*


def test_product_pixels_drop_the_white_background_and_keep_white_products():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    img.paste((200, 30, 40), (30, 30, 70, 70))
    px = pb.product_pixels(img, [0.0, 0.0, 1.0, 1.0])
    assert len(px) and (px.min(1) < pb.WHITE).all()
    white_shirt = Image.new("RGB", (50, 50), (250, 250, 250))
    assert len(pb.product_pixels(white_shirt, None)) == 50 * 50


def test_palette_finds_both_colours_with_their_shares_and_is_deterministic():
    px = np.array([[200, 30, 40]] * 300 + [[25, 25, 25]] * 100, dtype=np.uint8)
    pal = pb.palette(px)
    assert [e["name"] for e in pal] == ["red", "black"]
    assert [e["share"] for e in pal] == pytest.approx([0.75, 0.25])
    assert pb.palette(px) == pal


def test_palette_vector_similarity_and_shared_colours():
    red_black = pb.palette_vector([{"name": "red", "share": 0.75}, {"name": "black", "share": 0.25}])
    red = pb.palette_vector([{"name": "red", "share": 1.0}])
    blue = pb.palette_vector([{"name": "blue", "share": 1.0}])
    assert np.linalg.norm(red_black) == pytest.approx(1)
    assert red @ blue == 0 and red_black @ red > 0.9
    assert set(pb.shared_colours(red_black, red)) == {"red"}
