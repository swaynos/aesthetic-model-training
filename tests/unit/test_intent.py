"""Unit tests for aesthetic_scoring.intent."""
import numpy as np
import pytest
from aesthetic_scoring.intent import (
    load_mask, make_whole_image_mask, make_empty_mask,
    resolve_masks, extract_boundary_ring, tile_grid, tile_positions,
    MASK_POLICIES,
)
from tests.conftest import OSC_HUMAN_MASK


@pytest.fixture
def mask_path():
    return str(OSC_HUMAN_MASK)


def test_make_whole_image_mask():
    m = make_whole_image_mask(16, 32)
    assert m.shape == (16, 32)
    assert m.all()


def test_make_empty_mask():
    m = make_empty_mask(16, 32)
    assert not m.any()


def test_load_mask(mask_path):
    m = load_mask(mask_path, (256, 256))
    assert m.shape == (256, 256)
    assert m.dtype == bool
    assert m.any()


def test_load_mask_has_inside_and_outside(mask_path):
    m = load_mask(mask_path, (256, 256))
    assert m.any() and (~m).any()


def test_resolve_masks_none():
    inside, outside = resolve_masks(64, 64, "none")
    assert inside is None
    assert outside is None


def test_resolve_masks_whole_image():
    inside, outside = resolve_masks(64, 64, "whole_image")
    assert inside.all()
    assert not outside.any()


def test_resolve_masks_custom_intent(mask_path):
    inside, outside = resolve_masks(256, 256, "custom_intent", intent_mask_path=mask_path)
    assert inside is not None
    assert outside is not None
    assert (inside | outside).all()


def test_resolve_masks_invalid_policy():
    with pytest.raises(ValueError, match="mask_policy"):
        resolve_masks(64, 64, "invalid_policy")


def test_resolve_masks_custom_intent_no_path():
    inside, outside = resolve_masks(64, 64, "custom_intent", intent_mask_path=None)
    assert inside is None and outside is None


def test_boundary_ring_shape(mask_path):
    mask = load_mask(mask_path, (256, 256))
    ring = extract_boundary_ring(mask, ring_width=4)
    assert ring.shape == mask.shape
    assert ring.dtype == bool
    assert ring.any()


def test_boundary_ring_empty_mask():
    m = make_empty_mask(64, 64)
    ring = extract_boundary_ring(m, ring_width=4)
    assert not ring.any()


def test_tile_grid_full_coverage():
    arr = np.zeros((128, 128))
    tiles = tile_grid(arr, tile_size=32)
    assert len(tiles) == 16


def test_tile_grid_non_divisible():
    arr = np.zeros((100, 100))
    tiles = tile_grid(arr, tile_size=32)
    assert len(tiles) > 0


def test_tile_positions():
    positions = tile_positions(128, 128, tile_size=64)
    assert len(positions) == 4
    for y0, y1, x0, x1 in positions:
        assert y1 - y0 > 0
        assert x1 - x0 > 0
