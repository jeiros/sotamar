"""Tests for sotamar.wreck_detect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine

from sotamar.wreck_detect import detect_wrecks_near, plot_wreck_candidates


@pytest.fixture
def synthetic_cog_with_wreck(tmp_path):
    """A 600 × 600 m raster with flat seabed at -25 m and a 30 × 8 m
    bump rising 5 m above seabed at the centre."""
    path = tmp_path / "synthetic.tif"
    H = W = 600
    elev = np.full((H, W), -25.0, dtype=np.float32)
    # Add a 30 × 8 m wreck-like bump centred at (300, 300)
    for r in range(285, 315):
        for c in range(296, 304):
            # Gaussian peak +5 m at centre
            dr = r - 300
            dc = c - 300
            elev[r, c] = -25.0 + 5.0 * np.exp(-((dr/12)**2 + (dc/3)**2))

    # UTM origin chosen to put centre near 41.83 / 3.12 (Palamós area)
    profile: dict[str, Any] = {
        "driver": "GTiff", "dtype": "float32",
        "width": W, "height": H, "count": 1,
        "crs": "EPSG:25831",
        "transform": Affine(1.0, 0.0, 510000.0 - W/2,
                            0.0, -1.0, 4631400.0 + H/2),
        "nodata": -9999.0, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elev, 1)
    return path


@pytest.fixture
def synthetic_empty_cog(tmp_path):
    """A 200 × 200 raster of pure NoData."""
    path = tmp_path / "empty.tif"
    elev = np.full((200, 200), -9999.0, dtype=np.float32)
    profile: dict[str, Any] = {
        "driver": "GTiff", "dtype": "float32",
        "width": 200, "height": 200, "count": 1,
        "crs": "EPSG:25831",
        "transform": Affine(1.0, 0.0, 509900.0,
                            0.0, -1.0, 4631500.0),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elev, 1)
    return path


# -- detect_wrecks_near ------------------------------------------------------

class TestDetectWrecks:
    def test_finds_synthetic_blob(self, synthetic_cog_with_wreck):
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)
        seed_lon, seed_lat = t.transform(510000.0, 4631400.0)

        candidates, debug = detect_wrecks_near(
            seed_lat=seed_lat, seed_lon=seed_lon,
            cog_path=synthetic_cog_with_wreck,
            search_radius_m=200,
            source_poi_id="synthetic",
        )
        assert candidates, "expected at least one candidate"
        # Top candidate should be near the centre with a meaningful peak
        top = candidates[0]
        assert top.peak_residual_m > 1.0
        assert top.footprint_m2 >= 10
        assert top.rank == 1

    def test_empty_window_returns_empty(self, synthetic_empty_cog):
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)
        seed_lon, seed_lat = t.transform(510000.0, 4631400.0)

        candidates, _ = detect_wrecks_near(
            seed_lat=seed_lat, seed_lon=seed_lon,
            cog_path=synthetic_empty_cog,
            search_radius_m=80,
        )
        assert candidates == []

    def test_ranking_is_descending(self, synthetic_cog_with_wreck):
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)
        seed_lon, seed_lat = t.transform(510000.0, 4631400.0)

        candidates, _ = detect_wrecks_near(
            seed_lat=seed_lat, seed_lon=seed_lon,
            cog_path=synthetic_cog_with_wreck,
            search_radius_m=200,
        )
        if len(candidates) >= 2:
            for a, b in zip(candidates, candidates[1:]):
                assert a.plausibility >= b.plausibility
                assert a.rank < b.rank


# -- plot_wreck_candidates ---------------------------------------------------

class TestPlot:
    def test_writes_png(self, synthetic_cog_with_wreck, tmp_path):
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)
        seed_lon, seed_lat = t.transform(510000.0, 4631400.0)
        candidates, debug = detect_wrecks_near(
            seed_lat=seed_lat, seed_lon=seed_lon,
            cog_path=synthetic_cog_with_wreck,
            search_radius_m=200,
        )
        out = tmp_path / "subdir" / "wreck.png"
        plot_wreck_candidates(
            candidates, debug, out, source_name="Synthetic",
            seed_lat=seed_lat, seed_lon=seed_lon, radius_m=200,
        )
        assert out.exists()
        assert out.stat().st_size > 1000
