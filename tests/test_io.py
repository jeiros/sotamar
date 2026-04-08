"""Tests for sotamar.io: COG resolution, GeoTIFF I/O, statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from sotamar.io import (
    NODATA_GEOTIFF,
    compute_stats,
    find_cog,
    read_bathymetry_window,
    save_geotiff,
    save_stats,
)
from tests.conftest import make_test_profile


# -- find_cog -----------------------------------------------------------------

class TestFindCog:
    def test_explicit_path_exists(self, synthetic_cog):
        result = find_cog(synthetic_cog)
        assert result == synthetic_cog

    def test_explicit_path_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="COG not found"):
            find_cog(tmp_path / "nonexistent.tif")

    def test_none_and_default_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Download from ICGC"):
            find_cog(None)

    def test_accepts_string_path(self, synthetic_cog):
        result = find_cog(str(synthetic_cog))
        assert result.exists()


# -- read_bathymetry_window ---------------------------------------------------

class TestReadBathymetryWindow:
    def test_returns_three_tuple(self, synthetic_cog):
        bounds = (500000, 4599950, 500100, 4600050)
        elev, mask, profile = read_bathymetry_window(bounds, synthetic_cog)
        assert isinstance(elev, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert isinstance(profile, dict)

    def test_elevation_dtype_float32(self, synthetic_cog):
        bounds = (500000, 4599950, 500100, 4600050)
        elev, _, _ = read_bathymetry_window(bounds, synthetic_cog)
        assert elev.dtype == np.float32

    def test_mask_dtype_bool(self, synthetic_cog):
        bounds = (500000, 4599950, 500100, 4600050)
        _, mask, _ = read_bathymetry_window(bounds, synthetic_cog)
        assert mask.dtype == bool

    def test_nodata_becomes_nan(self, tmp_path):
        """Pixels with nodata=-9999 in the file should be NaN in output."""
        path = tmp_path / "nodata_test.tif"
        data = np.full((100, 100), -20.0, dtype=np.float32)
        data[:10, :] = -9999.0  # nodata band

        profile = make_test_profile()
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

        bounds = (500000, 4599950, 500100, 4600050)
        elev, mask, _ = read_bathymetry_window(bounds, path)
        assert mask[:10, :].all()  # top 10 rows should be masked
        assert np.all(np.isnan(elev[:10, :]))

    def test_profile_has_required_keys(self, synthetic_cog):
        bounds = (500000, 4599950, 500100, 4600050)
        _, _, profile = read_bathymetry_window(bounds, synthetic_cog)
        required = {"driver", "dtype", "width", "height", "count", "crs", "transform", "nodata"}
        assert required.issubset(profile.keys())

    def test_profile_nodata_is_minus_9999(self, synthetic_cog):
        bounds = (500000, 4599950, 500100, 4600050)
        _, _, profile = read_bathymetry_window(bounds, synthetic_cog)
        assert profile["nodata"] == -9999.0

    def test_bounds_clipping(self, synthetic_cog):
        """Requesting bounds partially outside the raster should not error."""
        # The synthetic raster covers 500000-500100 E, 4599950-4600050 N
        # Request a window extending beyond the right edge
        bounds = (500050, 4599950, 500200, 4600050)
        elev, mask, _ = read_bathymetry_window(bounds, synthetic_cog)
        assert elev.shape[1] == 50  # clipped to available data

    def test_full_window_read(self, synthetic_cog):
        """Reading the full extent should return the whole raster."""
        bounds = (500000, 4599950, 500100, 4600050)
        elev, mask, _ = read_bathymetry_window(bounds, synthetic_cog)
        assert elev.shape == (100, 100)


# -- save_geotiff -------------------------------------------------------------

class TestSaveGeotiff:
    def test_creates_file(self, tmp_path, flat_surface, test_profile):
        elev, _ = flat_surface
        path = tmp_path / "output" / "test.tif"
        save_geotiff(elev, test_profile, path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path, flat_surface, test_profile):
        elev, _ = flat_surface
        path = tmp_path / "a" / "b" / "c" / "test.tif"
        save_geotiff(elev, test_profile, path)
        assert path.exists()

    def test_nan_becomes_nodata(self, tmp_path, test_profile):
        """NaN values in the array should be written as -9999.0."""
        elev = np.full((100, 100), -20.0, dtype=np.float32)
        elev[0, 0] = np.nan

        path = tmp_path / "nan_test.tif"
        save_geotiff(elev, test_profile, path)

        with rasterio.open(path) as src:
            data = src.read(1)
            assert data[0, 0] == pytest.approx(-9999.0)
            assert src.nodata == -9999.0

    def test_round_trip_values(self, tmp_path, test_profile):
        """Write and re-read should preserve values."""
        elev = np.arange(10000, dtype=np.float32).reshape(100, 100) * -0.1
        path = tmp_path / "roundtrip.tif"
        save_geotiff(elev, test_profile, path)

        with rasterio.open(path) as src:
            data = src.read(1)
        np.testing.assert_allclose(data, elev, atol=1e-5)

    def test_output_is_lzw_compressed(self, tmp_path, flat_surface, test_profile):
        elev, _ = flat_surface
        path = tmp_path / "compressed.tif"
        save_geotiff(elev, test_profile, path)
        with rasterio.open(path) as src:
            assert src.compression == rasterio.enums.Compression.lzw

    def test_output_is_float32(self, tmp_path, test_profile):
        elev = np.zeros((100, 100), dtype=np.float64)  # input is float64
        path = tmp_path / "dtype_test.tif"
        save_geotiff(elev, test_profile, path)
        with rasterio.open(path) as src:
            assert src.dtypes[0] == "float32"


# -- compute_stats ------------------------------------------------------------

class TestComputeStats:
    def test_basic_stats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32).reshape(1, 5)
        mask = np.zeros_like(arr, dtype=bool)
        stats = compute_stats({"test": arr}, mask)

        assert stats["test"]["min"] == 1.0
        assert stats["test"]["max"] == 5.0
        assert stats["test"]["mean"] == 3.0
        assert stats["nodata_pct"] == 0.0

    def test_nodata_percentage(self):
        arr = np.full((10, 10), -20.0, dtype=np.float32)
        mask = np.zeros_like(arr, dtype=bool)
        mask[:5, :] = True
        arr[mask] = np.nan

        stats = compute_stats({"depth": arr}, mask)
        assert stats["nodata_pct"] == 50.0

    def test_vrm_threshold_metric(self):
        vrm = np.array([0.001, 0.002, 0.004, 0.005, 0.010], dtype=np.float32).reshape(1, 5)
        mask = np.zeros_like(vrm, dtype=bool)
        stats = compute_stats({"vrm": vrm}, mask)

        assert "vrm_pct_above_003" in stats
        # 3 out of 5 values > 0.003
        assert stats["vrm_pct_above_003"] == 60.0

    def test_no_vrm_key_no_threshold(self):
        arr = np.ones((5, 5), dtype=np.float32)
        mask = np.zeros_like(arr, dtype=bool)
        stats = compute_stats({"slope": arr}, mask)
        assert "vrm_pct_above_003" not in stats

    def test_all_nodata(self):
        arr = np.full((5, 5), np.nan, dtype=np.float32)
        mask = np.ones_like(arr, dtype=bool)
        stats = compute_stats({"depth": arr}, mask)
        assert stats["depth"]["min"] is None

    def test_multiple_arrays(self):
        depth = np.full((5, 5), -10.0, dtype=np.float32)
        slope = np.full((5, 5), 5.0, dtype=np.float32)
        mask = np.zeros((5, 5), dtype=bool)
        stats = compute_stats({"depth": depth, "slope": slope}, mask)
        assert "depth" in stats
        assert "slope" in stats
        assert stats["depth"]["mean"] == -10.0
        assert stats["slope"]["mean"] == 5.0


# -- save_stats ---------------------------------------------------------------

class TestSaveStats:
    def test_creates_json_file(self, tmp_path):
        stats = {"nodata_pct": 10.0, "depth": {"min": -50.0, "max": 0.0}}
        path = tmp_path / "stats.json"
        save_stats(stats, path)
        assert path.exists()

    def test_valid_json(self, tmp_path):
        stats = {"slug": "test", "nodata_pct": 5.0}
        path = tmp_path / "stats.json"
        save_stats(stats, path)
        loaded = json.loads(path.read_text())
        assert loaded == stats

    def test_creates_parent_dirs(self, tmp_path):
        stats = {"test": True}
        path = tmp_path / "deep" / "nested" / "stats.json"
        save_stats(stats, path)
        assert path.exists()
