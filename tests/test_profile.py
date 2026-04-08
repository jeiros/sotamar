"""Tests for sotamar.profile: depth transect extraction."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from sotamar.profile import DIVE_THRESHOLDS, extract_depth_profile


# -- Constants ----------------------------------------------------------------

class TestDiveThresholds:
    def test_four_thresholds(self):
        assert len(DIVE_THRESHOLDS) == 4

    def test_all_negative(self):
        for depth, label, color in DIVE_THRESHOLDS:
            assert depth < 0, f"Threshold {label} should be negative"

    def test_ordered_deepest_last(self):
        depths = [d for d, _, _ in DIVE_THRESHOLDS]
        assert depths == sorted(depths, reverse=True)

    def test_has_labels_and_colors(self):
        for depth, label, color in DIVE_THRESHOLDS:
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(color, str) and len(color) > 0


# -- extract_depth_profile ----------------------------------------------------

class TestExtractDepthProfile:
    @pytest.fixture
    def simple_raster(self):
        """100x100 raster with values equal to column index, origin at (1000, 2100)."""
        cols = np.arange(100, dtype=np.float32)
        elev = np.broadcast_to(cols, (100, 100)).copy()
        # 1 m resolution, origin top-left at (1000, 2100)
        transform = Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2100.0)
        return elev, transform

    def test_returns_two_arrays(self, simple_raster):
        elev, transform = simple_raster
        distances, depths = extract_depth_profile(
            elev, transform, (1010, 2050), (1090, 2050), n_points=50,
        )
        assert isinstance(distances, np.ndarray)
        assert isinstance(depths, np.ndarray)

    def test_correct_length(self, simple_raster):
        elev, transform = simple_raster
        distances, depths = extract_depth_profile(
            elev, transform, (1010, 2050), (1090, 2050), n_points=200,
        )
        assert len(distances) == 200
        assert len(depths) == 200

    def test_distance_starts_at_zero(self, simple_raster):
        elev, transform = simple_raster
        distances, _ = extract_depth_profile(
            elev, transform, (1010, 2050), (1090, 2050),
        )
        assert distances[0] == pytest.approx(0.0)

    def test_distance_end_matches_geometry(self, simple_raster):
        """Distance from (1010,2050) to (1090,2050) should be 80 m."""
        elev, transform = simple_raster
        distances, _ = extract_depth_profile(
            elev, transform, (1010, 2050), (1090, 2050),
        )
        assert distances[-1] == pytest.approx(80.0, abs=0.5)

    def test_diagonal_distance(self, simple_raster):
        """Diagonal from (1000,2100) to (1099,2001) should be ~139.7 m."""
        elev, transform = simple_raster
        distances, _ = extract_depth_profile(
            elev, transform, (1000, 2100), (1099, 2001),
        )
        expected = np.sqrt(99**2 + 99**2)
        assert distances[-1] == pytest.approx(expected, abs=1.0)

    def test_ew_transect_reads_increasing_values(self, simple_raster):
        """E-W transect across column-indexed raster should show increasing values."""
        elev, transform = simple_raster
        _, depths = extract_depth_profile(
            elev, transform, (1010, 2050), (1090, 2050), n_points=50,
        )
        valid = depths[~np.isnan(depths)]
        assert len(valid) > 0
        # Values should generally increase (column index increases eastward)
        assert valid[-1] > valid[0]

    def test_out_of_bounds_returns_nan(self, simple_raster):
        """Points outside the raster should be NaN."""
        elev, transform = simple_raster
        # Start well outside the raster
        _, depths = extract_depth_profile(
            elev, transform, (500, 2050), (600, 2050), n_points=20,
        )
        assert np.all(np.isnan(depths))

    def test_partially_out_of_bounds(self, simple_raster):
        """Transect extending beyond raster edge: some NaN, some valid."""
        elev, transform = simple_raster
        # Raster covers x=[1000,1100], request x=[1050,1150]
        _, depths = extract_depth_profile(
            elev, transform, (1050, 2050), (1150, 2050), n_points=100,
        )
        assert np.any(np.isnan(depths))
        assert np.any(~np.isnan(depths))

    def test_nodata_in_raster_propagated(self):
        """NaN values in the elevation array should appear in the profile."""
        elev = np.full((100, 100), -20.0, dtype=np.float32)
        elev[50, 40:60] = np.nan  # NaN band across the transect
        transform = Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 2100.0)

        _, depths = extract_depth_profile(
            elev, transform, (1000, 2050), (1099, 2050), n_points=100,
        )
        assert np.any(np.isnan(depths))

    def test_single_point(self, simple_raster):
        elev, transform = simple_raster
        distances, depths = extract_depth_profile(
            elev, transform, (1050, 2050), (1050, 2050), n_points=1,
        )
        assert len(distances) == 1
        assert distances[0] == 0.0
