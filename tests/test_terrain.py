"""Tests for sotamar.terrain: slope, hillshade, BPI, VRM."""

from __future__ import annotations

import numpy as np

from sotamar.terrain import (
    _make_annular_kernel,
    compute_bpi,
    compute_depth_zones,
    compute_hillshade,
    compute_slope,
    compute_vrm,
)


# -- Slope --------------------------------------------------------------------

class TestComputeSlope:
    def test_flat_surface_zero_slope(self, flat_surface):
        elev, mask = flat_surface
        slope = compute_slope(elev, mask)
        valid = slope[~np.isnan(slope)]
        assert np.allclose(valid, 0.0, atol=1e-6)

    def test_45_degree_ramp(self, ramp_surface):
        """A ramp rising 1 m per pixel at 1 m resolution = 45 degrees."""
        elev, mask = ramp_surface
        slope = compute_slope(elev, mask)
        # Interior pixels (away from edges where gradient is different)
        interior = slope[10:-10, 10:-10]
        valid = interior[~np.isnan(interior)]
        assert np.allclose(valid, 45.0, atol=0.5)

    def test_returns_float32(self, flat_surface):
        elev, mask = flat_surface
        slope = compute_slope(elev, mask)
        assert slope.dtype == np.float32

    def test_same_shape_as_input(self, flat_surface):
        elev, mask = flat_surface
        slope = compute_slope(elev, mask)
        assert slope.shape == elev.shape

    def test_nodata_propagated(self, surface_with_nodata):
        elev, mask = surface_with_nodata
        slope = compute_slope(elev, mask)
        # Original nodata pixels remain NaN
        assert np.all(np.isnan(slope[mask]))

    def test_nodata_neighbors_become_nan(self, surface_with_nodata):
        """Gradient propagates NaN one pixel into valid data."""
        elev, mask = surface_with_nodata
        slope = compute_slope(elev, mask)
        # Pixel just inside the border should also be NaN (gradient uses neighbors)
        assert np.isnan(slope[20, 20])

    def test_slope_range(self, peaked_surface):
        elev, mask = peaked_surface
        slope = compute_slope(elev, mask)
        valid = slope[~np.isnan(slope)]
        assert valid.min() >= 0.0
        assert valid.max() <= 90.0

    def test_peaked_surface_has_nonzero_slope(self, peaked_surface):
        elev, mask = peaked_surface
        slope = compute_slope(elev, mask)
        valid = slope[~np.isnan(slope)]
        assert valid.max() > 5.0  # Gaussian peak should have steep flanks


# -- Hillshade ----------------------------------------------------------------

class TestComputeHillshade:
    def test_returns_float32(self, flat_surface):
        elev, mask = flat_surface
        hs = compute_hillshade(elev, mask)
        assert hs.dtype == np.float32

    def test_same_shape_as_input(self, flat_surface):
        elev, mask = flat_surface
        hs = compute_hillshade(elev, mask)
        assert hs.shape == elev.shape

    def test_range_zero_to_one(self, peaked_surface):
        elev, mask = peaked_surface
        hs = compute_hillshade(elev, mask)
        valid = hs[~np.isnan(hs)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_flat_surface_uniform_illumination(self, flat_surface):
        """Flat surface should have uniform hillshade (no shadows)."""
        elev, mask = flat_surface
        hs = compute_hillshade(elev, mask)
        valid = hs[~np.isnan(hs)]
        assert np.std(valid) < 0.01

    def test_nodata_is_nan(self, surface_with_nodata):
        elev, mask = surface_with_nodata
        hs = compute_hillshade(elev, mask)
        assert np.all(np.isnan(hs[mask]))

    def test_custom_parameters(self, peaked_surface):
        elev, mask = peaked_surface
        hs1 = compute_hillshade(elev, mask, azimuth=0, altitude=90)
        hs2 = compute_hillshade(elev, mask, azimuth=180, altitude=30)
        # Different params should produce different results
        assert not np.allclose(hs1[~np.isnan(hs1)], hs2[~np.isnan(hs2)])


# -- Annular kernel -----------------------------------------------------------

class TestMakeAnnularKernel:
    def test_shape(self):
        kernel = _make_annular_kernel(3, 5)
        assert kernel.shape == (11, 11)  # 2*5+1

    def test_center_excluded_when_inner_gt_zero(self):
        kernel = _make_annular_kernel(2, 5)
        center = kernel.shape[0] // 2
        assert kernel[center, center] == False

    def test_center_included_when_inner_zero(self):
        kernel = _make_annular_kernel(0, 3)
        center = kernel.shape[0] // 2
        assert kernel[center, center] == True

    def test_symmetry(self):
        kernel = _make_annular_kernel(3, 5)
        assert np.array_equal(kernel, kernel[::-1, :])  # vertical
        assert np.array_equal(kernel, kernel[:, ::-1])  # horizontal
        assert np.array_equal(kernel, kernel.T)          # diagonal

    def test_boolean_dtype(self):
        kernel = _make_annular_kernel(3, 5)
        assert kernel.dtype == bool

    def test_has_true_values(self):
        kernel = _make_annular_kernel(3, 5)
        assert kernel.sum() > 0

    def test_inner_ring_excluded(self):
        """Pixels closer than inner_radius should be False."""
        kernel = _make_annular_kernel(3, 5)
        center = 5
        # Center and immediate neighbors (dist < 3) should be False
        assert kernel[center, center] == False
        assert kernel[center, center + 1] == False
        assert kernel[center, center + 2] == False


# -- BPI ----------------------------------------------------------------------

class TestComputeBpi:
    def test_flat_surface_zero_bpi(self, flat_surface):
        elev, mask = flat_surface
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        valid = bpi[~np.isnan(bpi)]
        assert np.allclose(valid, 0.0, atol=1e-4)

    def test_returns_float32(self, flat_surface):
        elev, mask = flat_surface
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        assert bpi.dtype == np.float32

    def test_same_shape_as_input(self, flat_surface):
        elev, mask = flat_surface
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        assert bpi.shape == elev.shape

    def test_nodata_is_nan(self, surface_with_nodata):
        elev, mask = surface_with_nodata
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        assert np.all(np.isnan(bpi[mask]))

    def test_peak_has_positive_bpi(self, peaked_surface):
        """A peak should have positive BPI (above annular mean)."""
        elev, mask = peaked_surface
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        center = bpi[50, 50]
        assert not np.isnan(center)
        assert center > 0

    def test_trough_has_negative_bpi(self):
        """A depression should have negative BPI."""
        elev = np.zeros((100, 100), dtype=np.float32)
        elev[50, 50] = -10.0  # deep pit
        mask = np.zeros_like(elev, dtype=bool)
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        assert bpi[50, 50] < 0

    def test_broad_bpi_same_shape(self, peaked_surface):
        elev, mask = peaked_surface
        bpi = compute_bpi(elev, mask, inner_radius=25, outer_radius=50)
        assert bpi.shape == elev.shape

    def test_bpi_near_zero_mean(self, peaked_surface):
        """BPI should have a near-zero mean over the full grid."""
        elev, mask = peaked_surface
        bpi = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
        valid = bpi[~np.isnan(bpi)]
        assert abs(valid.mean()) < 1.0


# -- VRM ----------------------------------------------------------------------

class TestComputeVrm:
    def test_flat_surface_zero_vrm(self, flat_surface):
        elev, mask = flat_surface
        vrm = compute_vrm(elev, mask)
        # Check interior only — edges have boundary effects from uniform_filter
        interior = vrm[5:-5, 5:-5]
        valid = interior[~np.isnan(interior)]
        assert np.allclose(valid, 0.0, atol=1e-6)

    def test_returns_float32(self, flat_surface):
        elev, mask = flat_surface
        vrm = compute_vrm(elev, mask)
        assert vrm.dtype == np.float32

    def test_same_shape_as_input(self, flat_surface):
        elev, mask = flat_surface
        vrm = compute_vrm(elev, mask)
        assert vrm.shape == elev.shape

    def test_nodata_is_nan(self, surface_with_nodata):
        elev, mask = surface_with_nodata
        vrm = compute_vrm(elev, mask)
        assert np.all(np.isnan(vrm[mask]))

    def test_peaked_surface_nonzero_vrm(self, peaked_surface):
        """Terrain with relief should have VRM > 0."""
        elev, mask = peaked_surface
        vrm = compute_vrm(elev, mask)
        valid = vrm[~np.isnan(vrm)]
        assert valid.max() > 0.001

    def test_vrm_range(self, peaked_surface):
        """VRM should be in [0, 1] for valid pixels."""
        elev, mask = peaked_surface
        vrm = compute_vrm(elev, mask)
        valid = vrm[~np.isnan(vrm)]
        assert valid.min() >= -1e-6  # allow tiny float imprecision
        assert valid.max() <= 1.0 + 1e-6

    def test_steep_terrain_high_vrm(self):
        """A surface with random high-frequency relief should have high VRM."""
        rng = np.random.default_rng(42)
        elev = rng.standard_normal((100, 100)).astype(np.float32) * 10.0
        mask = np.zeros_like(elev, dtype=bool)
        vrm = compute_vrm(elev, mask)
        valid = vrm[~np.isnan(vrm)]
        assert valid.mean() > 0.05  # rough terrain has substantial VRM

    def test_custom_window_size(self, peaked_surface):
        elev, mask = peaked_surface
        vrm3 = compute_vrm(elev, mask, window_size=3)
        vrm5 = compute_vrm(elev, mask, window_size=5)
        # Different window sizes give different results
        valid3 = vrm3[30:-30, 30:-30][~np.isnan(vrm3[30:-30, 30:-30])]
        valid5 = vrm5[30:-30, 30:-30][~np.isnan(vrm5[30:-30, 30:-30])]
        assert not np.allclose(valid3, valid5)


# -- Depth zones --------------------------------------------------------------

class TestComputeDepthZones:
    def test_returns_float32(self, flat_surface):
        elev, mask = flat_surface
        zones = compute_depth_zones(elev, mask)
        assert zones.dtype == np.float32

    def test_same_shape_as_input(self, flat_surface):
        elev, mask = flat_surface
        zones = compute_depth_zones(elev, mask)
        assert zones.shape == elev.shape

    def test_zone_assignment(self):
        """Each depth should map to the correct zone."""
        # Depths spanning all zones plus emerged
        elev = np.array([5.0, 0.0, -10.0, -18.0, -25.0, -30.0, -35.0, -40.0, -50.0],
                        dtype=np.float32).reshape(1, -1)
        mask = np.zeros_like(elev, dtype=bool)
        zones = compute_depth_zones(elev, mask)
        # emerged (5.0) -> NaN; 0.0 -> zone 1; -10 -> zone 1; -18 -> zone 2 (boundary, deeper);
        # -25 -> zone 2; -30 -> zone 3 (boundary); -35 -> zone 3; -40 -> zone 4 (boundary); -50 -> zone 4
        expected = np.array([np.nan, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float32)
        np.testing.assert_array_equal(np.isnan(zones[0]), np.isnan(expected))
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(zones[0][valid], expected[valid])

    def test_emerged_is_nan(self):
        """Elevation > 0 should produce NaN."""
        elev = np.array([[1.0, 10.0, 0.1]], dtype=np.float32)
        mask = np.zeros_like(elev, dtype=bool)
        zones = compute_depth_zones(elev, mask)
        assert np.all(np.isnan(zones))

    def test_nodata_is_nan(self):
        """Masked pixels should produce NaN regardless of elevation value."""
        elev = np.array([[-25.0, -25.0, -25.0]], dtype=np.float32)
        mask = np.array([[True, False, True]])
        zones = compute_depth_zones(elev, mask)
        assert np.isnan(zones[0, 0])
        assert zones[0, 1] == 2.0  # -25 -> AOWD
        assert np.isnan(zones[0, 2])

    def test_only_submerged_zones_1_to_4(self, peaked_surface):
        """All non-NaN output values must be in {1, 2, 3, 4}."""
        elev, mask = peaked_surface
        zones = compute_depth_zones(elev, mask)
        valid = zones[~np.isnan(zones)]
        assert set(np.unique(valid).tolist()).issubset({1.0, 2.0, 3.0, 4.0})
