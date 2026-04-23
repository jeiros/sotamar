"""Tests for sotamar.figures: terrain analysis and depth profile figures."""

from __future__ import annotations

import numpy as np
import pytest

from sotamar.figures import (
    _make_extent,
    _save_figure,
    plot_depth_profile,
    plot_terrain_analysis,
)


# -- _make_extent -------------------------------------------------------------

class TestMakeExtent:
    def test_order(self):
        result = _make_extent((100, 200, 300, 400))
        assert result == (100, 300, 200, 400)  # (left, right, bottom, top)

    def test_returns_tuple(self):
        result = _make_extent((0, 0, 1, 1))
        assert isinstance(result, tuple)


# -- _save_figure -------------------------------------------------------------

class TestSaveFigure:
    def test_creates_png_and_pdf(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        _save_figure(fig, "test_fig", tmp_path)
        plt.close(fig)

        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()

    def test_creates_output_dir(self, tmp_path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "nested" / "dir"
        _save_figure(fig, "test", out)
        plt.close(fig)
        assert (out / "test.png").exists()


# -- plot_terrain_analysis ----------------------------------------------------

class TestPlotTerrainAnalysis:
    @pytest.fixture
    def synthetic_metrics(self):
        """Generate synthetic metric arrays for figure tests."""
        rng = np.random.default_rng(42)
        shape = (50, 50)
        elev = rng.uniform(-40, 5, shape).astype(np.float32)
        slope = rng.uniform(0, 45, shape).astype(np.float32)
        bpi_fine = rng.uniform(-5, 5, shape).astype(np.float32)
        bpi_broad = rng.uniform(-10, 10, shape).astype(np.float32)
        vrm = rng.uniform(0, 0.1, shape).astype(np.float32)
        zones = rng.integers(1, 5, shape).astype(np.float32)
        bounds = (500000, 4599950, 500050, 4600000)
        return elev, slope, bpi_fine, bpi_broad, vrm, zones, bounds

    def test_creates_png_and_pdf(self, tmp_path, synthetic_metrics):
        elev, slope, bpi_fine, bpi_broad, vrm, zones, bounds = synthetic_metrics
        out = tmp_path / "site_output"
        plot_terrain_analysis(elev, slope, bpi_fine, bpi_broad, vrm, zones, bounds, "Test", out)

        assert (out / "terrain_analysis.png").exists()
        assert (out / "terrain_analysis.pdf").exists()

    def test_png_not_empty(self, tmp_path, synthetic_metrics):
        elev, slope, bpi_fine, bpi_broad, vrm, zones, bounds = synthetic_metrics
        out = tmp_path / "site_output"
        plot_terrain_analysis(elev, slope, bpi_fine, bpi_broad, vrm, zones, bounds, "Test", out)

        assert (out / "terrain_analysis.png").stat().st_size > 1000

    def test_handles_nan_values(self, tmp_path):
        """Should not crash when arrays contain NaN."""
        shape = (50, 50)
        elev = np.full(shape, np.nan, dtype=np.float32)
        elev[10:40, 10:40] = -20.0
        slope = np.full(shape, np.nan, dtype=np.float32)
        slope[10:40, 10:40] = 5.0
        bpi_fine = np.full(shape, np.nan, dtype=np.float32)
        bpi_fine[10:40, 10:40] = 0.0
        bpi_broad = np.full(shape, np.nan, dtype=np.float32)
        bpi_broad[10:40, 10:40] = 0.0
        vrm = np.full(shape, np.nan, dtype=np.float32)
        vrm[10:40, 10:40] = 0.001
        zones = np.full(shape, np.nan, dtype=np.float32)
        zones[10:40, 10:40] = 3.0

        bounds = (0, 0, 50, 50)
        out = tmp_path / "nan_test"
        plot_terrain_analysis(elev, slope, bpi_fine, bpi_broad, vrm, zones, bounds, "NaN", out)
        assert (out / "terrain_analysis.png").exists()


# -- plot_depth_profile -------------------------------------------------------

class TestPlotDepthProfile:
    def test_creates_png_and_pdf(self, tmp_path):
        distances = np.linspace(0, 1000, 200)
        depths = np.sin(distances / 100) * -20 - 10
        out = tmp_path / "profile_output"
        plot_depth_profile(distances, depths, "Test Profile", out)

        assert (out / "depth_profile.png").exists()
        assert (out / "depth_profile.pdf").exists()

    def test_png_not_empty(self, tmp_path):
        distances = np.linspace(0, 500, 100)
        depths = np.linspace(-40, 0, 100)
        out = tmp_path / "profile"
        plot_depth_profile(distances, depths, "Test", out)
        assert (out / "depth_profile.png").stat().st_size > 1000

    def test_handles_all_nan_depths(self, tmp_path):
        """Should not crash with entirely NaN depths."""
        distances = np.linspace(0, 100, 50)
        depths = np.full(50, np.nan)
        out = tmp_path / "nan_profile"
        plot_depth_profile(distances, depths, "NaN", out)
        assert (out / "depth_profile.png").exists()

    def test_handles_partial_nan(self, tmp_path):
        distances = np.linspace(0, 200, 100)
        depths = np.linspace(-30, 0, 100).astype(np.float64)
        depths[:20] = np.nan
        out = tmp_path / "partial_nan"
        plot_depth_profile(distances, depths, "Partial", out)
        assert (out / "depth_profile.png").exists()
