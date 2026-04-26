"""Tests for sotamar.cli: Click commands and end-to-end pipeline."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from affine import Affine
from click.testing import CliRunner

from sotamar.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def small_cog(tmp_path):
    """Create a small synthetic COG for CLI integration tests.

    100x100 raster with a Gaussian peak, origin aligned so that a test site
    with easting=500050, northing=4600000, half_size=50 falls within it.
    """
    path = tmp_path / "test_cog.tif"

    y, x = np.mgrid[0:100, 0:100].astype(np.float32)
    elev = 20.0 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / (2 * 20**2))
    elev -= 25.0  # mostly submerged

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 100,
        "height": 100,
        "count": 1,
        "crs": "EPSG:25831",
        "transform": Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 4600050.0),
        "nodata": -9999.0,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(elev, 1)
    return path


@pytest.fixture
def mock_site():
    """Patch the site registry with a single test site matching small_cog."""
    from sotamar.sites import Site

    test_site = Site(
        slug="test_peak",
        name="Test Peak",
        easting=500050,
        northing=4600000,
        half_size=50,
        region="Test",
        character="Synthetic Gaussian peak",
    )
    return test_site


# -- list command -------------------------------------------------------------

class TestListCommand:
    def test_exit_code(self, runner):
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0

    def test_shows_all_sites(self, runner):
        result = runner.invoke(cli, ["list"])
        assert "med_meda_gran" in result.output
        assert "pal_boreas" in result.output
        assert "cdc_cap_norfeu" in result.output

    def test_shows_header(self, runner):
        result = runner.invoke(cli, ["list"])
        assert "Slug" in result.output
        assert "Name" in result.output
        assert "Region" in result.output

    def test_shows_count(self, runner):
        from sotamar.sites import all_sites
        result = runner.invoke(cli, ["list"])
        assert f"{len(all_sites())} sites registered" in result.output


# -- analyze command: argument validation -------------------------------------

class TestAnalyzeValidation:
    def test_no_args_errors(self, runner):
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0
        assert "Provide a site SLUG or use --all" in result.output

    def test_unknown_slug_errors(self, runner, small_cog):
        result = runner.invoke(cli, ["analyze", "atlantis", "--cog", str(small_cog)])
        assert result.exit_code != 0
        assert "Unknown site" in result.output

    def test_missing_cog_errors(self, runner, tmp_path):
        # chdir to tmp where no COG exists
        result = runner.invoke(
            cli, ["analyze", "pal_boreas", "--cog", str(tmp_path / "nope.tif")]
        )
        assert result.exit_code != 0


# -- analyze command: end-to-end pipeline -------------------------------------

class TestAnalyzePipeline:
    def test_single_site_produces_all_outputs(self, runner, small_cog, mock_site, tmp_path):
        """Full pipeline on a synthetic site produces expected output files."""
        output_base = tmp_path / "output"

        with patch("sotamar.cli.get_site", return_value=mock_site), \
             patch("sotamar.cli.list_sites", return_value=["test_peak"]):
            result = runner.invoke(cli, [
                "analyze", "test_peak",
                "--cog", str(small_cog),
                "--output", str(output_base),
            ])

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        site_dir = output_base / "test_peak"
        assert site_dir.exists()

        # GeoTIFFs
        for name in ("bathymetry", "slope", "hillshade", "bpi_fine", "bpi_broad", "vrm"):
            tif = site_dir / f"{name}.tif"
            assert tif.exists(), f"Missing {name}.tif"
            with rasterio.open(tif) as src:
                assert src.dtypes[0] == "float32"
                assert src.crs is not None

        # Stats JSON
        stats_path = site_dir / "stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        assert "nodata_pct" in stats
        assert "depth" in stats
        assert "slope" in stats
        assert "vrm" in stats

        # Figures
        for fig_name in ("terrain_analysis", "depth_profile"):
            assert (site_dir / f"{fig_name}.png").exists()
            assert (site_dir / f"{fig_name}.pdf").exists()

    def test_analyze_all_flag(self, runner, small_cog, mock_site, tmp_path):
        """--all flag processes all sites."""
        output_base = tmp_path / "all_output"

        with patch("sotamar.cli.all_sites", return_value=[mock_site]):
            result = runner.invoke(cli, [
                "analyze", "--all",
                "--cog", str(small_cog),
                "--output", str(output_base),
            ])

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert (output_base / "test_peak" / "stats.json").exists()

    def test_high_nodata_site_skipped(self, runner, tmp_path):
        """A site with >95% nodata should be skipped without error."""
        # Create a COG that's almost entirely nodata
        cog_path = tmp_path / "empty_cog.tif"
        data = np.full((100, 100), -9999.0, dtype=np.float32)
        data[0, 0] = -20.0  # just 1 pixel of data → 99% nodata

        profile: dict[str, Any] = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": 100,
            "height": 100,
            "count": 1,
            "crs": "EPSG:25831",
            "transform": Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 4600050.0),
            "nodata": -9999.0,
            "compress": "lzw",
        }
        with rasterio.open(cog_path, "w", **profile) as dst:
            dst.write(data, 1)

        from sotamar.sites import Site
        empty_site = Site(
            slug="empty", name="Empty", easting=500050, northing=4600000,
            half_size=50, region="Test", character="Empty",
        )

        output_base = tmp_path / "skip_output"
        with patch("sotamar.cli.get_site", return_value=empty_site), \
             patch("sotamar.cli.list_sites", return_value=["empty"]):
            result = runner.invoke(cli, [
                "analyze", "empty",
                "--cog", str(cog_path),
                "--output", str(output_base),
            ])

        assert result.exit_code == 0
        assert "SKIPPING" in result.output

    def test_output_dir_option(self, runner, small_cog, mock_site, tmp_path):
        """--output overrides the default output directory."""
        custom_dir = tmp_path / "custom"
        with patch("sotamar.cli.get_site", return_value=mock_site), \
             patch("sotamar.cli.list_sites", return_value=["test_peak"]):
            result = runner.invoke(cli, [
                "analyze", "test_peak",
                "--cog", str(small_cog),
                "--output", str(custom_dir),
            ])

        assert result.exit_code == 0
        assert (custom_dir / "test_peak" / "stats.json").exists()

    def test_stats_json_values_reasonable(self, runner, small_cog, mock_site, tmp_path):
        """Check that computed stats have physically reasonable values."""
        output_base = tmp_path / "stats_check"

        with patch("sotamar.cli.get_site", return_value=mock_site), \
             patch("sotamar.cli.list_sites", return_value=["test_peak"]):
            result = runner.invoke(cli, [
                "analyze", "test_peak",
                "--cog", str(small_cog),
                "--output", str(output_base),
            ])

        assert result.exit_code == 0
        stats = json.loads((output_base / "test_peak" / "stats.json").read_text())

        # Depth stats
        assert stats["depth"]["min"] < stats["depth"]["max"]
        assert stats["depth"]["std"] > 0

        # Slope should be non-negative
        assert stats["slope"]["min"] >= 0
        assert stats["slope"]["max"] <= 90

        # VRM should be in [0, 1]
        assert stats["vrm"]["min"] >= -0.001
        assert stats["vrm"]["max"] <= 1.001

    def test_geotiff_crs_preserved(self, runner, small_cog, mock_site, tmp_path):
        """Output GeoTIFFs should have EPSG:25831 CRS."""
        output_base = tmp_path / "crs_check"

        with patch("sotamar.cli.get_site", return_value=mock_site), \
             patch("sotamar.cli.list_sites", return_value=["test_peak"]):
            runner.invoke(cli, [
                "analyze", "test_peak",
                "--cog", str(small_cog),
                "--output", str(output_base),
            ])

        with rasterio.open(output_base / "test_peak" / "bathymetry.tif") as src:
            assert src.crs.to_epsg() == 25831


# -- check-coords command ----------------------------------------------------

class TestCheckCoordsCommand:
    @patch("sotamar.sites.verify_all_coordinates")
    def test_exit_code(self, mock_verify, runner):
        mock_verify.return_value = [
            {"site": "test", "distance_m": 100.0, "geocoded_address": "Test Place"},
        ]
        result = runner.invoke(cli, ["check-coords"])
        assert result.exit_code == 0

    @patch("sotamar.sites.verify_all_coordinates")
    def test_shows_distances(self, mock_verify, runner):
        mock_verify.return_value = [
            {"site": "illes_medes", "distance_m": 37.0, "geocoded_address": "Illes Medes, Girona"},
        ]
        result = runner.invoke(cli, ["check-coords"])
        assert "37" in result.output
        assert "illes_medes" in result.output

    @patch("sotamar.sites.verify_all_coordinates")
    def test_flags_large_distances(self, mock_verify, runner):
        mock_verify.return_value = [
            {"site": "bad_site", "distance_m": 5000.0, "geocoded_address": "Wrong Place"},
        ]
        result = runner.invoke(cli, ["check-coords"])
        assert "!!!" in result.output

    @patch("sotamar.sites.verify_all_coordinates")
    def test_handles_errors(self, mock_verify, runner):
        mock_verify.return_value = [
            {"site": "broken", "error": "Network timeout"},
        ]
        result = runner.invoke(cli, ["check-coords"])
        assert result.exit_code == 0
        assert "ERROR" in result.output


# -- _markers_for_site -------------------------------------------------------

class TestMarkersForSite:
    """Auto-marker integration: Site.markers ∪ POIs in window, deduped."""

    @pytest.fixture
    def site_with_marker(self):
        from sotamar.sites import Site
        return Site(
            slug="test", name="Test", easting=510000, northing=4631400,
            region="R", character="C", half_size=250,
            markers=((510018, 4631388, "Manual"),),
        )

    def test_combines_manual_and_pois(self, site_with_marker):
        from sotamar.cli import _markers_for_site
        from sotamar.pois import POI

        # POI inside the 500 m window, distinct from the manual marker
        in_window = POI(
            id="poi1", name="POI Inside", region="r", municipality=None,
            site_type="rock", latitude=41.83432, longitude=3.12200,
            coord_confidence="verified",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )
        # POI outside the window — should be excluded
        outside = POI(
            id="poi2", name="POI Outside", region="r", municipality=None,
            site_type="rock", latitude=42.5, longitude=3.5,
            coord_confidence="verified",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )

        with patch("sotamar.cli.load_pois", return_value=[in_window, outside]), \
             patch("sotamar.cli.DEFAULT_POIS_CSV") as mock_path:
            mock_path.exists.return_value = True
            markers = _markers_for_site(site_with_marker)

        labels = {label for _, _, label in markers}
        assert "Manual" in labels
        assert "POI Inside" in labels
        assert "POI Outside" not in labels

    def test_dedupes_overlapping_markers(self, site_with_marker):
        """A POI within 10 m of a manual marker should not be drawn twice."""
        from sotamar.cli import _markers_for_site
        from sotamar.pois import POI

        # POI at lat/lon that round-trips to UTM ~510018, 4631388 — same as
        # the site_with_marker manual marker.
        duplicate = POI(
            id="dup", name="Duplicate of Manual",
            region="r", municipality=None, site_type="wreck",
            latitude=41.83432, longitude=3.12065,
            coord_confidence="verified",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )

        with patch("sotamar.cli.load_pois", return_value=[duplicate]), \
             patch("sotamar.cli.DEFAULT_POIS_CSV") as mock_path:
            mock_path.exists.return_value = True
            markers = _markers_for_site(site_with_marker)

        # Only the manual marker survives the dedupe; the POI is dropped.
        assert len(markers) == 1
        assert markers[0][2] == "Manual"

    def test_no_csv_returns_only_site_markers(self, site_with_marker):
        from sotamar.cli import _markers_for_site

        with patch("sotamar.cli.DEFAULT_POIS_CSV") as mock_path:
            mock_path.exists.return_value = False
            markers = _markers_for_site(site_with_marker)

        assert markers == [(510018, 4631388, "Manual")]
