"""Tests for sotamar.viewer — downsampling + static-HTML CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from click.testing import CliRunner

from sotamar.cli import cli
from sotamar.db import SiteRow
from sotamar.viewer import (
    _zone_from_depth,
    build_overview_deck,
    build_site_deck,
    downsample_bathymetry,
    write_viewer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bathy_tif(
    path: Path,
    array: np.ndarray,
    *,
    origin_easting: float = 500000.0,
    origin_northing: float = 4600100.0,
    resolution: float = 1.0,
) -> Path:
    h, w = array.shape
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": w, "height": h, "count": 1,
        "crs": "EPSG:25831",
        "transform": Affine(resolution, 0, origin_easting,
                            0, -resolution, origin_northing),
        "nodata": -9999.0, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)
    return path


def _site_row(slug: str = "test_site",
              easting: float = 500050.0, northing: float = 4600050.0) -> SiteRow:
    """Minimal SiteRow with WGS84 derived from the UTM centre."""
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(
        "EPSG:25831", "EPSG:4326", always_xy=True,
    ).transform(easting, northing)
    return SiteRow(
        slug=slug, name=slug.replace("_", " ").title(),
        lon=lon, lat=lat,
        easting=easting, northing=northing,
        region="Test Region", character="Synthetic",
        description=None, max_depth=None,
        window_size=100,
        stats={},
        rasters=[],
    )


# ---------------------------------------------------------------------------
# Unit tests — downsampling + zone assignment
# ---------------------------------------------------------------------------

class TestZoneFromDepth:

    @pytest.mark.parametrize("depth,zone", [
        (-5.0, 1), (-17.99, 1),
        (-18.0, 2), (-29.99, 2),
        (-30.0, 3), (-39.99, 3),
        (-40.0, 4), (-80.0, 4),
    ])
    def test_boundary_values(self, depth, zone):
        assert _zone_from_depth(depth) == zone


class TestDownsample:

    def test_returns_grid_size_squared_for_flat_submerged(self, tmp_path):
        """A flat -20 m surface with no nodata → one record per downsampled cell."""
        arr = np.full((100, 100), -20.0, dtype=np.float32)
        tif = _make_bathy_tif(tmp_path / "bathy.tif", arr)

        records = downsample_bathymetry(tif, grid_size=10)
        assert len(records) == 100
        depths = [r["depth"] for r in records]
        assert all(d == pytest.approx(-20.0) for d in depths)
        zones = {r["zone"] for r in records}
        assert zones == {2}  # −20 m → zone 2 (AOWD)

    def test_excludes_emerged_cells(self, tmp_path):
        """Emerged (elev > 0) cells must be dropped from the record list."""
        arr = np.full((100, 100), -15.0, dtype=np.float32)
        arr[:50, :] = 5.0  # top half emerged
        tif = _make_bathy_tif(tmp_path / "bathy.tif", arr)

        records = downsample_bathymetry(tif, grid_size=10)
        # 10x10 grid; top 5 rows average to +5 (emerged, dropped), bottom 5
        # rows average to −15 → 50 records.
        assert len(records) == 50
        assert all(r["depth"] < 0 for r in records)

    def test_excludes_nodata_cells(self, tmp_path):
        arr = np.full((100, 100), -25.0, dtype=np.float32)
        arr[:, :50] = -9999.0  # left half nodata
        tif = _make_bathy_tif(tmp_path / "bathy.tif", arr)

        records = downsample_bathymetry(tif, grid_size=10)
        # Rasterio's average resampling over a block that is entirely nodata
        # yields the nodata sentinel; rows with mixed blocks may yield
        # blended values. We only assert that the fully-nodata half is dropped.
        assert len(records) <= 50
        assert all(r["depth"] > -9000 for r in records)

    def test_zone_assignment_follows_depth(self, tmp_path):
        """Ramp from shallow to deep should populate all four zones."""
        arr = np.zeros((100, 100), dtype=np.float32)
        # Columns 0..99 → depths −5..−60 linearly
        for c in range(100):
            arr[:, c] = -5.0 - 0.55 * c
        tif = _make_bathy_tif(tmp_path / "bathy.tif", arr)

        records = downsample_bathymetry(tif, grid_size=20)
        zones = {r["zone"] for r in records}
        assert zones == {1, 2, 3, 4}

    def test_coordinates_inside_site_bbox(self, tmp_path):
        """Generated lon/lat should fall inside the raster's geographic bounds."""
        arr = np.full((100, 100), -10.0, dtype=np.float32)
        tif = _make_bathy_tif(tmp_path / "bathy.tif", arr)

        records = downsample_bathymetry(tif, grid_size=10)
        lons = [r["lon"] for r in records]
        lats = [r["lat"] for r in records]
        # Origin (500000, 4600100) lies near the UTM 31N central meridian
        # (~3°E) — Catalan coast latitudes.
        assert 1.0 < min(lons) < max(lons) < 4.0
        assert 41.0 < min(lats) < max(lats) < 42.5


# ---------------------------------------------------------------------------
# Deck-building sanity (no HTML rendering)
# ---------------------------------------------------------------------------

class TestBuildDecks:

    def test_overview_deck_has_scatter_layer(self):
        rows = [_site_row("alpha", 510000, 4605000),
                _site_row("beta",  515000, 4610000)]
        deck = build_overview_deck(rows)
        types = [layer.type for layer in deck.layers]
        assert "ScatterplotLayer" in types

    def test_site_deck_has_column_and_text_layers(self, tmp_path):
        arr = np.full((50, 50), -25.0, dtype=np.float32)
        tif = _make_bathy_tif(tmp_path / "bathy.tif", arr)
        records = downsample_bathymetry(tif, grid_size=5)
        deck = build_site_deck(_site_row(), records, cell_metres=20.0)
        types = [layer.type for layer in deck.layers]
        assert "ColumnLayer" in types
        assert "TextLayer" in types


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestViewerCli:

    def test_from_files_writes_overview_and_per_site(self, tmp_path, monkeypatch):
        """With --from-files and one site's bathymetry on disk, generate both
        the overview and that site's 3D HTML."""
        from sotamar.sites import all_sites
        first = all_sites()[0]

        sites_dir = tmp_path / "sites"
        (sites_dir / first.slug).mkdir(parents=True)
        arr = np.full((100, 100), -22.0, dtype=np.float32)
        _make_bathy_tif(
            sites_dir / first.slug / "bathymetry.tif", arr,
            origin_easting=first.easting - 50,
            origin_northing=first.northing + 50,
        )

        output = tmp_path / "viewer"
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, [
            "viewer", "--from-files",
            "--sites-dir", str(sites_dir),
            "-o", str(output),
            "--grid-size", "10",
        ])
        assert result.exit_code == 0, result.output

        assert (output / "index.html").exists()
        assert (output / f"{first.slug}.html").exists()

        # pydeck's HTML embeds a deck.gl container — sanity-check the marker.
        idx_html = (output / "index.html").read_text()
        assert "deck-container" in idx_html or "DeckGL" in idx_html
        # The click-to-navigate handler is appended after the pydeck template.
        assert "deckInstance.setProps" in idx_html
        assert "window.location.href" in idx_html

    def test_db_unreachable_fails_loudly(self, tmp_path, monkeypatch):
        """Without --from-files, a bad DB URL must exit with a hint."""
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, [
            "viewer",
            "--db-url", "postgresql+psycopg://nope:nope@127.0.0.1:1/nodb",
            "-o", str(tmp_path / "viewer"),
        ])
        assert result.exit_code != 0
        assert "could not read from PostGIS" in result.output
        assert "--from-files" in result.output
        assert not (tmp_path / "viewer" / "index.html").exists()
