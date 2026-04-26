"""Tests for sotamar.viewer — multi-metric static HTML viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from click.testing import CliRunner

from sotamar.cli import cli
from sotamar.db import SiteRow
from sotamar.viewer import (
    METRICS,
    METRICS_BY_SLUG,
    _zone_from_depth,
    build_overview_deck,
    build_records,
    build_site_deck,
    compute_colors,
    downsample_raster,
    write_site_pages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tif(path: Path, array: np.ndarray, *,
              origin_easting: float = 500000.0,
              origin_northing: float = 4600100.0,
              resolution: float = 1.0) -> Path:
    h, w = array.shape
    profile: dict[str, Any] = {
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
              easting: float = 500050.0,
              northing: float = 4600050.0) -> SiteRow:
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(
        "EPSG:25831", "EPSG:4326", always_xy=True,
    ).transform(easting, northing)
    return SiteRow(
        slug=slug, name=slug.replace("_", " ").title(),
        lon=lon, lat=lat, easting=easting, northing=northing,
        region="Test", character="Synthetic",
        description=None, max_depth=None,
        window_size=100, stats={}, rasters=[],
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestMetricsRegistry:

    def test_six_metrics_registered(self):
        assert len(METRICS) == 6
        # Tabs ordered by dive utility: actionable metrics first
        # (depth, zone, slope), broad BPI ahead of fine BPI (walls/
        # terraces > boulder-scale for trip planning), VRM last.
        assert [m.slug for m in METRICS] == [
            "depth", "zone", "slope", "bpi_broad", "bpi_fine", "vrm",
        ]

    def test_lookup_by_slug(self):
        assert METRICS_BY_SLUG["slope"].label == "Slope"
        assert METRICS_BY_SLUG["vrm"].cmap == "inferno"

    def test_every_metric_has_caption(self):
        for m in METRICS:
            assert m.caption, f"{m.slug} missing caption"


class TestZoneFromDepth:

    @pytest.mark.parametrize("depth,zone", [
        (-5.0, 1), (-17.99, 1),
        (-18.0, 2), (-29.99, 2),
        (-30.0, 3), (-39.99, 3),
        (-40.0, 4), (-80.0, 4),
    ])
    def test_boundaries(self, depth, zone):
        assert _zone_from_depth(depth) == zone


class TestDownsample:

    def test_shape_and_nodata_handling(self, tmp_path):
        arr = np.full((100, 100), -20.0, dtype=np.float32)
        arr[:, :40] = -9999.0
        tif = _make_tif(tmp_path / "x.tif", arr)
        w = downsample_raster(tif, grid_size=10)
        assert w.arr.shape == (10, 10)
        # Left 4/10 columns become NaN (fully nodata), rest finite at −20.
        assert np.isnan(w.arr[0, 0])
        assert w.arr[0, 9] == pytest.approx(-20.0)


class TestComputeColorsByMetric:

    def test_depth_colormap_maps_to_rgb(self):
        arr = np.linspace(-60.0, 0.0, 50, dtype=np.float32).reshape(5, 10)
        rgb = compute_colors(arr, METRICS_BY_SLUG["depth"], bathy=arr)
        assert rgb.shape == (5, 10, 3)
        assert rgb.dtype == np.uint8
        # Surface (≈0 m, right-most) and deepest (left-most) should differ.
        assert not np.array_equal(rgb[0, 0], rgb[0, -1])

    def test_zone_colormap_uses_discrete_palette(self):
        # depths that fall in each of the 4 zones
        arr = np.array([[-5, -20, -35, -50]], dtype=np.float32)
        rgb = compute_colors(arr, METRICS_BY_SLUG["zone"], bathy=arr)
        from sotamar.viewer import ZONE_RGB
        assert list(rgb[0, 0]) == ZONE_RGB[0]  # zone 1 (OWD)
        assert list(rgb[0, 1]) == ZONE_RGB[1]  # zone 2 (AOWD)
        assert list(rgb[0, 2]) == ZONE_RGB[2]  # zone 3 (Deep)
        assert list(rgb[0, 3]) == ZONE_RGB[3]  # zone 4 (Tech)

    def test_symmetric_p99_around_zero(self):
        arr = np.linspace(-5.0, 5.0, 100, dtype=np.float32).reshape(10, 10)
        rgb = compute_colors(arr, METRICS_BY_SLUG["bpi_fine"], bathy=arr)
        # RdBu_r: middle value (≈0) should land near white/grey.
        # Extremes of opposite sign should differ from each other.
        assert not np.array_equal(rgb[0, 0], rgb[9, 9])


class TestBuildRecords:

    def test_emerged_cells_dropped(self, tmp_path):
        arr = np.full((50, 50), -15.0, dtype=np.float32)
        arr[:25, :] = 5.0  # top half emerged
        tif = _make_tif(tmp_path / "bathy.tif", arr)
        w = downsample_raster(tif, grid_size=10)
        recs = build_records(w, None, METRICS_BY_SLUG["depth"])
        # 10x10 grid; top 5 rows emerged (dropped), bottom 5 rows present.
        assert len(recs) == 50
        assert all(r["depth"] < 0 for r in recs)

    def test_metric_value_in_record(self, tmp_path):
        bathy = np.full((50, 50), -22.0, dtype=np.float32)
        slope = np.full((50, 50), 14.3, dtype=np.float32)
        bt = _make_tif(tmp_path / "bathy.tif", bathy)
        st = _make_tif(tmp_path / "slope.tif", slope)
        bw = downsample_raster(bt, grid_size=10)
        sw = downsample_raster(st, grid_size=10)
        recs = build_records(bw, sw, METRICS_BY_SLUG["slope"])
        # Each record should carry the slope value for the tooltip.
        assert all("slope" in r for r in recs)
        assert all(r["slope"] != "—" for r in recs)


class TestDeckBuilders:

    def test_overview_has_scatterplot(self):
        rows = [_site_row("a", 510000, 4605000),
                _site_row("b", 515000, 4610000)]
        deck = build_overview_deck(rows)
        assert "ScatterplotLayer" in [l.type for l in deck.layers]

    def test_site_has_gridcell_and_text(self, tmp_path):
        bathy = np.full((50, 50), -22.0, dtype=np.float32)
        bt = _make_tif(tmp_path / "bathy.tif", bathy)
        bw = downsample_raster(bt, grid_size=10)
        recs = build_records(bw, None, METRICS_BY_SLUG["depth"])
        deck = build_site_deck(
            _site_row(), recs, cell_metres=10.0,
            spec=METRICS_BY_SLUG["depth"],
        )
        types = [l.type for l in deck.layers]
        assert "GridCellLayer" in types
        assert "TextLayer" in types


# ---------------------------------------------------------------------------
# End-to-end: write_site_pages
# ---------------------------------------------------------------------------

class TestWriteSitePages:

    @pytest.fixture
    def synthetic_site(self, tmp_path):
        """A site dir with a bathymetry, slope, and BPI fine raster plus
        a terrain_analysis.png fake and a stats.json."""
        site_dir = tmp_path / "sites" / "test_site"
        site_dir.mkdir(parents=True)

        bathy = np.full((50, 50), -22.0, dtype=np.float32)
        slope = np.full((50, 50), 8.0, dtype=np.float32)
        bpi   = np.full((50, 50), 0.3, dtype=np.float32)
        vrm   = np.full((50, 50), 0.02, dtype=np.float32)

        _make_tif(site_dir / "bathymetry.tif", bathy)
        _make_tif(site_dir / "slope.tif", slope)
        _make_tif(site_dir / "bpi_fine.tif", bpi)
        _make_tif(site_dir / "bpi_broad.tif", bpi)
        _make_tif(site_dir / "vrm.tif", vrm)

        (site_dir / "stats.json").write_text(json.dumps({
            "nodata_pct": 0.0,
            "depth":     {"min": -22, "max": -22, "mean": -22, "std": 0},
            "slope":     {"min": 8, "max": 8, "mean": 8, "std": 0},
            "bpi_fine":  {"min": 0.3, "max": 0.3, "mean": 0.3, "std": 0},
            "bpi_broad": {"min": 0.3, "max": 0.3, "mean": 0.3, "std": 0},
            "vrm":       {"min": 0.02, "max": 0.02, "mean": 0.02, "std": 0},
            "depth_zones": {
                "owd_pct": 0, "aowd_pct": 100, "deep_pct": 0, "tech_pct": 0,
            },
        }))

        # 1-pixel fake PNG; enough to test copy behaviour.
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
            b"\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa06\x81\x1f\x00\x00"
            b"\x00\x00IEND\xaeB`\x82"
        )
        (site_dir / "terrain_analysis.png").write_bytes(png_bytes)
        (site_dir / "depth_profile.png").write_bytes(png_bytes)

        return site_dir

    def test_produces_six_html_files(self, synthetic_site, tmp_path):
        row = _site_row("test_site", 500050, 4600050)
        out_dir = tmp_path / "viewer"
        out_dir.mkdir()
        result = write_site_pages(row, synthetic_site, out_dir, grid_size=10)
        assert result is not None
        site_out, cells = result
        assert cells > 0
        for m in METRICS:
            path = site_out / f"{m.slug}.html"
            assert path.exists(), f"missing {m.slug}.html"
            content = path.read_text()
            assert "GridCellLayer" in content
            assert "sotamar-tabs" in content
            assert "sotamar-panels" in content

    def test_active_tab_marked_per_page(self, synthetic_site, tmp_path):
        row = _site_row("test_site", 500050, 4600050)
        out_dir = tmp_path / "viewer"
        out_dir.mkdir()
        write_site_pages(row, synthetic_site, out_dir, grid_size=10)

        for m in METRICS:
            html = (out_dir / "test_site" / f"{m.slug}.html").read_text()
            active_anchor = f'<a href="{m.slug}.html" class="active">'
            assert active_anchor in html, f"{m.slug}.html missing active tab"

    def test_copies_figures(self, synthetic_site, tmp_path):
        row = _site_row("test_site", 500050, 4600050)
        out_dir = tmp_path / "viewer"
        out_dir.mkdir()
        write_site_pages(row, synthetic_site, out_dir, grid_size=10)
        site_out = out_dir / "test_site"
        assert (site_out / "terrain_analysis.png").exists()
        assert (site_out / "depth_profile.png").exists()

    def test_stats_table_contains_numbers(self, synthetic_site, tmp_path):
        row = _site_row("test_site", 500050, 4600050)
        out_dir = tmp_path / "viewer"
        out_dir.mkdir()
        write_site_pages(row, synthetic_site, out_dir, grid_size=10)
        html = (out_dir / "test_site" / "depth.html").read_text()
        assert 'class="stats"' in html
        assert "-22.00 m" in html or "-22 m" in html or "-22.00" in html


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestViewerCli:

    def test_from_files_writes_overview_and_per_site_dirs(
        self, tmp_path, monkeypatch,
    ):
        from sotamar.sites import all_sites
        first = all_sites()[0]

        sites_dir = tmp_path / "sites"
        (sites_dir / first.slug).mkdir(parents=True)
        bathy = np.full((100, 100), -22.0, dtype=np.float32)
        _make_tif(sites_dir / first.slug / "bathymetry.tif", bathy,
                  origin_easting=first.easting - 50,
                  origin_northing=first.northing + 50)

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
        site_dir = output / first.slug
        assert site_dir.is_dir()
        # Only bathymetry was supplied in this fixture → depth + zone tabs
        # only. The other four metrics are skipped gracefully.
        assert (site_dir / "depth.html").exists()
        assert (site_dir / "zone.html").exists()
        assert not (site_dir / "slope.html").exists()

        # Overview still has the click handler pointing at depth.html.
        idx = (output / "index.html").read_text()
        assert "deckInstance.setProps" in idx
        assert f'"href": "{first.slug}/depth.html"' in idx

class TestRegionPages:
    """Tests for write_region_pages — the regional drill-down view."""

    def _make_pois(self, region: str, count: int, spread_m: float):
        from pyproj import Transformer
        rng = np.random.default_rng(42)
        # Centre near Boreas seabed area but place pois on a grid in UTM
        cx, cy = 510000.0, 4631400.0
        rows: list[SiteRow] = []
        for i in range(count):
            angle = 2 * np.pi * i / count
            e = cx + (spread_m / 2) * np.cos(angle)
            n = cy + (spread_m / 2) * np.sin(angle)
            lon, lat = Transformer.from_crs(
                "EPSG:25831", "EPSG:4326", always_xy=True,
            ).transform(e, n)
            rows.append(SiteRow(
                slug=f"{region}_p{i}", name=f"P{i}",
                lon=lon, lat=lat, easting=e, northing=n,
                region=region, character="t",
                description=None, max_depth=None,
                window_size=200, stats={}, rasters=[],
            ))
        return rows

    def test_filters_small_regions(self, tmp_path):
        from sotamar.viewer import write_region_pages
        rows = self._make_pois("two_pois", 2, 200)
        out = write_region_pages(rows, tmp_path, cog_path=None, min_pois=3)
        assert out == []

    def test_filters_oversized_bbox(self, tmp_path):
        from sotamar.viewer import write_region_pages
        rows = self._make_pois("sparse", 5, spread_m=8000)
        out = write_region_pages(
            rows, tmp_path, cog_path=None,
            min_pois=3, max_bbox_m=4000,
        )
        # All 5 fit in a 5-region but bbox > 4 km → skipped
        assert out == []

    def test_generates_page_for_tight_cluster(self, tmp_path):
        """COG must exist; this test reads from data/icgc/."""
        from sotamar.viewer import write_region_pages
        cog = Path("data/icgc/batimetria-v2r1-elevacions-2021-2025.tif")
        if not cog.exists():
            pytest.skip("ICGC COG not present; integration test only")
        rows = self._make_pois("cluster", 4, spread_m=400)
        out = write_region_pages(
            rows, tmp_path, cog_path=cog,
            min_pois=3, max_bbox_m=4000, grid_size=50,
        )
        assert len(out) == 1
        region, html_path, n = out[0]
        assert region == "cluster"
        assert n == 4
        assert html_path.exists()
        # Chrome was injected.
        html = html_path.read_text()
        assert "sotamar-scalebar" in html
        assert "sotamar-region-header" in html
        assert "deckInstance.setProps" in html  # click handler
        # Each pin's href points one level up.
        assert '"href": "../cluster_p0/depth.html"' in html


class TestViewerCliDb:
    def test_db_unreachable_fails_loudly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, [
            "viewer",
            "--db-url", "postgresql+psycopg://nope:nope@127.0.0.1:1/nodb",
            "-o", str(tmp_path / "viewer"),
        ])
        assert result.exit_code != 0
        assert "could not read from PostGIS" in result.output
        assert "--from-files" in result.output
