"""Tests for sotamar.db — unit (no DB) and integration (skip if no DB)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine

from sotamar.db import (
    flatten_stats_for_row,
    load_all_sites,
    raster_bbox_utm_wkt,
    site_bbox_utm_wkt,
    site_point_utm_wkt,
    upsert_dive_site,
    upsert_site_rasters,
    upsert_terrain_stats,
)


# ---------------------------------------------------------------------------
# Unit tests (no DB)
# ---------------------------------------------------------------------------

class TestGeometryBuilders:

    def test_site_point_wkt(self, sample_site):
        assert site_point_utm_wkt(sample_site) == "POINT(500000.0 4600000.0)"

    def test_site_bbox_wkt_closes_ring(self, sample_site):
        wkt = site_bbox_utm_wkt(sample_site)
        # Extract the coordinate list and confirm first == last.
        inner = wkt[wkt.index("((") + 2 : wkt.index("))")]
        pts = [p.strip() for p in inner.split(",")]
        assert pts[0] == pts[-1]
        assert len(pts) == 5

    def test_site_bbox_uses_half_size(self, sample_site):
        # half_size=50 => bounds ±50 around centre.
        wkt = site_bbox_utm_wkt(sample_site)
        assert "499950" in wkt
        assert "500050" in wkt
        assert "4599950" in wkt
        assert "4600050" in wkt

    def test_raster_bbox_utm_from_synthetic(self, synthetic_cog):
        wkt = raster_bbox_utm_wkt(synthetic_cog)
        assert wkt.startswith("POLYGON((")
        # make_test_profile origin (500000, 4600050) with 1 m resolution,
        # 100x100 shape => bottom-right (500100, 4599950).
        assert "500000" in wkt
        assert "500100" in wkt
        assert "4600050" in wkt
        assert "4599950" in wkt

    def test_raster_bbox_raises_on_wrong_crs(self, tmp_path):
        path = tmp_path / "wrong_crs.tif"
        profile: dict[str, Any] = {
            "driver": "GTiff", "dtype": "float32",
            "width": 10, "height": 10, "count": 1,
            "crs": "EPSG:4326",
            "transform": Affine(0.001, 0, 3.0, 0, -0.001, 42.0),
            "nodata": -9999.0,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.zeros((10, 10), dtype=np.float32), 1)
        with pytest.raises(ValueError, match="25831"):
            raster_bbox_utm_wkt(path)


class TestFlattenStats:

    def test_fresh_stats_has_all_keys(self):
        fresh = {
            "nodata_pct": 1.0,
            "depth":     {"min": -60, "max": 0, "mean": -20, "std": 10},
            "slope":     {"min": 0, "max": 80, "mean": 5, "std": 3},
            "bpi_fine":  {"min": -10, "max": 10, "mean": 0, "std": 1},
            "bpi_broad": {"min": -40, "max": 40, "mean": 0, "std": 5},
            "vrm":       {"min": 0, "max": 0.5, "mean": 0.01, "std": 0.02},
            "vrm_pct_above_003": 5.0,
            "depth_zones": {
                "owd_pct": 10, "aowd_pct": 20,
                "deep_pct": 30, "tech_pct": 40,
            },
        }
        row = flatten_stats_for_row(fresh)
        assert row["depth_min"] == -60
        assert row["slope_std"] == 3
        assert row["vrm_pct_above_003"] == 5.0
        assert row["zone_owd_pct"] == 10
        assert row["zone_tech_pct"] == 40

    def test_stale_stats_missing_zones(self):
        stale = {
            "nodata_pct": 1.0,
            "depth": {"min": -60, "max": 0, "mean": -20, "std": 10},
            "slope": {"min": 0, "max": 80, "mean": 5, "std": 3},
        }
        row = flatten_stats_for_row(stale)
        assert row["zone_owd_pct"] is None
        assert row["zone_aowd_pct"] is None
        assert row["zone_deep_pct"] is None
        assert row["zone_tech_pct"] is None
        # Missing metrics yield None, not KeyError
        assert row["bpi_fine_min"] is None
        assert row["vrm_std"] is None

    def test_empty_depth_zones_tolerated(self):
        row = flatten_stats_for_row({"nodata_pct": 0, "depth_zones": {}})
        assert row["zone_owd_pct"] is None


# ---------------------------------------------------------------------------
# Integration tests (require PostGIS running)
# ---------------------------------------------------------------------------

class TestUpserts:
    """Exercise the upsert helpers against a live PostGIS instance."""

    def test_upsert_dive_site_roundtrip(self, clean_db, sample_site):
        """Trigger-derived geom should match ST_Transform of geom_utm."""
        import sqlalchemy
        with clean_db.begin() as conn:
            site_id = upsert_dive_site(conn, sample_site)
        assert site_id > 0
        with clean_db.connect() as conn:
            row = conn.execute(sqlalchemy.text(
                "SELECT slug, ST_X(geom) AS lon, ST_Y(geom) AS lat, "
                "ST_X(geom_utm) AS e, ST_Y(geom_utm) AS n, window_size "
                "FROM dive_sites WHERE id = :i"
            ), {"i": site_id}).one()
        assert row.slug == sample_site.slug
        assert row.e == pytest.approx(sample_site.easting)
        assert row.n == pytest.approx(sample_site.northing)
        assert row.window_size == sample_site.half_size * 2
        # Derived WGS84: just sanity bounds for eastern Spain.
        assert 0.5 < row.lon < 4.0
        assert 41.0 < row.lat < 43.0

    def test_upsert_dive_site_is_idempotent(self, clean_db, sample_site):
        import sqlalchemy
        with clean_db.begin() as conn:
            first_id = upsert_dive_site(conn, sample_site)
        with clean_db.begin() as conn:
            second_id = upsert_dive_site(conn, sample_site)
        assert first_id == second_id
        with clean_db.connect() as conn:
            count = conn.execute(sqlalchemy.text(
                "SELECT count(*) FROM dive_sites WHERE slug = :s"
            ), {"s": sample_site.slug}).scalar_one()
            ts = conn.execute(sqlalchemy.text(
                "SELECT updated_at > created_at AS refreshed "
                "FROM dive_sites WHERE id = :i"
            ), {"i": first_id}).scalar_one()
        assert count == 1
        assert ts is True

    def test_analysis_bbox_matches_site_bounds(self, clean_db, sample_site):
        import sqlalchemy
        with clean_db.begin() as conn:
            site_id = upsert_dive_site(conn, sample_site)
        with clean_db.connect() as conn:
            row = conn.execute(sqlalchemy.text(
                "SELECT ST_XMin(analysis_bbox) AS x0, ST_XMax(analysis_bbox) AS x1, "
                "ST_YMin(analysis_bbox) AS y0, ST_YMax(analysis_bbox) AS y1 "
                "FROM dive_sites WHERE id = :i"
            ), {"i": site_id}).one()
        l, b, r, t = sample_site.bounds
        assert row.x0 == pytest.approx(l)
        assert row.x1 == pytest.approx(r)
        assert row.y0 == pytest.approx(b)
        assert row.y1 == pytest.approx(t)

    def test_upsert_stats_with_missing_zones(self, clean_db, sample_site):
        import sqlalchemy
        stale = {
            "nodata_pct": 2.0,
            "depth": {"min": -30, "max": -5, "mean": -15, "std": 4},
        }
        with clean_db.begin() as conn:
            site_id = upsert_dive_site(conn, sample_site)
            upsert_terrain_stats(conn, site_id, stale)
        with clean_db.connect() as conn:
            row = conn.execute(sqlalchemy.text(
                "SELECT nodata_pct, depth_min, depth_max, "
                "zone_owd_pct, zone_tech_pct "
                "FROM site_terrain_stats WHERE site_id = :i"
            ), {"i": site_id}).one()
        assert row.nodata_pct == pytest.approx(2.0)
        assert row.depth_min == pytest.approx(-30)
        assert row.zone_owd_pct is None
        assert row.zone_tech_pct is None

    def test_upsert_site_rasters(self, clean_db, sample_site, tmp_path,
                                 flat_surface):
        """Write a synthetic slope.tif and confirm registration."""
        import sqlalchemy
        site_dir = tmp_path / sample_site.slug
        site_dir.mkdir()
        elev, _ = flat_surface
        profile: dict[str, Any] = {
            "driver": "GTiff", "dtype": "float32",
            "width": 100, "height": 100, "count": 1,
            "crs": "EPSG:25831",
            "transform": Affine(
                1.0, 0, sample_site.easting - sample_site.half_size,
                0, -1.0, sample_site.northing + sample_site.half_size,
            ),
            "nodata": -9999.0,
        }
        with rasterio.open(site_dir / "slope.tif", "w", **profile) as dst:
            dst.write(elev.astype("float32"), 1)
        with clean_db.begin() as conn:
            site_id = upsert_dive_site(conn, sample_site)
            registered = upsert_site_rasters(conn, site_id, site_dir)
        assert registered == ["slope"]
        with clean_db.connect() as conn:
            row = conn.execute(sqlalchemy.text(
                "SELECT layer_name, file_path FROM site_rasters "
                "WHERE site_id = :i"
            ), {"i": site_id}).one()
        assert row.layer_name == "slope"
        assert row.file_path.endswith("slope.tif")


class TestLoadAllSites:
    """End-to-end loader — uses a temp sites dir with synthetic stats."""

    def test_load_tolerates_missing_stats(self, clean_db, tmp_path):
        """Loader should upsert every registered site even when stats/rasters
        are absent, without raising."""
        summary = load_all_sites(clean_db, sites_dir=tmp_path)
        from sotamar.sites import all_sites
        assert summary.sites == len(all_sites())
        assert summary.stats == 0
        assert summary.rasters == 0
        assert summary.skipped == []

    def test_load_handles_partial_stats_json(self, clean_db, tmp_path):
        """One site has a stale stats.json, others have nothing."""
        from sotamar.sites import all_sites
        first = all_sites()[0]
        site_dir = tmp_path / first.slug
        site_dir.mkdir()
        (site_dir / "stats.json").write_text(json.dumps({
            "nodata_pct": 5.0,
            "depth": {"min": -40, "max": -5, "mean": -20, "std": 8},
        }))
        summary = load_all_sites(clean_db, sites_dir=tmp_path)
        assert summary.stats == 1
        assert first.slug in summary.sites_without_zones
