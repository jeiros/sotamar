"""PostGIS loader: Core tables, geometry builders, upserts, GeoJSON read."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rasterio
from geoalchemy2 import Geometry
from pyproj import Transformer
from sqlalchemy import (
    Column,
    Engine,
    ForeignKey,
    Integer,
    MetaData,
    REAL as Real,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from sotamar.sites import Site, all_sites

log = logging.getLogger(__name__)

DEFAULT_DB_URL = "postgresql+psycopg://sotamar:sotamar@localhost:5432/sotamar"

LAYER_NAMES = (
    "bathymetry", "slope", "hillshade", "bpi_fine",
    "bpi_broad", "vrm", "depth_zones",
)

METRICS = ("depth", "slope", "bpi_fine", "bpi_broad", "vrm")
ZONE_KEYS = ("owd_pct", "aowd_pct", "deep_pct", "tech_pct")


# -- Engine -------------------------------------------------------------------

def get_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine. Uses SOTAMAR_DB_URL env var if url is None."""
    resolved = url or os.environ.get("SOTAMAR_DB_URL", DEFAULT_DB_URL)
    return create_engine(resolved, future=True)


# -- Schema declaration (Core) ------------------------------------------------
# spatial_index=False on every Geometry column because sql/schema.sql already
# declares GIST indexes. Prevents metadata.create_all() from duplicating them.

metadata = MetaData()

dive_sites = Table(
    "dive_sites", metadata,
    Column("id", Integer, primary_key=True),
    Column("slug", Text, nullable=False, unique=True),
    Column("name", Text, nullable=False),
    Column("geom", Geometry("POINT", srid=4326, spatial_index=False)),
    Column("geom_utm", Geometry("POINT", srid=25831, spatial_index=False),
           nullable=False),
    Column("region", Text, nullable=False),
    Column("character", Text, nullable=False),
    Column("description", Text),
    Column("max_depth", Real),
    Column("window_size", Integer, nullable=False),
    Column("analysis_bbox",
           Geometry("POLYGON", srid=25831, spatial_index=False),
           nullable=False),
)

site_terrain_stats = Table(
    "site_terrain_stats", metadata,
    Column("id", Integer, primary_key=True),
    Column("site_id", Integer,
           ForeignKey("dive_sites.id", ondelete="CASCADE"),
           nullable=False, unique=True),
    Column("nodata_pct", Real),
    *[Column(f"{m}_{s}", Real)
      for m in METRICS for s in ("min", "max", "mean", "std")],
    Column("vrm_pct_above_003", Real),
    Column("zone_owd_pct", Real),
    Column("zone_aowd_pct", Real),
    Column("zone_deep_pct", Real),
    Column("zone_tech_pct", Real),
)

site_rasters = Table(
    "site_rasters", metadata,
    Column("id", Integer, primary_key=True),
    Column("site_id", Integer,
           ForeignKey("dive_sites.id", ondelete="CASCADE"),
           nullable=False),
    Column("layer_name", Text, nullable=False),
    Column("file_path", Text, nullable=False),
    Column("bbox",
           Geometry("POLYGON", srid=25831, spatial_index=False),
           nullable=False),
    UniqueConstraint("site_id", "layer_name"),
)


# -- Geometry builders (pure) -------------------------------------------------

def site_point_utm_wkt(site: Site) -> str:
    return f"POINT({site.easting} {site.northing})"


def site_bbox_utm_wkt(site: Site) -> str:
    left, bottom, right, top = site.bounds
    return (
        f"POLYGON(({left} {bottom}, {right} {bottom}, "
        f"{right} {top}, {left} {top}, {left} {bottom}))"
    )


def raster_bbox_utm_wkt(tif_path: Path) -> str:
    """Read raster bounds and return a WKT polygon in EPSG:25831.

    Raises ValueError if the raster is not EPSG:25831.
    """
    with rasterio.open(tif_path) as src:
        epsg = src.crs.to_epsg() if src.crs else None
        if epsg != 25831:
            raise ValueError(
                f"{tif_path}: expected EPSG:25831, got {src.crs}"
            )
        b = src.bounds
    return (
        f"POLYGON(({b.left} {b.bottom}, {b.right} {b.bottom}, "
        f"{b.right} {b.top}, {b.left} {b.top}, {b.left} {b.bottom}))"
    )


# -- Stats flattening ---------------------------------------------------------

def flatten_stats_for_row(stats: dict) -> dict:
    """Convert nested stats.json shape to flat column values.

    Tolerates stale files: missing metric keys or missing depth_zones yield
    None for the corresponding columns.
    """
    row: dict = {"nodata_pct": stats.get("nodata_pct")}
    for metric in METRICS:
        sub = stats.get(metric) or {}
        for s in ("min", "max", "mean", "std"):
            row[f"{metric}_{s}"] = sub.get(s)
    row["vrm_pct_above_003"] = stats.get("vrm_pct_above_003")
    zones = stats.get("depth_zones") or {}
    for k in ZONE_KEYS:
        row[f"zone_{k}"] = zones.get(k)
    return row


# -- Upserts ------------------------------------------------------------------

def upsert_dive_site(conn, site: Site) -> int:
    """Insert or update by slug. Returns the site id."""
    values = dict(
        slug=site.slug,
        name=site.name,
        geom_utm=func.ST_GeomFromText(site_point_utm_wkt(site), 25831),
        region=site.region,
        character=site.character,
        description=site.description,
        max_depth=site.max_depth,
        window_size=site.half_size * 2,
        analysis_bbox=func.ST_GeomFromText(site_bbox_utm_wkt(site), 25831),
    )
    stmt = pg_insert(dive_sites).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["slug"],
        set_={
            c: stmt.excluded[c] for c in (
                "name", "geom_utm", "region", "character",
                "description", "max_depth", "window_size", "analysis_bbox",
            )
        },
    ).returning(dive_sites.c.id)
    return conn.execute(stmt).scalar_one()


def upsert_terrain_stats(conn, site_id: int, stats: dict) -> None:
    """Insert or update stats row for site_id."""
    row = flatten_stats_for_row(stats)
    row["site_id"] = site_id
    stmt = pg_insert(site_terrain_stats).values(**row)
    stmt = stmt.on_conflict_do_update(
        index_elements=["site_id"],
        set_={k: stmt.excluded[k] for k in row if k != "site_id"},
    )
    conn.execute(stmt)


def upsert_site_rasters(conn, site_id: int, site_dir: Path) -> list[str]:
    """Register the seven canonical raster layers present on disk.

    Returns the list of layer names successfully upserted.
    """
    registered: list[str] = []
    for layer in LAYER_NAMES:
        tif = site_dir / f"{layer}.tif"
        if not tif.exists():
            log.warning("Missing raster %s for site_id=%d", tif, site_id)
            continue
        try:
            bbox_wkt = raster_bbox_utm_wkt(tif)
        except Exception as exc:
            log.warning("Skipping %s: %s", tif, exc)
            continue
        stmt = pg_insert(site_rasters).values(
            site_id=site_id,
            layer_name=layer,
            file_path=f"{site_dir.name}/{tif.name}",
            bbox=func.ST_GeomFromText(bbox_wkt, 25831),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["site_id", "layer_name"],
            set_={
                "file_path": stmt.excluded.file_path,
                "bbox": stmt.excluded.bbox,
            },
        )
        conn.execute(stmt)
        registered.append(layer)
    return registered


# -- Orchestration ------------------------------------------------------------

@dataclass
class LoadSummary:
    sites: int = 0
    stats: int = 0
    rasters: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    sites_without_zones: list[str] = field(default_factory=list)


def load_all_sites(
    engine: Engine,
    sites_dir: Path = Path("data/sites"),
    slugs: list[str] | None = None,
    strict_stats: bool = False,
) -> LoadSummary:
    """Upsert every registered site plus its stats and raster layers.

    Each site runs in its own transaction, so one bad row doesn't roll back
    the batch. Returns a summary dict.
    """
    summary = LoadSummary()
    for site in all_sites():
        if slugs is not None and site.slug not in slugs:
            continue
        site_out = sites_dir / site.slug
        try:
            with engine.begin() as conn:
                site_id = upsert_dive_site(conn, site)
                summary.sites += 1

                stats_path = site_out / "stats.json"
                if stats_path.exists():
                    stats = json.loads(stats_path.read_text())
                    upsert_terrain_stats(conn, site_id, stats)
                    summary.stats += 1
                    if not stats.get("depth_zones"):
                        summary.sites_without_zones.append(site.slug)
                elif strict_stats:
                    raise FileNotFoundError(stats_path)

                if site_out.exists():
                    layers = upsert_site_rasters(conn, site_id, site_out)
                    summary.rasters += len(layers)
        except Exception as exc:
            log.exception("Failed to load site %s", site.slug)
            summary.skipped.append((site.slug, str(exc)))
    return summary


# -- Read for export ----------------------------------------------------------

@dataclass
class SiteRow:
    slug: str
    name: str
    lon: float | None
    lat: float | None
    easting: float
    northing: float
    region: str
    character: str
    description: str | None
    max_depth: float | None
    window_size: int
    stats: dict
    rasters: list[dict]


def fetch_sites_with_stats(engine: Engine) -> list[SiteRow]:
    """Query every site with its stats and raster references.

    Returns a list of SiteRow. Stats and rasters may be empty dicts/lists
    for sites without loaded data.
    """
    stats_cols = [c for c in site_terrain_stats.c if c.name != "id"]
    q = (
        select(
            dive_sites.c.id,
            dive_sites.c.slug,
            dive_sites.c.name,
            func.ST_X(dive_sites.c.geom).label("lon"),
            func.ST_Y(dive_sites.c.geom).label("lat"),
            func.ST_X(dive_sites.c.geom_utm).label("easting"),
            func.ST_Y(dive_sites.c.geom_utm).label("northing"),
            dive_sites.c.region,
            dive_sites.c.character,
            dive_sites.c.description,
            dive_sites.c.max_depth,
            dive_sites.c.window_size,
            *stats_cols,
        )
        .select_from(
            dive_sites.outerjoin(
                site_terrain_stats,
                site_terrain_stats.c.site_id == dive_sites.c.id,
            )
        )
        .order_by(dive_sites.c.slug)
    )

    rasters_q = (
        select(
            site_rasters.c.site_id,
            site_rasters.c.layer_name,
            site_rasters.c.file_path,
        )
        .order_by(site_rasters.c.site_id, site_rasters.c.layer_name)
    )

    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()
        raster_rows = conn.execute(rasters_q).mappings().all()

    rasters_by_site: dict[int, list[dict]] = {}
    for r in raster_rows:
        rasters_by_site.setdefault(r["site_id"], []).append(
            {"layer": r["layer_name"], "path": r["file_path"]}
        )

    stat_keys = [c.name for c in stats_cols if c.name != "site_id"]
    results: list[SiteRow] = []
    for row in rows:
        stats = {k: row[k] for k in stat_keys if row[k] is not None}
        results.append(SiteRow(
            slug=row["slug"], name=row["name"],
            lon=float(row["lon"]) if row["lon"] is not None else None,
            lat=float(row["lat"]) if row["lat"] is not None else None,
            easting=float(row["easting"]),
            northing=float(row["northing"]),
            region=row["region"], character=row["character"],
            description=row["description"],
            max_depth=float(row["max_depth"]) if row["max_depth"] is not None else None,
            window_size=row["window_size"],
            stats=stats,
            rasters=rasters_by_site.get(row["id"], []),
        ))
    return results


# -- GeoJSON construction (works from DB or file fallback) --------------------

_utm_to_wgs84 = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)


def site_rows_from_files(sites_dir: Path = Path("data/sites")) -> list[SiteRow]:
    """Fallback: build SiteRow list from the Python registry + stats.json."""
    results: list[SiteRow] = []
    for site in all_sites():
        stats_path = sites_dir / site.slug / "stats.json"
        stats: dict = {}
        if stats_path.exists():
            raw = json.loads(stats_path.read_text())
            stats = flatten_stats_for_row(raw)
            stats = {k: v for k, v in stats.items() if v is not None}

        rasters = []
        if (sites_dir / site.slug).exists():
            for layer in LAYER_NAMES:
                tif = sites_dir / site.slug / f"{layer}.tif"
                if tif.exists():
                    rasters.append({
                        "layer": layer,
                        "path": f"{site.slug}/{tif.name}",
                    })

        lon, lat = _utm_to_wgs84.transform(site.easting, site.northing)
        results.append(SiteRow(
            slug=site.slug, name=site.name,
            lon=lon, lat=lat,
            easting=site.easting, northing=site.northing,
            region=site.region, character=site.character,
            description=site.description, max_depth=site.max_depth,
            window_size=site.half_size * 2,
            stats=stats,
            rasters=rasters,
        ))
    return results


def site_rows_to_geojson(rows: list[SiteRow]) -> dict:
    """Build a FeatureCollection dict in WGS84."""
    from datetime import datetime, timezone
    features: list[dict[str, Any]] = []
    for r in rows:
        props: dict[str, Any] = {
            "slug": r.slug,
            "name": r.name,
            "region": r.region,
            "character": r.character,
            "description": r.description,
            "max_depth": r.max_depth,
            "window_size": r.window_size,
            "easting_25831": r.easting,
            "northing_25831": r.northing,
        }
        props.update(r.stats)
        props["rasters"] = r.rasters
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r.lon, r.lat],
            },
            "properties": props,
        })
    return {
        "type": "FeatureCollection",
        "name": "sotamar_dive_sites",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }
