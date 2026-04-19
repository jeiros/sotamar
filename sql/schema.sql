-- SotaMar — PostGIS spatial data model for Catalan coast dive sites.
-- Target: PostgreSQL 16 + PostGIS 3.4.
-- Authoritative CRS: EPSG:25831 (ETRS89 / UTM zone 31N). WGS84 geometries
-- are derived from the UTM source via a BEFORE INSERT/UPDATE trigger.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- dive_sites: catalog of dive sites.
-- geom_utm is the source of truth; geom (WGS84) is trigger-maintained.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dive_sites (
    id             SERIAL PRIMARY KEY,
    slug           TEXT NOT NULL UNIQUE,
    name           TEXT NOT NULL,
    geom           geometry(Point,   4326),
    geom_utm       geometry(Point,   25831) NOT NULL,
    region         TEXT NOT NULL,
    character      TEXT NOT NULL,
    description    TEXT,
    max_depth      REAL CHECK (max_depth IS NULL OR max_depth > 0),
    window_size    INTEGER NOT NULL CHECK (window_size > 0),
    analysis_bbox  geometry(Polygon, 25831) NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dive_sites_geom_gix     ON dive_sites USING GIST (geom);
CREATE INDEX IF NOT EXISTS dive_sites_geom_utm_gix ON dive_sites USING GIST (geom_utm);
CREATE INDEX IF NOT EXISTS dive_sites_bbox_gix     ON dive_sites USING GIST (analysis_bbox);
CREATE INDEX IF NOT EXISTS dive_sites_region_idx   ON dive_sites (region);

CREATE OR REPLACE FUNCTION dive_sites_sync() RETURNS TRIGGER AS $$
BEGIN
    NEW.geom := ST_Transform(NEW.geom_utm, 4326);
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS dive_sites_sync_trg ON dive_sites;
CREATE TRIGGER dive_sites_sync_trg
    BEFORE INSERT OR UPDATE ON dive_sites
    FOR EACH ROW EXECUTE FUNCTION dive_sites_sync();

-- ---------------------------------------------------------------------------
-- site_terrain_stats: one-to-one precomputed analysis results per site.
-- Zones are the four recreational-diving classes (OWD / AOWD / Deep / Tech).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS site_terrain_stats (
    id                 SERIAL PRIMARY KEY,
    site_id            INTEGER NOT NULL UNIQUE
                         REFERENCES dive_sites(id) ON DELETE CASCADE,
    nodata_pct         REAL,

    depth_min          REAL, depth_max     REAL, depth_mean     REAL, depth_std     REAL,
    slope_min          REAL, slope_max     REAL, slope_mean     REAL, slope_std     REAL,
    bpi_fine_min       REAL, bpi_fine_max  REAL, bpi_fine_mean  REAL, bpi_fine_std  REAL,
    bpi_broad_min      REAL, bpi_broad_max REAL, bpi_broad_mean REAL, bpi_broad_std REAL,
    vrm_min            REAL, vrm_max       REAL, vrm_mean       REAL, vrm_std       REAL,

    vrm_pct_above_003  REAL,

    zone_owd_pct       REAL,
    zone_aowd_pct      REAL,
    zone_deep_pct      REAL,
    zone_tech_pct      REAL,

    computed_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS site_terrain_stats_site_idx
    ON site_terrain_stats (site_id);

-- ---------------------------------------------------------------------------
-- site_rasters: file references for the seven per-site GeoTIFFs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS site_rasters (
    id          SERIAL PRIMARY KEY,
    site_id     INTEGER NOT NULL REFERENCES dive_sites(id) ON DELETE CASCADE,
    layer_name  TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    bbox        geometry(Polygon, 25831) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (site_id, layer_name),
    CHECK (layer_name IN (
        'bathymetry', 'slope', 'hillshade', 'bpi_fine',
        'bpi_broad', 'vrm', 'depth_zones'
    ))
);

CREATE INDEX IF NOT EXISTS site_rasters_bbox_gix ON site_rasters USING GIST (bbox);
CREATE INDEX IF NOT EXISTS site_rasters_site_idx ON site_rasters (site_id);
