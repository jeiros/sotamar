# sotamar

Bathymetric terrain analysis and spatial data model for Catalan coast dive sites.

Part of a TFG (UOC Computer Science) implementing a standardised spatial
data infrastructure for recreational dive-site characterisation from ICGC
coastal bathymetry.

## Install

```bash
uv sync
```

## Pipeline

```bash
# Analyse one site
uv run sotamar analyze illes_medes

# Analyse every registered site
uv run sotamar analyze --all

# List sites and sanity-check their coordinates
uv run sotamar list
uv run sotamar check-coords
```

Per-site outputs land in `data/sites/{slug}/`: seven GeoTIFFs (bathymetry,
slope, hillshade, BPI fine/broad, VRM, depth zones), `stats.json`, and
two figures (`terrain_analysis.*`, `depth_profile.*`).

## Viewer

Generate a standalone HTML viewer — overview map of every site plus a
3D ColumnLayer seabed per site:

```bash
uv run sotamar viewer                # reads from PostGIS
uv run sotamar viewer --from-files   # bypasses DB
```

Output lands in `data/viewer/`: `index.html` (18 markers on a CARTO
basemap) and one `{slug}.html` per site (3D seabed, coloured by
recreational dive zone). Open `index.html` in any browser — no server
needed.

## Database

A PostGIS service (Docker Compose) provides the catalogue store.

```bash
# Bring up PostGIS (schema auto-loads on first boot)
docker compose up -d
docker compose exec postgis pg_isready -U sotamar -d sotamar

# Load the registry + per-site stats + raster references
uv run sotamar load-db

# Export a GeoJSON FeatureCollection (for QGIS or the web viewer)
uv run sotamar export-geojson            # from DB
uv run sotamar export-geojson --from-files   # no DB needed
```

Schema is versioned in `sql/schema.sql` (three tables: `dive_sites`,
`site_terrain_stats`, `site_rasters`). `geom_utm` (EPSG:25831) is the
source of truth; `geom` (EPSG:4326) is trigger-derived.

`site_rasters.file_path` stores a path relative to the sites directory
(e.g. `illes_medes/slope.tif`) — join with the directory passed to
`--sites-dir` (default `data/sites`) to resolve it.

**Schema re-apply** after edits (Docker only runs init scripts on empty
volumes):
```bash
docker compose exec postgis psql -U sotamar -d sotamar \
    -f /docker-entrypoint-initdb.d/schema.sql
```

## Tests

```bash
uv run pytest
```

Database tests skip automatically if PostGIS is not reachable.
