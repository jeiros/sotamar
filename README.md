# sotamar

Bathymetric terrain analysis and spatial data model for Catalan coast dive sites.

Part of a TFG (UOC Computer Science) implementing a standardised spatial
data infrastructure for recreational dive-site characterisation from ICGC
coastal bathymetry.

---

## Quickstart

Everything below runs from the repo root.

### 1. Prerequisites

- **Python ≥ 3.12**, installed via [`uv`](https://docs.astral.sh/uv/).
- **Docker + Docker Compose** (for the PostGIS service).
- **ICGC bathymetry COG.** Download the 1 m topobathymetric DEM for
  the Catalan coast from the ICGC portal
  (<https://www.icgc.cat/ca/Geoinformacio-i-mapes/Dades-i-productes/Elevacions/Batimetria>,
  CC BY 4.0) and place it at:
  ```
  data/icgc/batimetria-v2r1-elevacions-2021-2025.tif
  ```
  The file is several GB. The analyser reads windowed subsets via
  rasterio — you don't need to slice it yourself.

### 2. Install Python deps

```bash
uv sync
```

### 3. Bring up PostGIS

```bash
docker compose up -d
docker compose exec postgis pg_isready -U sotamar -d sotamar
```

The schema in `sql/schema.sql` auto-loads on first boot. Default
credentials: `sotamar / sotamar` at `localhost:5432/sotamar`.

### 4. Run the terrain-analysis pipeline

```bash
# List the 18 registered dive sites
uv run sotamar list

# Analyse every site (takes a few minutes per site)
uv run sotamar analyze --all

# …or just one, for a faster first run
uv run sotamar analyze illes_medes
```

Per-site outputs land in `data/sites/{slug}/`:
seven GeoTIFFs (bathymetry, slope, hillshade, BPI fine/broad, VRM,
depth zones), `stats.json`, and two figures
(`terrain_analysis.{pdf,png}`, `depth_profile.{pdf,png}`).

### 5. Load the catalogue into PostGIS

```bash
uv run sotamar load-db
```

Inspect the result:
```bash
docker compose exec postgis psql -U sotamar -d sotamar \
    -c "SELECT slug, name, region, max_depth FROM dive_sites ORDER BY slug;"
docker compose exec postgis psql -U sotamar -d sotamar \
    -c "SELECT slug, layer_name, file_path FROM site_rasters
        JOIN dive_sites ON site_rasters.site_id = dive_sites.id LIMIT 10;"
```

### 6. Generate the web viewer

```bash
uv run sotamar viewer
```

This writes a nested tree under `data/viewer/`:

```
data/viewer/
  index.html                       ← overview, 18 markers on a CARTO basemap
  illes_medes/
    depth.html                     ← default entry point
    zone.html
    slope.html
    bpi_fine.html
    bpi_broad.html
    vrm.html
    terrain_analysis.png           ← copied from data/sites/
    depth_profile.png
  roses/…
```

Each per-site page is a **tabbed multi-metric view**:

- The top tab bar (`Depth · Zone · Slope · Fine BPI · Broad BPI · VRM`)
  re-colours the same 3D `GridCellLayer` surface using the matching
  raster and a matplotlib colormap aligned with the static thesis
  figures. Geometry (cell positions, heights) is shared across tabs so
  switching tab compares the same terrain under different analytical
  lenses.
- The surface extrudes upward on an ocean-blue canvas (no basemap):
  column height = `max|depth| − |depth|` with 5× vertical exaggeration,
  so shallow reef crests tower and the deepest flat areas lie flush
  with the floor.
- Below the 3D view every page embeds the pre-rendered 6-panel
  `terrain_analysis.png`, `depth_profile.png`, and a compact
  `stats.json` table.
- Missing raster → tab is omitted for that site. Sites with no
  bathymetry on disk are listed under "skipped" in the CLI output.

No server is needed — the files are self-contained and pull the
deck.gl bundle from unpkg on open.

### 7. Open it

```bash
# Linux / macOS
xdg-open data/viewer/index.html   # Linux
open     data/viewer/index.html   # macOS

# WSL
explorer.exe data/viewer/index.html
```

Click a marker → hover popup → **View 3D →** link navigates to the
per-site page. Drag to pan, right-drag (or `ctrl`-drag) to orbit, wheel
to zoom. Hover a column to see its depth and zone.

---

## Command reference

All commands take `--help` for full options.

| Command | Purpose |
|---|---|
| `sotamar list` | List registered sites and their UTM coordinates. |
| `sotamar analyze <slug> \| --all` | Run the full terrain analysis for one or every site. |
| `sotamar check-coords` | Geocode each site name (Nominatim) and flag registered coordinates that drift >2 km. Requires network. |
| `sotamar load-db` | Upsert the site registry, `stats.json`, and raster references into PostGIS. |
| `sotamar export-geojson` | Export all sites as a GeoJSON `FeatureCollection` (WGS84). Reads from the DB by default; `--from-files` bypasses it. |
| `sotamar viewer` | Generate the static HTML viewer. Same DB/`--from-files` contract as `export-geojson`. |

Both `export-geojson` and `viewer` fail loudly when the DB is
unreachable unless you pass `--from-files`, so you always know which
source produced the output.

---

## Data model

Schema in `sql/schema.sql` — three tables:

- `dive_sites` — catalogue. `geom_utm` (EPSG:25831) is the source of
  truth; `geom` (EPSG:4326) is maintained by a trigger.
- `site_terrain_stats` — one-to-one precomputed stats (per-metric
  min/max/mean/std, zone percentages, VRM threshold).
- `site_rasters` — references to the seven per-site GeoTIFFs.
  `file_path` is stored relative to the sites directory
  (e.g. `illes_medes/slope.tif`) — join with `--sites-dir`
  (default `data/sites`) to resolve.

### Schema re-apply

Docker only runs the init script when the volume is empty. To re-apply
after editing `sql/schema.sql` without nuking your data:

```bash
docker compose exec postgis psql -U sotamar -d sotamar \
    -f /docker-entrypoint-initdb.d/schema.sql
```

To start from a clean slate:
```bash
docker compose down -v     # drops the volume
docker compose up -d       # schema will auto-load
```

---

## Tests

```bash
uv run pytest
```

Integration tests that need PostGIS skip automatically if the service
isn't reachable.

---

## Teardown

```bash
docker compose down        # keep the DB volume
docker compose down -v     # …or wipe it
```

Generated artefacts live under `data/`; nothing outside the repo is
touched.
