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
docker compose up -d                    # production at :5432, test at :5433
docker compose exec postgis pg_isready -U sotamar -d sotamar
```

Two PostGIS instances are defined in `docker-compose.yml`:

- **`postgis`** on `localhost:5432` — your production data (DBeaver target).
- **`postgis-test`** on `localhost:5433` — disposable instance used by the
  pytest integration suite. Tests `TRUNCATE` the catalogue tables on every
  run, so they need their own cluster to keep the production DB pristine.

Both share the same `sql/schema.sql` (auto-loads on first boot) and the
same default credentials: `sotamar / sotamar / sotamar`.

### 4. Run the terrain-analysis pipeline

The site registry is CSV-driven: every verified POI in
`data/dive_sites.csv` becomes an analysis site at runtime
(currently 52 verified sites across 9 regions). Site slugs equal
POI ids — e.g. `med_meda_gran`, `pal_boreas`, `cdc_cap_creus`.

```bash
# List every registered dive site
uv run sotamar list

# Analyse every site (takes a few minutes per site)
uv run sotamar analyze --all

# …or just one, for a faster first run
uv run sotamar analyze med_meda_gran
```

Per-site outputs land in `data/sites/{slug}/`:
seven GeoTIFFs (bathymetry, slope, hillshade, BPI fine/broad, VRM,
depth zones), `stats.json`, and two figures
(`terrain_analysis.{pdf,png}`, `depth_profile.{pdf,png}`). POIs
listed in `data/dive_sites.csv` that fall inside a site's analysis
window are auto-overlaid on `terrain_analysis.png`.

Optional: scan unverified wreck POIs for wreck-shaped anomalies in
the bathymetry:

```bash
uv run sotamar detect-wrecks                # unverified wrecks only
uv run sotamar detect-wrecks --all-wrecks   # also re-runs against verified ones
```

This writes `data/wreck_candidates.csv` plus a per-POI
`wreck_candidates.png` debug figure under `data/sites/<poi_id>/`.

### 5. Load the catalogue into PostGIS

```bash
uv run sotamar load-db
```

This upserts the site registry, `stats.json`, raster references,
and the full POI catalogue from `data/dive_sites.csv` (verified
*and* unverified). Inspect the result:
```bash
docker compose exec postgis psql -U sotamar -d sotamar \
    -c "SELECT slug, name, region, max_depth FROM dive_sites ORDER BY slug;"
docker compose exec postgis psql -U sotamar -d sotamar \
    -c "SELECT slug, layer_name, file_path FROM site_rasters
        JOIN dive_sites ON site_rasters.site_id = dive_sites.id LIMIT 10;"
docker compose exec postgis psql -U sotamar -d sotamar \
    -c "SELECT id, name, region, site_type, coord_confidence
        FROM dive_site_pois ORDER BY region, id LIMIT 10;"
```

### 6. Generate the web viewer

```bash
uv run sotamar viewer                 # uses default --grid-size 100, no regional COG read
uv run sotamar viewer --cog data/icgc/batimetria-v2r1-elevacions-2021-2025.tif
```

Passing `--cog` enables the regional drill-down pages (one per
CSV region whose POIs cluster within a 4 km bbox). They render a
single shared bathymetry surface for the whole region with
clickable POI pins.

This writes a nested tree under `data/viewer/`:

```
data/viewer/
  index.html                       ← overview, one marker per renderable site
  med_meda_gran/
    depth.html                     ← default entry point
    zone.html
    slope.html
    bpi_fine.html
    bpi_broad.html
    vrm.html
    terrain_analysis.png           ← copied from data/sites/
    depth_profile.png
  pal_boreas/…
  illes_medes/                     ← regional page (when --cog is passed)
    index.html
  cap_de_creus/
    index.html
```

Each per-site page is a **tabbed multi-metric view**:

- The top tab bar (`Depth · Zone · Slope · Broad BPI · Fine BPI · VRM`)
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
| `sotamar detect-wrecks` | Scan bathymetry around unverified wreck POIs for wreck-shaped anomalies; write `data/wreck_candidates.csv` and per-POI debug figures. `--all-wrecks` also runs against verified wrecks. |
| `sotamar check-coords` | Geocode each site name (Nominatim) and flag registered coordinates that drift >2 km. Requires network. |
| `sotamar load-db` | Upsert the site registry, `stats.json`, raster references, and the POI catalogue into PostGIS. |
| `sotamar export-geojson` | Export all sites as a GeoJSON `FeatureCollection` (WGS84). Reads from the DB by default; `--from-files` bypasses it. |
| `sotamar viewer` | Generate the static HTML viewer. Same DB/`--from-files` contract as `export-geojson`. Pass `--cog` to also build per-region drill-down pages. |

Both `export-geojson` and `viewer` fail loudly when the DB is
unreachable unless you pass `--from-files`, so you always know which
source produced the output.

---

## Data model

Schema in `sql/schema.sql` — four tables:

- `dive_sites` — analysis-window catalogue (one row per verified
  POI used as a site). `geom_utm` (EPSG:25831) is the source of
  truth; `geom` (EPSG:4326) is maintained by a trigger.
- `site_terrain_stats` — one-to-one precomputed stats (per-metric
  min/max/mean/std, zone percentages, VRM threshold).
- `site_rasters` — references to the seven per-site GeoTIFFs.
  `file_path` is stored relative to the sites directory
  (e.g. `med_meda_gran/slope.tif`) — join with `--sites-dir`
  (default `data/sites`) to resolve.
- `dive_site_pois` — full POI catalogue from `data/dive_sites.csv`
  (verified + unverified: wrecks, pinnacles, coves, islands, walls,
  caves, headlands). `site_id` is a soft FK populated by point-in-
  polygon against `dive_sites.analysis_bbox`; NULL for POIs outside
  every analysis window.

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
docker compose up -d postgis-test    # one-time: start the test PostGIS
uv run pytest
```

Integration tests target `postgis-test` (port 5433); they skip silently
if it isn't reachable. The `db_url` fixture in `tests/conftest.py`
refuses to point at port 5432 so `TRUNCATE` can never hit production.
Override with `SOTAMAR_TEST_DB_URL` if you want to run against a
different test DB.

---

## Teardown

```bash
docker compose down        # keep the DB volume
docker compose down -v     # …or wipe it
```

Generated artefacts live under `data/`; nothing outside the repo is
touched.
