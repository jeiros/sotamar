"""Static HTML viewer: overview map + per-site 3D seabed via pydeck/deck.gl.

v2: per-site pages are a tabbed multi-metric view. Each tab (Depth, Dive
zone, Slope, Fine BPI, Broad BPI, VRM) re-colours the same 3D
GridCellLayer surface using the matching raster and a matplotlib
colormap. Beneath the 3D view, the pre-rendered terrain_analysis.png
figure, depth_profile.png, and a stats table make the full analytical
output visible without interaction.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib import colormaps
import numpy as np
import pydeck as pdk
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling

from sotamar.db import SiteRow
from sotamar.figures import ZONE_COLORS

log = logging.getLogger(__name__)

BASEMAP_STYLE = "light"

# Vertical exaggeration for per-site 3D views. Bathymetric relief (metres)
# is tiny next to site widths (kilometres), so a multiplier is standard
# in GIS bathymetry viz. 5× renders a 60 m depth range as 300 m visual
# height — legible at zoom 13 with pitch 55°.
VERTICAL_EXAGGERATION = 5.0

OCEAN_BG = "#0a3d62"  # deep Mediterranean; picked to contrast all colormaps

ZONE_KEYS = ("owd", "aowd", "deep", "tech")
ZONE_LABEL_BY_NUM = {
    1: "Zone 1 (OWD, 0 to −18 m)",
    2: "Zone 2 (AOWD, −18 to −30 m)",
    3: "Zone 3 (Deep, −30 to −40 m)",
    4: "Zone 4 (Technical, < −40 m)",
}

_utm_to_wgs84 = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)


def _hex_to_rgb(color: str) -> list[int]:
    c = color.lstrip("#")
    return [int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)]


ZONE_RGB = [_hex_to_rgb(c) for c in ZONE_COLORS]


# -- Metric registry ----------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    slug: str           # URL/filename key ("depth", "slope", "bpi_fine", …)
    label: str          # tab button text
    raster: str | None  # filename under data/sites/{slug}/, or None (derived)
    cmap: str           # matplotlib colormap
    norm: str           # "depth" | "zone" | "symmetric_p99" | "linear_p99"
    units: str          # tooltip units, "" if none
    precision: int      # decimal places in tooltip
    value_field: str    # key the tooltip references: "{depth}", "{slope}", …


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("depth",     "Depth",     "bathymetry.tif", "viridis",   "depth",         " m", 2, "depth"),
    MetricSpec("zone",      "Dive zone", None,             "",          "zone",          "",   0, "zone_label"),
    MetricSpec("slope",     "Slope",     "slope.tif",      "YlOrRd",    "linear_p99",    "°",  1, "slope"),
    MetricSpec("bpi_fine",  "Fine BPI",  "bpi_fine.tif",   "RdBu_r",    "symmetric_p99", "",   2, "bpi_fine"),
    MetricSpec("bpi_broad", "Broad BPI", "bpi_broad.tif",  "RdBu_r",    "symmetric_p99", "",   2, "bpi_broad"),
    MetricSpec("vrm",       "VRM",       "vrm.tif",        "inferno",   "linear_p99",    "",   4, "vrm"),
)

METRICS_BY_SLUG = {m.slug: m for m in METRICS}


# -- Zone helpers -------------------------------------------------------------

def _zone_from_depth(depth: float) -> int:
    """Map a depth in metres to a 1..4 zone (matches terrain.compute_depth_zones)."""
    if depth <= -40.0:
        return 4
    if depth <= -30.0:
        return 3
    if depth <= -18.0:
        return 2
    return 1


def _dominant_zone(stats: dict) -> int:
    """Return the 1..4 index of the dive zone holding the largest area %."""
    pcts = [stats.get(f"zone_{k}_pct") or 0.0 for k in ZONE_KEYS]
    if max(pcts) <= 0:
        return 1
    return int(np.argmax(pcts)) + 1


# -- Raster I/O ---------------------------------------------------------------

@dataclass
class RasterWindow:
    arr: np.ndarray
    bounds: tuple[float, float, float, float]  # (left, bottom, right, top) UTM
    nodata_mask: np.ndarray  # True where NoData/NaN


def downsample_raster(tif_path: Path, grid_size: int) -> RasterWindow:
    """Read a 1 m UTM raster resampled to (grid_size, grid_size) cells."""
    with rasterio.open(tif_path) as src:
        if src.crs is None or src.crs.to_epsg() != 25831:
            raise ValueError(f"{tif_path}: expected EPSG:25831, got {src.crs}")
        nodata = src.nodata
        data = src.read(
            1, out_shape=(grid_size, grid_size),
            resampling=Resampling.average,
        ).astype(np.float64)
        b = src.bounds
    mask = np.zeros_like(data, dtype=bool)
    if nodata is not None:
        mask |= data == nodata
    mask |= ~np.isfinite(data)
    data[mask] = np.nan
    return RasterWindow(arr=data, bounds=(b.left, b.bottom, b.right, b.top),
                        nodata_mask=mask)


# -- Colour mapping -----------------------------------------------------------

def _normaliser(spec: MetricSpec, arr: np.ndarray):
    """Build a matplotlib Normalize object appropriate for the metric."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    if spec.norm == "depth":
        vmin = float(valid.min())
        vmax = 0.0
        if vmin >= vmax:  # degenerate (all-zero or all-positive window)
            vmax = vmin + 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if spec.norm == "symmetric_p99":
        vmax = float(np.nanpercentile(np.abs(valid), 99)) or 1.0
        return mcolors.Normalize(vmin=-vmax, vmax=vmax)
    if spec.norm == "linear_p99":
        vmax = float(np.nanpercentile(valid, 99)) or 1.0
        return mcolors.Normalize(vmin=0.0, vmax=vmax)
    raise ValueError(f"unknown norm: {spec.norm!r}")


def compute_colors(arr: np.ndarray, spec: MetricSpec,
                   bathy: np.ndarray | None = None) -> np.ndarray:
    """Map a 2D value array to a 2D RGB uint8 array via the metric's cmap.

    For the discrete zone metric, look up ZONE_RGB by depth-derived zone.
    """
    if spec.norm == "zone":
        if bathy is None:
            raise ValueError("zone metric requires the bathymetry array")
        zones = np.where(
            np.isfinite(bathy),
            np.select(
                [bathy <= -40.0, bathy <= -30.0, bathy <= -18.0, bathy <= 0.0],
                [4, 3, 2, 1], default=0,
            ),
            0,
        )
        rgb = np.zeros((*zones.shape, 3), dtype=np.uint8)
        for z in (1, 2, 3, 4):
            rgb[zones == z] = ZONE_RGB[z - 1]
        return rgb
    cmap = colormaps[spec.cmap]
    norm = _normaliser(spec, arr)
    rgba = cmap(norm(arr))  # (H, W, 4), NaNs map to the "bad" colour
    rgb_float = rgba[..., :3]
    return (rgb_float * 255.0).round().astype(np.uint8)


# -- Record construction ------------------------------------------------------

def build_records(bathy_window: RasterWindow,
                  metric_window: RasterWindow | None,
                  spec: MetricSpec) -> list[dict]:
    """Per-cell records in WGS84 with geometry from bathymetry and colour
    from the metric. NoData/emerged cells are dropped so the surface has
    the same holes across every metric.
    """
    grid_size = bathy_window.arr.shape[0]
    left, bottom, right, top = bathy_window.bounds
    dx = (right - left) / grid_size
    dy = (top - bottom) / grid_size

    max_abs = float(np.nanmax(np.abs(bathy_window.arr[bathy_window.arr < 0]))) \
        if np.any(bathy_window.arr < 0) else 0.0

    if spec.norm == "zone":
        rgb = compute_colors(bathy_window.arr, spec, bathy=bathy_window.arr)
        metric_arr = bathy_window.arr  # not used in tooltip for zone
    else:
        marr = metric_window.arr if metric_window is not None else bathy_window.arr
        rgb = compute_colors(marr, spec, bathy=bathy_window.arr)
        metric_arr = marr

    records: list[dict] = []
    for r in range(grid_size):
        northing = top - (r + 0.5) * dy
        for c in range(grid_size):
            z = float(bathy_window.arr[r, c])
            if not np.isfinite(z) or z > 0.0:
                continue
            easting = left + (c + 0.5) * dx
            lon, lat = _utm_to_wgs84.transform(easting, northing)
            zone = _zone_from_depth(z)
            height_m = round(max_abs - abs(z), 2)

            # Metric value for tooltip (may be NaN for nodata-on-metric
            # but bathy-valid cells; render as dash).
            mv = float(metric_arr[r, c])
            if not np.isfinite(mv):
                mv_str = "—"
            else:
                mv_str = f"{round(mv, spec.precision)}"

            rec = {
                "lon": lon, "lat": lat, "height_m": height_m,
                "depth": round(z, 2),
                "zone": zone, "zone_label": ZONE_LABEL_BY_NUM[zone],
                "color": [int(rgb[r, c, 0]), int(rgb[r, c, 1]),
                          int(rgb[r, c, 2])],
            }
            # Metric-specific value field so the tooltip template resolves.
            if spec.value_field not in ("depth", "zone_label"):
                rec[spec.value_field] = mv_str
            records.append(rec)
    return records


# -- Overview -----------------------------------------------------------------

def _row_to_overview_record(row: SiteRow) -> dict:
    zone = _dominant_zone(row.stats)
    return {
        "slug": row.slug, "name": row.name, "region": row.region,
        "character": row.character, "description": row.description or "",
        "max_depth": row.max_depth, "lon": row.lon, "lat": row.lat,
        "owd_pct":  row.stats.get("zone_owd_pct"),
        "aowd_pct": row.stats.get("zone_aowd_pct"),
        "deep_pct": row.stats.get("zone_deep_pct"),
        "tech_pct": row.stats.get("zone_tech_pct"),
        "color": ZONE_RGB[zone - 1],
        "href": f"{row.slug}/depth.html",  # nested layout in v2
    }


def build_overview_deck(rows: list[SiteRow]) -> pdk.Deck:
    """Overview map: clickable site markers on a CARTO basemap."""
    records = [_row_to_overview_record(r) for r in rows if r.lon and r.lat]
    layer = pdk.Layer(
        "ScatterplotLayer", data=records,
        get_position=["lon", "lat"], get_fill_color="color",
        get_radius=600, radius_min_pixels=4, radius_max_pixels=18,
        pickable=True, stroked=True, get_line_color=[30, 30, 30],
        line_width_min_pixels=1,
    )
    lons = [r["lon"] for r in records]
    lats = [r["lat"] for r in records]
    view = pdk.ViewState(
        longitude=float(np.mean(lons)) if lons else 2.5,
        latitude=float(np.mean(lats)) if lats else 41.5,
        zoom=7.5, pitch=0, bearing=0,
    )
    tooltip = {
        "html": (
            "<b>{name}</b><br/>"
            "<i>{region} — {character}</i><br/>"
            "Max depth: {max_depth} m<br/>"
            "OWD {owd_pct}% · AOWD {aowd_pct}%<br/>"
            "Deep {deep_pct}% · Tech {tech_pct}%<br/>"
            "<span style='color:#1f78b4;'>Click marker to open 3D view →</span>"
        ),
        "style": {"backgroundColor": "white", "color": "#222",
                  "fontFamily": "system-ui, sans-serif", "fontSize": "12px"},
    }
    return pdk.Deck(
        layers=[layer], initial_view_state=view,
        map_style=BASEMAP_STYLE, tooltip=tooltip,
    )


_OVERVIEW_CLICK_SCRIPT = """
<script>
  deckInstance.setProps({
    getCursor: ({isHovering}) => (isHovering ? 'pointer' : 'grab'),
    onClick: (info) => {
      if (info && info.object && info.object.href) {
        window.location.href = info.object.href;
      }
    }
  });
</script>
"""


def _inject_overview_click_handler(html_path: Path) -> None:
    """Post-process pydeck static HTML to make markers click-navigate."""
    html = html_path.read_text()
    if "deckInstance" not in html:
        raise RuntimeError(
            f"{html_path}: pydeck template no longer exposes 'deckInstance'"
        )
    html = html.replace("</html>", f"{_OVERVIEW_CLICK_SCRIPT}</html>")
    html_path.write_text(html)


# -- Per-site deck ------------------------------------------------------------

def _tooltip_for(spec: MetricSpec) -> dict:
    """Per-metric tooltip text."""
    if spec.slug == "depth":
        body = "<b>Depth:</b> {depth} m<br/><b>{zone_label}</b>"
    elif spec.slug == "zone":
        body = "<b>{zone_label}</b><br/><b>Depth:</b> {depth} m"
    else:
        body = (
            f"<b>{spec.label}:</b> {{{spec.value_field}}}{spec.units}"
            "<br/><b>Depth:</b> {depth} m"
        )
    body += (
        f"<br/><i style='color:#666;'>Taller = shallower reef, ×"
        f"{VERTICAL_EXAGGERATION:g} vertical exaggeration</i>"
    )
    return {
        "html": body,
        "style": {"backgroundColor": "white", "color": "#222",
                  "fontFamily": "system-ui, sans-serif", "fontSize": "12px"},
    }


def build_site_deck(site_row: SiteRow, records: list[dict],
                    cell_metres: float, spec: MetricSpec) -> pdk.Deck:
    """Per-site 3D view: GridCellLayer surface, ocean-blue canvas (no basemap)."""
    cell_layer = pdk.Layer(
        "GridCellLayer", data=records,
        cell_size=cell_metres,
        get_position=["lon", "lat"], get_elevation="height_m",
        elevation_scale=VERTICAL_EXAGGERATION,
        get_fill_color="color",
        pickable=True, extruded=True, auto_highlight=True,
    )
    label_layer = pdk.Layer(
        "TextLayer",
        data=[{"lon": site_row.lon, "lat": site_row.lat,
               "text": f"{site_row.name} — {spec.label}"}],
        get_position=["lon", "lat"], get_text="text",
        get_size=22, get_color=[240, 240, 240],
        get_alignment_baseline="'bottom'", billboard=True,
    )
    view = pdk.ViewState(
        longitude=site_row.lon, latitude=site_row.lat,
        zoom=13, pitch=55, bearing=0,
    )
    return pdk.Deck(
        layers=[cell_layer, label_layer], initial_view_state=view,
        map_style=None, map_provider=None,
        tooltip=_tooltip_for(spec),
    )


# -- HTML chrome --------------------------------------------------------------

_SITE_STYLE = f"""
<style>
  body {{ margin: 0; background: {OCEAN_BG};
         font-family: system-ui, sans-serif; color: #eee; }}
  #deck-container {{ position: relative; width: 100vw; height: 80vh;
                     background: {OCEAN_BG} !important; }}
  #deck-container canvas {{ background: {OCEAN_BG} !important; }}

  #sotamar-tabs {{
    position: sticky; top: 0; z-index: 20;
    display: flex; background: #07304d; padding: 0 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
  }}
  #sotamar-tabs a {{
    display: inline-block; padding: 10px 16px; color: #cfd8dc;
    text-decoration: none; font-size: 13px; border-bottom: 2px solid transparent;
  }}
  #sotamar-tabs a:hover {{ color: #fff; }}
  #sotamar-tabs a.active {{
    color: #fff; border-bottom-color: #1f78b4; background: #0a3d62;
  }}

  #sotamar-legend {{
    position: absolute; top: 70px; left: 12px; z-index: 10;
    background: rgba(255,255,255,0.92); padding: 10px 14px;
    border-radius: 6px; font: 12px system-ui, sans-serif; color: #222;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3); max-width: 240px;
  }}
  #sotamar-legend h4 {{ margin: 0 0 6px 0; font-size: 13px; }}
  #sotamar-legend .row {{ display: flex; align-items: center; margin: 2px 0; }}
  #sotamar-legend .sw {{
    display: inline-block; width: 14px; height: 14px;
    margin-right: 8px; border: 1px solid #555;
  }}
  #sotamar-legend .note {{
    margin-top: 8px; font-style: italic; color: #555; font-size: 11px;
  }}

  #sotamar-panels {{
    padding: 20px 24px 40px 24px; color: #eee;
  }}
  #sotamar-panels h3 {{
    margin: 24px 0 10px 0; font-weight: 600; font-size: 15px;
    color: #e1e8ed; border-bottom: 1px solid #1f4a6b; padding-bottom: 4px;
  }}
  #sotamar-panels img {{ max-width: 100%; height: auto; border-radius: 4px;
                         background: white; padding: 6px; }}
  #sotamar-panels table.stats {{
    border-collapse: collapse; width: 100%; max-width: 720px; font-size: 13px;
  }}
  #sotamar-panels table.stats td {{
    padding: 4px 10px; border-bottom: 1px solid #1f4a6b;
  }}
  #sotamar-panels table.stats td:first-child {{ color: #9fb7cc; width: 34%; }}
  #sotamar-panels table.stats td:nth-child(2) {{ color: #d1e2f2; width: 30%; }}
  #sotamar-panels table.stats td:last-child {{ font-family: monospace; }}
</style>
"""


def _render_tab_bar(active_slug: str,
                    available_slugs: list[str]) -> str:
    parts = ['<div id="sotamar-tabs">']
    for m in METRICS:
        if m.slug not in available_slugs:
            continue
        cls = ' class="active"' if m.slug == active_slug else ""
        parts.append(f'<a href="{m.slug}.html"{cls}>{m.label}</a>')
    parts.append("</div>")
    return "".join(parts)


_LEGEND_BODY = f"""
<div id="sotamar-legend">
  <h4>Dive zone (by depth)</h4>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[0]};"></span>Zone 1 — OWD (0 to −18 m)</div>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[1]};"></span>Zone 2 — AOWD (−18 to −30 m)</div>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[2]};"></span>Zone 3 — Deep (−30 to −40 m)</div>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[3]};"></span>Zone 4 — Technical (&lt; −40 m)</div>
  <div class="note">Column height = rise above the deepest point in the window, ×{VERTICAL_EXAGGERATION:g} vertical exaggeration. Taller = shallower reef; flat floor = deepest parts. Emerged land (elevation &gt; 0 m) is excluded from the analysis.</div>
</div>
"""


def _fmt(v, suffix: str = "", places: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{places}f}{suffix}"


def _render_stats_table(stats: dict) -> str:
    """HTML table fragment summarising stats.json."""
    def cell_quad(key: str, unit: str = "", places: int = 2) -> str:
        s = stats.get(key) or {}
        parts = [_fmt(s.get(k), unit, places) for k in ("min", "max", "mean", "std")]
        return " / ".join(parts)

    zones = stats.get("depth_zones") or {}
    rows = [
        ("Depth", "min / max / mean / std", cell_quad("depth", " m", 2)),
        ("Slope", "min / max / mean / std", cell_quad("slope", "°", 2)),
        ("Fine BPI", "min / max / mean / std", cell_quad("bpi_fine", "", 2)),
        ("Broad BPI", "min / max / mean / std", cell_quad("bpi_broad", "", 2)),
        ("VRM", "min / max / mean / std", cell_quad("vrm", "", 3)),
        ("VRM > 0.003", "", _fmt(stats.get("vrm_pct_above_003"), " %", 2)),
        ("Dive zones", "OWD / AOWD / Deep / Tech",
         " / ".join(_fmt(zones.get(k), " %", 1)
                    for k in ("owd_pct", "aowd_pct", "deep_pct", "tech_pct"))),
        ("NoData (survey gap)", "", _fmt(stats.get("nodata_pct"), " %", 2)),
        ("Emerged (above sea)", "", _fmt(stats.get("emerged_pct"), " %", 2)),
    ]
    html = ['<table class="stats">']
    for name, detail, value in rows:
        html.append(f"<tr><td>{name}</td><td>{detail}</td><td>{value}</td></tr>")
    html.append("</table>")
    return "".join(html)


def _render_panels(stats: dict, figures_present: dict[str, bool]) -> str:
    parts = ['<div id="sotamar-panels">']
    parts.append("<h3>Numerical summary</h3>")
    parts.append(_render_stats_table(stats))
    if figures_present.get("terrain_analysis"):
        parts.append("<h3>Terrain analysis (6 panels)</h3>")
        parts.append('<img src="terrain_analysis.png" alt="terrain analysis">')
    if figures_present.get("depth_profile"):
        parts.append("<h3>Depth profile</h3>")
        parts.append('<img src="depth_profile.png" alt="depth profile">')
    parts.append("</div>")
    return "".join(parts)


def _inject_site_chrome(
    html_path: Path, active_slug: str, available_slugs: list[str],
    stats: dict, figures_present: dict[str, bool],
) -> None:
    """Add tab bar, legend, styles, and bottom panels to a per-site HTML."""
    html = html_path.read_text()
    html = html.replace("</head>", f"{_SITE_STYLE}</head>")
    tabs = _render_tab_bar(active_slug, available_slugs)
    panels = _render_panels(stats, figures_present)
    html = html.replace("<body>", f"<body>{tabs}{_LEGEND_BODY}")
    html = html.replace("</body>", f"{panels}</body>")
    html_path.write_text(html)


# -- Orchestration ------------------------------------------------------------

@dataclass
class ViewerSummary:
    overview: Path
    sites: list[tuple[str, Path, int]]  # (slug, directory, cell_count)
    skipped: list[tuple[str, str]]


def _copy_figure(src_dir: Path, dst_dir: Path, name: str) -> bool:
    src = src_dir / name
    if not src.exists():
        return False
    shutil.copyfile(src, dst_dir / name)
    return True


def write_site_pages(
    row: SiteRow, site_src_dir: Path, output_dir: Path, grid_size: int,
) -> tuple[Path, int] | None:
    """Emit one HTML per metric plus copy the figures. Returns
    (site_output_dir, submerged_cell_count) or None if the site is unplayable.
    """
    bathy_tif = site_src_dir / "bathymetry.tif"
    if not bathy_tif.exists():
        return None

    bathy_window = downsample_raster(bathy_tif, grid_size)
    if not np.any(bathy_window.arr < 0):
        return None  # all nodata / all emerged — nothing to draw

    cell_metres = row.window_size / grid_size
    site_out = output_dir / row.slug
    site_out.mkdir(parents=True, exist_ok=True)

    # Load stats.json (per-site numerical summary).
    import json
    stats_path = site_src_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    # Copy figures (both optional).
    figures_present = {
        "terrain_analysis": _copy_figure(site_src_dir, site_out,
                                         "terrain_analysis.png"),
        "depth_profile":    _copy_figure(site_src_dir, site_out,
                                         "depth_profile.png"),
    }

    # First pass: figure out which metrics we can actually render for this
    # site. Depth and Zone only need bathymetry (already loaded). The other
    # four need their raster on disk. Missing raster → skip that metric
    # entirely; tab bar only shows available tabs.
    metric_windows: dict[str, RasterWindow | None] = {}
    available: list[MetricSpec] = []
    for spec in METRICS:
        if spec.raster is None or spec.raster == "bathymetry.tif":
            metric_windows[spec.slug] = None  # uses bathy_window
            available.append(spec)
            continue
        tif = site_src_dir / spec.raster
        if not tif.exists():
            log.warning("Missing raster for %s/%s, skipping tab",
                        row.slug, spec.slug)
            continue
        try:
            metric_windows[spec.slug] = downsample_raster(tif, grid_size)
            available.append(spec)
        except Exception:
            log.exception("Downsample failed for %s/%s",
                          row.slug, spec.raster)

    available_slugs = [m.slug for m in available]

    cell_count = 0
    for spec in available:
        records = build_records(bathy_window, metric_windows[spec.slug], spec)
        if not cell_count:
            cell_count = len(records)
        deck = build_site_deck(row, records, cell_metres, spec)
        html_path = site_out / f"{spec.slug}.html"
        deck.to_html(str(html_path), iframe_height=800, notebook_display=False)
        _inject_site_chrome(html_path, spec.slug, available_slugs,
                            stats, figures_present)

    return site_out, cell_count


def write_viewer(
    rows: list[SiteRow], output_dir: Path, sites_dir: Path,
    grid_size: int = 100,
) -> ViewerSummary:
    """Write the overview plus a multi-metric per-site page tree."""
    # Wipe previous output so nested vs flat layouts don't collide.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    overview_path = output_dir / "index.html"
    build_overview_deck(rows).to_html(
        str(overview_path), iframe_height=800, notebook_display=False,
    )
    _inject_overview_click_handler(overview_path)

    sites_out: list[tuple[str, Path, int]] = []
    skipped: list[tuple[str, str]] = []
    for row in rows:
        tif = sites_dir / row.slug / "bathymetry.tif"
        if not tif.exists():
            skipped.append((row.slug, f"missing {tif}"))
            continue
        try:
            result = write_site_pages(row, sites_dir / row.slug,
                                      output_dir, grid_size)
        except Exception as exc:
            log.exception("Site render failed for %s", row.slug)
            skipped.append((row.slug, str(exc)))
            continue
        if result is None:
            skipped.append((row.slug, "no submerged cells after downsampling"))
            continue
        site_dir, cell_count = result
        sites_out.append((row.slug, site_dir, cell_count))

    return ViewerSummary(
        overview=overview_path, sites=sites_out, skipped=skipped,
    )
