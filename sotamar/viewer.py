"""Static HTML viewer: overview map + per-site 3D seabed via pydeck/deck.gl.

v2: per-site pages are a tabbed multi-metric view. Each tab (Depth, Dive
zone, Slope, Broad BPI, Fine BPI, VRM) re-colours the same 3D
GridCellLayer surface using the matching raster and a matplotlib
colormap. Beneath the 3D view, the pre-rendered terrain_analysis.png
figure, depth_profile.png, and a stats table make the full analytical
output visible without interaction.

Each tab is a separate static page. A small script mirrors the deck.gl
camera into sessionStorage so the orientation survives tab switches, each
page carries a legend matched to its own metric, and the tab bar links
back to the overview map. A plain mouse wheel scrolls the page (so the
stats and figures below the deck stay reachable); Ctrl/⌘ + wheel zooms
the 3D view.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pydeck as pdk
import rasterio
import rasterio.windows
from matplotlib import colormaps
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

# deck.gl's TextLayer builds its glyph atlas from an ASCII-only default
# character set, so Catalan site names (Tascó, Dofí, …) render with holes.
# pydeck runs every string prop through @deck.gl/json's expression parser,
# which chokes on bare `'`, `(`, `"`… — both as array elements and inside
# unquoted strings. The one shape that survives is pydeck's string-literal
# convention (same as get_alignment_baseline="'bottom'"): a single
# '…'-quoted string, which TextLayer then consumes as an iterable of
# characters. It cannot contain a straight quote, so labels render
# apostrophes with the typographic ’ (see build_region_deck).
_LABEL_CHARSET = (
    "' abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "’-–—(),.&/·º"
    "àáèéíïòóúüçÀÈÉÍÒÓÚÇŀ'"
)


# -- Metric registry ----------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    slug: str  # URL/filename key ("depth", "slope", "bpi_fine", …)
    label: str  # tab button text
    raster: str | None  # filename under data/sites/{slug}/, or None (derived)
    cmap: str  # matplotlib colormap
    norm: str  # "depth" | "zone" | "symmetric_p99" | "linear_p99"
    units: str  # tooltip units, "" if none
    precision: int  # decimal places in tooltip
    value_field: str  # key the tooltip references: "{depth}", "{slope}", …
    caption: str = ""  # one-line explanation rendered under the tab bar


# Tabs are ordered by usefulness for dive planning: the first three are
# directly actionable; broad BPI helps identify wall vs slope; fine BPI and
# VRM are scientific terrain-characterisation metrics most useful for the
# thesis side rather than for a diver picking a site.
METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "depth",
        "Depth",
        "bathymetry.tif",
        "viridis",
        "depth",
        " m",
        2,
        "depth",
        caption="Seabed depth in metres. The first thing every diver looks at.",
    ),
    MetricSpec(
        "zone",
        "Dive zone",
        None,
        "",
        "zone",
        "",
        0,
        "zone_label",
        caption="Recreational dive zones by depth: OWD / AOWD / Deep / Tech.",
    ),
    MetricSpec(
        "slope",
        "Slope",
        "slope.tif",
        "YlOrRd",
        "linear_p99",
        "°",
        1,
        "slope",
        caption="Terrain steepness in degrees. Steep ≈ wall, gentle ≈ sloping reef.",
    ),
    MetricSpec(
        "bpi_broad",
        "Broad BPI",
        "bpi_broad.tif",
        "RdBu_r",
        "symmetric_p99",
        "",
        2,
        "bpi_broad",
        caption=(
            "Broad-scale Bathymetric Position Index (25–50 m). "
            "Positive ≈ ridge or wall, negative ≈ basin or channel."
        ),
    ),
    MetricSpec(
        "bpi_fine",
        "Fine BPI",
        "bpi_fine.tif",
        "RdBu_r",
        "symmetric_p99",
        "",
        2,
        "bpi_fine",
        caption=(
            "Fine-scale BPI (3–5 m). Highlights individual boulders, "
            "outcrops and debris."
        ),
    ),
    MetricSpec(
        "vrm",
        "VRM",
        "vrm.tif",
        "inferno",
        "linear_p99",
        "",
        4,
        "vrm",
        caption=(
            "Vector Ruggedness Measure (3×3). Seabed roughness independent of slope "
            "— high values  mark structurally complex terrain (which is a recognised proxy for richer habitat.)."
        ),
    ),
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
            1,
            out_shape=(grid_size, grid_size),
            resampling=Resampling.average,
        ).astype(np.float64)
        b = src.bounds
    mask = np.zeros_like(data, dtype=bool)
    if nodata is not None:
        mask |= data == nodata
    mask |= ~np.isfinite(data)
    data[mask] = np.nan
    return RasterWindow(
        arr=data, bounds=(b.left, b.bottom, b.right, b.top), nodata_mask=mask
    )


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


def compute_colors(
    arr: np.ndarray, spec: MetricSpec, bathy: np.ndarray | None = None
) -> np.ndarray:
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
                [4, 3, 2, 1],
                default=0,
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


def build_records(
    bathy_window: RasterWindow, metric_window: RasterWindow | None, spec: MetricSpec
) -> list[dict]:
    """Per-cell records in WGS84 with geometry from bathymetry and colour
    from the metric. NoData/emerged cells are dropped so the surface has
    the same holes across every metric.
    """
    grid_size = bathy_window.arr.shape[0]
    left, bottom, right, top = bathy_window.bounds
    dx = (right - left) / grid_size
    dy = (top - bottom) / grid_size

    max_abs = (
        float(np.nanmax(np.abs(bathy_window.arr[bathy_window.arr < 0])))
        if np.any(bathy_window.arr < 0)
        else 0.0
    )

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
                "lon": lon,
                "lat": lat,
                "height_m": height_m,
                "depth": round(z, 2),
                "zone": zone,
                "zone_label": ZONE_LABEL_BY_NUM[zone],
                "color": [int(rgb[r, c, 0]), int(rgb[r, c, 1]), int(rgb[r, c, 2])],
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
        "slug": row.slug,
        "name": row.name,
        "region": row.region,
        "character": row.character,
        "description": row.description or "",
        "max_depth": row.max_depth,
        "lon": row.lon,
        "lat": row.lat,
        "owd_pct": row.stats.get("zone_owd_pct"),
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
        "ScatterplotLayer",
        data=records,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=600,
        radius_min_pixels=4,
        radius_max_pixels=18,
        pickable=True,
        stroked=True,
        get_line_color=[30, 30, 30],
        line_width_min_pixels=1,
    )
    lons = [r["lon"] for r in records]
    lats = [r["lat"] for r in records]
    view = pdk.ViewState(
        longitude=float(np.mean(lons)) if lons else 2.5,
        latitude=float(np.mean(lats)) if lats else 41.5,
        zoom=7.5,
        pitch=0,
        bearing=0,
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
        "style": {
            "backgroundColor": "white",
            "color": "#222",
            "fontFamily": "system-ui, sans-serif",
            "fontSize": "12px",
        },
    }
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style=BASEMAP_STYLE,
        tooltip=tooltip,  # pyright: ignore[reportArgumentType]
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
        "style": {
            "backgroundColor": "white",
            "color": "#222",
            "fontFamily": "system-ui, sans-serif",
            "fontSize": "12px",
        },
    }


def build_site_deck(
    site_row: SiteRow, records: list[dict], cell_metres: float, spec: MetricSpec
) -> pdk.Deck:
    """Per-site 3D view: GridCellLayer surface, ocean-blue canvas (no basemap).

    The site name lives in the HTML tab bar, not in the scene: an
    in-canvas TextLayer at z=0 gets clipped by the extruded terrain and
    reads as a glitch from most camera angles.
    """
    cell_layer = pdk.Layer(
        "GridCellLayer",
        data=records,
        cell_size=cell_metres,
        get_position=["lon", "lat"],
        get_elevation="height_m",
        elevation_scale=VERTICAL_EXAGGERATION,
        get_fill_color="color",
        pickable=True,
        extruded=True,
        auto_highlight=True,
    )
    # Frame the analysis window instead of opening at a fixed zoom: a
    # 150 m wreck window at zoom 13 is a ~10-pixel blob. Aim for the
    # window spanning ~210 px at the focal plane (Web Mercator:
    # mpp = 156543.03·cos(lat) / 2^zoom) — at pitch 55° the foreground
    # stretches ~2-3×, so this fills the canvas with margin to orbit.
    mpp_wanted = site_row.window_size / 210.0
    lat = site_row.lat if site_row.lat is not None else 41.5  # Catalan coast
    zoom = math.log2(156543.03 * math.cos(math.radians(lat)) / mpp_wanted)
    zoom = min(max(zoom, 13.0), 17.5)
    view = pdk.ViewState(
        longitude=site_row.lon,
        latitude=site_row.lat,
        zoom=zoom,
        pitch=55,
        bearing=0,
    )
    return pdk.Deck(
        layers=[cell_layer],
        initial_view_state=view,
        map_style=None,
        map_provider=None,  # pyright: ignore[reportArgumentType]
        tooltip=_tooltip_for(spec),  # pyright: ignore[reportArgumentType]
    )


# -- HTML chrome --------------------------------------------------------------

_SITE_STYLE = f"""
<style>
  /* pydeck's template sets body {{ overflow: hidden }}, which kills page
     scrolling entirely — the stats/figure panels below the deck would be
     unreachable. This block is injected after pydeck's, so it wins. */
  body {{ margin: 0; background: {OCEAN_BG};
         overflow-y: auto; overflow-x: hidden;
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
  #sotamar-tabs a.back {{
    color: #9fc6e8; border-right: 1px solid #0a3d62; margin-right: 8px;
  }}
  #sotamar-tabs .site {{
    padding: 10px 14px 10px 4px; color: #fff;
    font-size: 13px; font-weight: 600;
  }}
  .metric-caption {{
    background: #07304d; color: #b0c4de; padding: 6px 18px 8px 18px;
    font-size: 12px; font-style: italic; border-top: 1px solid #0a3d62;
  }}

  #sotamar-scalebar {{
    position: absolute; bottom: 14px; left: 14px; z-index: 10;
    background: rgba(0,0,0,0.55); color: #fff; padding: 4px 8px 6px 8px;
    border-radius: 4px; font: 11px system-ui, sans-serif;
    pointer-events: none;
  }}
  #sotamar-scalebar .bar {{
    height: 4px; background: #fff; margin-bottom: 3px;
    border-left: 1px solid #fff; border-right: 1px solid #fff;
    transition: width 0.1s linear;
  }}
  #sotamar-scalebar .label {{ text-align: center; }}

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
  #sotamar-legend .grad {{
    height: 12px; border: 1px solid #555; border-radius: 2px;
  }}
  #sotamar-legend .gradlabels {{
    display: flex; justify-content: space-between;
    font-size: 11px; color: #333; margin-top: 2px;
  }}

  #sotamar-scroll-hint {{
    position: absolute; left: 50%; top: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.65); color: #fff;
    padding: 10px 18px; border-radius: 6px;
    font: 13px system-ui, sans-serif;
    opacity: 0; transition: opacity 0.25s ease;
    pointer-events: none; z-index: 12;
  }}
  #sotamar-scroll-hint.visible {{ opacity: 1; }}

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


def _render_tab_bar(
    active_slug: str, available_slugs: list[str], site_name: str
) -> str:
    parts = [
        '<div id="sotamar-tabs">',
        '<a class="back" href="../index.html">← All sites</a>',
        f'<span class="site">{escape(site_name)}</span>',
    ]
    active_caption = ""
    for m in METRICS:
        if m.slug not in available_slugs:
            continue
        cls = ' class="active"' if m.slug == active_slug else ""
        parts.append(f'<a href="{m.slug}.html"{cls}>{m.label}</a>')
        if m.slug == active_slug:
            active_caption = m.caption
    parts.append("</div>")
    if active_caption:
        parts.append(f'<div class="metric-caption">{active_caption}</div>')
    return "".join(parts)


_LEGEND_NOTE = (
    "Column height = rise above the deepest point in the window, "
    f"×{VERTICAL_EXAGGERATION:g} vertical exaggeration. Taller = shallower "
    "reef; flat floor = deepest parts. Emerged land (elevation &gt; 0 m) is "
    "excluded from the analysis."
)

_ZONE_LEGEND_BODY = f"""
<div id="sotamar-legend">
  <h4>Dive zone (by depth)</h4>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[0]};"></span>Zone 1 — OWD (0 to −18 m)</div>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[1]};"></span>Zone 2 — AOWD (−18 to −30 m)</div>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[2]};"></span>Zone 3 — Deep (−30 to −40 m)</div>
  <div class="row"><span class="sw" style="background:{ZONE_COLORS[3]};"></span>Zone 4 — Technical (&lt; −40 m)</div>
  <div class="note">{_LEGEND_NOTE}</div>
</div>
"""


def _cmap_css_gradient(cmap_name: str, n_stops: int = 12) -> str:
    """CSS linear-gradient sampling a matplotlib colormap left→right."""
    cmap = colormaps[cmap_name]
    stops = ", ".join(mcolors.to_hex(cmap(i / (n_stops - 1))) for i in range(n_stops))
    return f"linear-gradient(to right, {stops})"


def _legend_for(spec: MetricSpec, arr: np.ndarray) -> str:
    """Legend matched to the active metric: zone swatches for the discrete
    dive-zone tab, otherwise the metric's colormap as a gradient bar
    labelled with this site's actual value range."""
    if spec.norm == "zone":
        return _ZONE_LEGEND_BODY
    norm = _normaliser(spec, arr)
    # _normaliser always sets both bounds; the Nones are only in the stubs.
    vmin = float(norm.vmin) if norm.vmin is not None else 0.0
    vmax = float(norm.vmax) if norm.vmax is not None else 1.0

    def fmt(v: float, signed: bool = False) -> str:
        text = f"{v:+.{spec.precision}f}" if signed else f"{v:.{spec.precision}f}"
        return f"{text}{spec.units}"

    if spec.norm == "symmetric_p99":
        labels = (fmt(vmin, signed=True), fmt(0.0), fmt(vmax, signed=True))
    else:
        labels = (fmt(vmin), fmt((vmin + vmax) / 2.0), fmt(vmax))
    spans = "".join(f"<span>{t}</span>" for t in labels)

    note = _LEGEND_NOTE
    if spec.norm.endswith("_p99"):
        note = (
            "Colour range clipped at the 99th percentile for contrast "
            "(extremes saturate). " + _LEGEND_NOTE
        )
    return (
        '<div id="sotamar-legend">'
        f"<h4>{spec.label}</h4>"
        f'<div class="grad" style="background:{_cmap_css_gradient(spec.cmap)};"></div>'
        f'<div class="gradlabels">{spans}</div>'
        f'<div class="note">{note}</div>'
        "</div>"
    )


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
        (
            "Dive zones",
            "OWD / AOWD / Deep / Tech",
            " / ".join(
                _fmt(zones.get(k), " %", 1)
                for k in ("owd_pct", "aowd_pct", "deep_pct", "tech_pct")
            ),
        ),
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


_SCALEBAR_BODY = """
<div id="sotamar-scalebar">
  <div class="bar"></div>
  <div class="label">— m</div>
</div>
"""

# Suppress the browser context menu only on the WebGL canvas so right-click
# drag can rotate the deck.gl camera without a context menu interrupting.
# Pairs with the dynamic scale bar that reads deck.gl's current viewport.
_SITE_CANVAS_SCRIPT = """
<script>
  (function () {
    document.addEventListener('contextmenu', function (e) {
      if (e.target && e.target.tagName === 'CANVAS') e.preventDefault();
    });

    var bar = document.getElementById('sotamar-scalebar');
    if (!bar || typeof deckInstance === 'undefined') return;
    // Anchor the scale bar to the 3D container, not the document: as a
    // direct <body> child its absolute position resolves against the
    // page, where it overlaps the panels and scrolls away with them.
    var dc = document.getElementById('deck-container');
    if (dc) dc.appendChild(bar);
    var barEl = bar.querySelector('.bar');
    var labelEl = bar.querySelector('.label');

    function pickNiceLength(maxMeters) {
      var candidates = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
      for (var i = candidates.length - 1; i >= 0; i--) {
        if (candidates[i] <= maxMeters) return candidates[i];
      }
      return candidates[0];
    }

    function update() {
      try {
        var vps = deckInstance.viewManager && deckInstance.viewManager.getViewports();
        if (!vps || !vps[0]) return;
        var vp = vps[0];
        var mpp = vp.metersPerPixel;
        if (typeof mpp !== 'number') {
          // Fallback: Web Mercator approximation at the viewport latitude
          var lat = (vp.latitude || 0) * Math.PI / 180;
          mpp = 40075016.686 * Math.cos(lat) / Math.pow(2, vp.zoom + 8);
        }
        var targetPx = 120;
        var nice = pickNiceLength(targetPx * mpp);
        var widthPx = Math.max(20, nice / mpp);
        barEl.style.width = widthPx + 'px';
        labelEl.innerText = nice >= 1000 ? (nice / 1000) + ' km' : nice + ' m';
      } catch (err) { /* swallow — bar just won't update this frame */ }
    }

    // Drive updates on every animation frame: cheap, and catches every
    // viewstate change including user-driven pan/zoom/rotate.
    function tick() { update(); requestAnimationFrame(tick); }
    requestAnimationFrame(tick);
  })();
</script>
"""

# Persist the deck.gl camera across metric tabs. Each tab is a separate
# static page, so without this every switch snaps back to the build-time
# initialViewState and whatever orientation the user set up is lost. The
# live viewState is mirrored into sessionStorage (keyed by the site
# directory) and re-applied on the next page load.
_SITE_CAMERA_SCRIPT = """
<script>
  (function () {
    if (typeof deckInstance === 'undefined') return;
    var storage = null;
    try { storage = window.sessionStorage; } catch (err) { return; }
    if (!storage) return;
    var key = 'sotamar-camera:' + location.pathname.replace(/[^/]*$/, '');

    deckInstance.setProps({
      onViewStateChange: function (ev) {
        if (!ev || !ev.viewState) return;
        var vs = ev.viewState;
        // Store only the plain camera fields; transition props don't
        // survive JSON and must not leak into initialViewState.
        var plain = {
          longitude: vs.longitude, latitude: vs.latitude,
          zoom: vs.zoom, pitch: vs.pitch, bearing: vs.bearing
        };
        try { storage.setItem(key, JSON.stringify(plain)); }
        catch (err) { /* storage blocked — tabs just won't share camera */ }
      }
    });

    var saved = null;
    try { saved = JSON.parse(storage.getItem(key) || 'null'); }
    catch (err) { saved = null; }
    if (saved && typeof saved.longitude === 'number'
              && typeof saved.latitude === 'number') {
      // A fresh initialViewState object makes deck.gl rebase the camera.
      deckInstance.setProps({ initialViewState: saved });
    }
  })();
</script>
"""

# Wheel gate (cooperative gestures). The 3D canvas fills 80vh, and deck.gl
# preventDefault()s every wheel event over it, so the stats table and
# figures below the fold are unreachable with a mouse wheel. Standard
# embedded-map fix: plain wheel scrolls the page, Ctrl/⌘ + wheel zooms the
# deck. The capture-phase stopPropagation keeps unmodified wheels from
# mjolnir.js's canvas listener (and its preventDefault), so the browser's
# default page scroll proceeds. Trackpad pinch arrives as ctrl+wheel and
# still zooms. A transient overlay teaches the modifier.
_SITE_SCROLL_SCRIPT = """
<script>
  (function () {
    var container = document.getElementById('deck-container');
    if (!container) return;

    var hint = document.createElement('div');
    hint.id = 'sotamar-scroll-hint';
    var mod = /Mac|iP(hone|ad|od)/.test(navigator.platform || '') ? '⌘' : 'Ctrl';
    hint.textContent = 'Use ' + mod + ' + scroll to zoom the 3D view';
    container.appendChild(hint);

    var hideTimer = null;
    container.addEventListener('wheel', function (e) {
      if (e.ctrlKey || e.metaKey) return;  // deck.gl zooms
      e.stopPropagation();                 // deck never sees it → page scrolls
      hint.classList.add('visible');
      clearTimeout(hideTimer);
      hideTimer = setTimeout(function () {
        hint.classList.remove('visible');
      }, 1200);
    }, {capture: true, passive: true});
  })();
</script>
"""


def _inject_site_chrome(
    html_path: Path,
    active_slug: str,
    available_slugs: list[str],
    stats: dict,
    figures_present: dict[str, bool],
    legend_html: str,
    site_name: str,
) -> None:
    """Add tab bar, metric legend, styles, scale bar, camera persistence,
    and bottom panels to a per-site HTML."""
    html = html_path.read_text()
    html = html.replace(
        "<title>pydeck</title>", f"<title>{escape(site_name)} — SotAMar</title>", 1
    )
    html = html.replace("</head>", f"{_SITE_STYLE}</head>")
    tabs = _render_tab_bar(active_slug, available_slugs, site_name)
    panels = _render_panels(stats, figures_present)
    html = html.replace(
        "<body>",
        f"<body>{tabs}{legend_html}{_SCALEBAR_BODY}",
    )
    html = html.replace("</body>", f"{panels}</body>")
    # Scripts that touch deckInstance must come AFTER pydeck's own script,
    # which the template places between </body> and </html> — injecting
    # them at </body> would run them before deckInstance exists and their
    # typeof guards would silently no-op (region pages already do this).
    html = html.replace(
        "</html>",
        f"{_SITE_CANVAS_SCRIPT}{_SITE_CAMERA_SCRIPT}{_SITE_SCROLL_SCRIPT}</html>",
    )
    html_path.write_text(html)


# -- Orchestration ------------------------------------------------------------


@dataclass
class ViewerSummary:
    overview: Path
    sites: list[tuple[str, Path, int]]  # (slug, directory, cell_count)
    regions: list[tuple[str, Path, int]] = field(default_factory=list)
    # (region_name, html_path, n_pois) for every regional page generated
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _copy_figure(src_dir: Path, dst_dir: Path, name: str) -> bool:
    src = src_dir / name
    if not src.exists():
        return False
    shutil.copyfile(src, dst_dir / name)
    return True


def write_site_pages(
    row: SiteRow,
    site_src_dir: Path,
    output_dir: Path,
    grid_size: int,
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
        "terrain_analysis": _copy_figure(
            site_src_dir, site_out, "terrain_analysis.png"
        ),
        "depth_profile": _copy_figure(site_src_dir, site_out, "depth_profile.png"),
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
            log.warning("Missing raster for %s/%s, skipping tab", row.slug, spec.slug)
            continue
        try:
            metric_windows[spec.slug] = downsample_raster(tif, grid_size)
            available.append(spec)
        except Exception:
            log.exception("Downsample failed for %s/%s", row.slug, spec.raster)

    available_slugs = [m.slug for m in available]

    cell_count = 0
    for spec in available:
        records = build_records(bathy_window, metric_windows[spec.slug], spec)
        if not cell_count:
            cell_count = len(records)
        deck = build_site_deck(row, records, cell_metres, spec)
        html_path = site_out / f"{spec.slug}.html"
        deck.to_html(str(html_path), iframe_height=800, notebook_display=False)
        metric_window = metric_windows[spec.slug]
        legend_arr = (
            metric_window.arr if metric_window is not None else bathy_window.arr
        )
        _inject_site_chrome(
            html_path,
            spec.slug,
            available_slugs,
            stats,
            figures_present,
            legend_html=_legend_for(spec, legend_arr),
            site_name=row.name,
        )

    return site_out, cell_count


def _read_region_bathymetry(
    bounds: tuple[float, float, float, float],
    cog_path: Path | None,
    grid_size: int,
) -> RasterWindow:
    """Read a UTM bbox from the master COG, resampled to grid_size × grid_size."""
    from sotamar.io import find_cog

    cog = find_cog(cog_path)
    left, bottom, right, top = bounds
    with rasterio.open(cog) as src:
        if src.crs is None or src.crs.to_epsg() != 25831:
            raise ValueError(f"{cog}: expected EPSG:25831, got {src.crs}")
        window = rasterio.windows.from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src.transform,
        )
        nodata = src.nodata
        data = src.read(
            1,
            window=window,
            out_shape=(grid_size, grid_size),
            resampling=Resampling.average,
            boundless=True,
            fill_value=nodata if nodata is not None else -9999.0,
        ).astype(np.float64)
    mask = np.zeros_like(data, dtype=bool)
    if nodata is not None:
        mask |= data == nodata
    mask |= ~np.isfinite(data)
    data[mask] = np.nan
    return RasterWindow(arr=data, bounds=bounds, nodata_mask=mask)


def _region_records(
    bathy: RasterWindow,
    max_abs: float,
    region_pois: list[SiteRow],
) -> list[dict]:
    """Build GridCellLayer records for a regional bathymetry window.

    Uses the depth metric; emerged cells (z > 0) are dropped.
    """
    spec = METRICS_BY_SLUG["depth"]
    rgb = compute_colors(bathy.arr, spec)
    grid_h, grid_w = bathy.arr.shape
    left, bottom, right, top = bathy.bounds
    dx = (right - left) / grid_w
    dy = (top - bottom) / grid_h
    records: list[dict] = []
    for r in range(grid_h):
        northing = top - (r + 0.5) * dy
        for c in range(grid_w):
            z = float(bathy.arr[r, c])
            if not np.isfinite(z) or z > 0.0:
                continue
            easting = left + (c + 0.5) * dx
            lon, lat = _utm_to_wgs84.transform(easting, northing)
            records.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "depth": round(z, 2),
                    "height_m": round(max_abs - abs(z), 2),
                    "color": [int(rgb[r, c, 0]), int(rgb[r, c, 1]), int(rgb[r, c, 2])],
                }
            )
    return records


def build_region_deck(
    region_name: str,
    pois: list[SiteRow],
    bathy: RasterWindow,
    cell_metres: float,
) -> pdk.Deck:
    """Regional 3D view: seabed surface + clickable POI pins."""
    valid = bathy.arr[np.isfinite(bathy.arr) & (bathy.arr < 0)]
    max_abs = float(np.abs(valid).max()) if valid.size else 1.0
    records = _region_records(bathy, max_abs, pois)
    if not records:
        # Fall back to an "empty seabed" — still draw the pins.
        records = []

    cell_layer = pdk.Layer(
        "GridCellLayer",
        data=records,
        cell_size=cell_metres,
        get_position=["lon", "lat"],
        get_elevation="height_m",
        elevation_scale=VERTICAL_EXAGGERATION,
        get_fill_color="color",
        pickable=False,
        extruded=True,
    )

    # 3D beacon pins. Flat scatter dots at z=0 vanish behind the extruded
    # terrain as soon as the camera tilts or zooms in, and their labels
    # clip through the columns. Instead each POI gets a thin needle from
    # the seabed to above the tallest terrain column, topped by a
    # billboarded head and its label — visible from any angle, and both
    # needle and head stay clickable.
    beacon_top = max_abs * VERTICAL_EXAGGERATION + 30.0
    pin_data = [
        {
            # Straight apostrophes break pydeck's expression parser when
            # they appear in the TextLayer character set, so labels use
            # the typographic ’ (L'Encalladora → L’Encalladora).
            "lon": p.lon,
            "lat": p.lat,
            "name": p.name.replace("'", "’"),
            "top": beacon_top,
            "color": _row_to_overview_record(p)["color"],
            "href": f"../{p.slug}/depth.html",
        }
        for p in pois
        if p.lon and p.lat
    ]
    # Needle radius is in metres: radius_units="pixels" renders giant
    # cylinders through pydeck's JSON pipeline (deck.gl picks the wrong
    # projection scale), so stick to world units — ~3 px at the default
    # zoom, a slim pole when zoomed in. Flat-lit white, not zone-coloured:
    # when the camera is down among the terrain columns the pole is the
    # only part of the beacon in view, and viridis contains no white, so
    # it stays unmissable at every zoom (the head keeps the zone colour).
    # Markers render with the depth test disabled (luma.gl v9 parameter
    # names): when the camera sits down among the terrain columns, any
    # depth-tested marker is occluded by the nearer columns — beacons
    # must draw through the terrain, like waypoints in a game HUD.
    _XRAY = {"depthCompare": "always", "depthWriteEnabled": False}
    needle_layer = pdk.Layer(
        "ColumnLayer",
        data=pin_data,
        get_position=["lon", "lat"],
        get_elevation="top",
        radius=10,
        get_fill_color=[250, 250, 252],
        material=False,
        pickable=True,
        extruded=True,
        auto_highlight=True,
        parameters=_XRAY,
    )
    head_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pin_data,
        get_position=["lon", "lat", "top"],
        get_fill_color="color",
        billboard=True,
        get_radius=12,
        radius_min_pixels=5,
        radius_max_pixels=14,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
        parameters=_XRAY,
    )
    label_layer = pdk.Layer(
        "TextLayer",
        data=pin_data,
        get_position=["lon", "lat", "top"],
        get_text="name",
        get_color=[255, 255, 255],
        get_size=12,
        get_pixel_offset=[0, -16],
        get_alignment_baseline="'bottom'",
        billboard=True,
        background=True,
        get_background_color=[7, 48, 77, 200],
        background_padding=[4, 2],
        character_set=_LABEL_CHARSET,
        parameters=_XRAY,
    )

    lons = [float(r["lon"]) for r in pin_data]
    lats = [float(r["lat"]) for r in pin_data]
    view = pdk.ViewState(
        longitude=float(np.mean(lons)) if lons else 2.5,
        latitude=float(np.mean(lats)) if lats else 41.5,
        zoom=14,
        pitch=45,
        bearing=0,
    )
    tooltip = {
        "html": "<b>{name}</b><br/><i style='color:#1f78b4;'>Click pin → site page</i>",
        "style": {
            "backgroundColor": "white",
            "color": "#222",
            "fontFamily": "system-ui, sans-serif",
            "fontSize": "12px",
        },
    }
    return pdk.Deck(
        layers=[cell_layer, needle_layer, head_layer, label_layer],
        initial_view_state=view,
        map_style=None,
        map_provider=None,  # pyright: ignore[reportArgumentType]
        tooltip=tooltip,  # pyright: ignore[reportArgumentType]
    )


def _inject_region_chrome(html_path: Path, region_name: str, n_pois: int) -> None:
    """Add scale bar, context-menu suppressor, click handler, and a back link."""
    html = html_path.read_text()
    if "deckInstance" not in html:
        raise RuntimeError(
            f"{html_path}: pydeck template no longer exposes 'deckInstance'"
        )
    header = (
        '<div id="sotamar-region-header">'
        f'<a href="../index.html">← Back to overview</a> · '
        f'<b>{region_name}</b> &nbsp;<span class="meta">'
        f"{n_pois} sites · click a pin to drill in</span>"
        "</div>"
    )
    html = html.replace("</head>", f"{_SITE_STYLE}{_REGION_HEADER_CSS}</head>")
    html = html.replace(
        "<body>",
        f"<body>{header}{_SCALEBAR_BODY}",
    )
    html = html.replace(
        "</html>",
        f"{_SITE_CANVAS_SCRIPT}{_OVERVIEW_CLICK_SCRIPT}</html>",
    )
    html_path.write_text(html)


_REGION_HEADER_CSS = """
<style>
  #sotamar-region-header {
    position: sticky; top: 0; z-index: 30;
    background: #07304d; color: #cfd8dc;
    padding: 8px 18px; font: 13px system-ui, sans-serif;
    border-bottom: 1px solid #0a3d62;
  }
  #sotamar-region-header a { color: #aacfff; text-decoration: none; }
  #sotamar-region-header a:hover { text-decoration: underline; }
  #sotamar-region-header b { color: #fff; }
  #sotamar-region-header .meta { color: #88a3bc; font-size: 12px; }
</style>
"""


def write_region_pages(
    rows: list[SiteRow],
    output_dir: Path,
    cog_path: Path | None = None,
    min_pois: int = 3,
    max_bbox_m: float = 4000,
    grid_size: int = 200,
    margin_m: float = 200,
) -> list[tuple[str, Path, int]]:
    """For each CSV region with ≥min_pois sites within max_bbox_m, write a
    regional 3D view with clickable POI pins.

    Returns a list of (region_name, html_path, n_pois) for every region
    page actually generated.
    """
    by_region: dict[str, list[SiteRow]] = {}
    for row in rows:
        if row.lon is None or row.lat is None:
            continue
        by_region.setdefault(row.region, []).append(row)

    generated: list[tuple[str, Path, int]] = []
    for region, pois in sorted(by_region.items()):
        if len(pois) < min_pois:
            continue
        eastings = [p.easting for p in pois]
        northings = [p.northing for p in pois]
        left, right = min(eastings) - margin_m, max(eastings) + margin_m
        bottom, top = min(northings) - margin_m, max(northings) + margin_m
        width = right - left
        height = top - bottom
        if width > max_bbox_m or height > max_bbox_m:
            log.info(
                "Skipping region %s: bbox %.0fx%.0fm exceeds %.0fm limit",
                region,
                width,
                height,
                max_bbox_m,
            )
            continue
        try:
            bathy = _read_region_bathymetry(
                (left, bottom, right, top),
                cog_path,
                grid_size,
            )
        except Exception as exc:
            log.warning("Skipping region %s: COG read failed (%s)", region, exc)
            continue

        cell_metres = max(width, height) / grid_size
        deck = build_region_deck(region, pois, bathy, cell_metres)
        region_dir = output_dir / region
        region_dir.mkdir(parents=True, exist_ok=True)
        html_path = region_dir / "index.html"
        deck.to_html(
            str(html_path),
            iframe_height=800,
            notebook_display=False,
        )
        _inject_region_chrome(html_path, region, len(pois))
        generated.append((region, html_path, len(pois)))

    return generated


def write_viewer(
    rows: list[SiteRow],
    output_dir: Path,
    sites_dir: Path,
    grid_size: int = 100,
    cog_path: Path | None = None,
) -> ViewerSummary:
    """Write the overview plus a multi-metric per-site page tree."""
    # Wipe previous output so nested vs flat layouts don't collide.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Only sites with materialised bathymetry are browseable. Filter both
    # the overview map and the per-site loop against the same list so a
    # pin on the overview always corresponds to a real per-site page.
    renderable: list[SiteRow] = []
    skipped: list[tuple[str, str]] = []
    for row in rows:
        tif = sites_dir / row.slug / "bathymetry.tif"
        if tif.exists():
            renderable.append(row)
        else:
            skipped.append((row.slug, f"missing {tif}"))

    overview_path = output_dir / "index.html"
    build_overview_deck(renderable).to_html(
        str(overview_path),
        iframe_height=800,
        notebook_display=False,
    )
    _inject_overview_click_handler(overview_path)

    sites_out: list[tuple[str, Path, int]] = []
    for row in renderable:
        try:
            result = write_site_pages(row, sites_dir / row.slug, output_dir, grid_size)
        except Exception as exc:
            log.exception("Site render failed for %s", row.slug)
            skipped.append((row.slug, str(exc)))
            continue
        if result is None:
            skipped.append((row.slug, "no submerged cells after downsampling"))
            continue
        site_dir, cell_count = result
        sites_out.append((row.slug, site_dir, cell_count))

    # Regional drill-down pages — one per CSV region whose POIs cluster
    # tightly enough (≥3 POIs, bbox ≤4 km on each axis).
    try:
        regions_out = write_region_pages(
            renderable,
            output_dir,
            cog_path=cog_path,
        )
    except Exception as exc:
        log.exception("Region pages failed")
        regions_out = []
        skipped.append(("__regions__", str(exc)))

    if regions_out:
        _inject_overview_region_sidebar(overview_path, regions_out)

    return ViewerSummary(
        overview=overview_path,
        sites=sites_out,
        regions=regions_out,
        skipped=skipped,
    )


def _inject_overview_region_sidebar(
    overview_path: Path,
    regions: list[tuple[str, Path, int]],
) -> None:
    """Append a small sidebar panel listing regional drill-down links."""
    items = "".join(
        f'<li><a href="{name}/index.html">{name.replace("_", " ").title()}'
        f' <span class="count">({n})</span></a></li>'
        for name, _, n in sorted(regions)
    )
    sidebar = (
        '<div id="sotamar-regions">'
        "<h4>Regional 3D views</h4>"
        f"<ul>{items}</ul>"
        '<div class="hint">Drill into a cluster of nearby dive sites.</div>'
        "</div>"
    )
    css = """
<style>
  #sotamar-regions {
    position: absolute; top: 12px; right: 12px; z-index: 15;
    background: rgba(255,255,255,0.94); padding: 10px 14px;
    border-radius: 6px; font: 13px system-ui, sans-serif; color: #222;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3); max-width: 240px;
  }
  #sotamar-regions h4 { margin: 0 0 6px 0; font-size: 13px; color: #07304d; }
  #sotamar-regions ul { list-style: none; margin: 0; padding: 0; }
  #sotamar-regions li { margin: 3px 0; }
  #sotamar-regions a { color: #1f78b4; text-decoration: none; }
  #sotamar-regions a:hover { text-decoration: underline; }
  #sotamar-regions .count { color: #777; font-size: 11px; }
  #sotamar-regions .hint {
    margin-top: 8px; font-style: italic; color: #555; font-size: 11px;
  }
</style>
"""
    html = overview_path.read_text()
    html = html.replace("</head>", f"{css}</head>")
    html = html.replace("<body>", f"<body>{sidebar}")
    overview_path.write_text(html)
