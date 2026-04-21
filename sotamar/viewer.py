"""Static HTML viewer: overview map + per-site 3D seabed via pydeck/deck.gl."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydeck as pdk
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling

from sotamar.db import SiteRow
from sotamar.figures import ZONE_COLORS

log = logging.getLogger(__name__)

# CARTO light basemap — no API token, matches pydeck's default when map_style
# is not set. Explicit here so the generated HTML is reproducible.
BASEMAP_STYLE = "light"

# Vertical exaggeration for per-site 3D views. Bathymetric relief (metres) is
# tiny next to site widths (kilometres), so a multiplier is standard practice
# in GIS. 5× renders a 60 m depth range as 300 m visual height — legible at
# zoom 14 with pitch 50° without looking cartoonish.
VERTICAL_EXAGGERATION = 5.0

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


def _dominant_zone(stats: dict) -> int:
    """Return the 1..4 index of the dive zone holding the largest area %.

    Defaults to zone 1 (OWD) when no zone percentages are available.
    """
    pcts = [stats.get(f"zone_{k}_pct") or 0.0 for k in ZONE_KEYS]
    if max(pcts) <= 0:
        return 1
    return int(np.argmax(pcts)) + 1


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
        "href": f"{row.slug}.html",
    }


def build_overview_deck(rows: list[SiteRow]) -> pdk.Deck:
    """Build the overview map: 18 clickable site markers on a CARTO basemap."""
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
        "style": {"backgroundColor": "white", "color": "#222",
                  "fontFamily": "system-ui, sans-serif", "fontSize": "12px"},
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style=BASEMAP_STYLE,
        tooltip=tooltip,
    )


# Injected at the bottom of the overview HTML so clicking a marker opens
# its per-site page. The pydeck template names the instance `deckInstance`
# — stable across pydeck 0.9.x.
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
    """Post-process pydeck's static HTML to make markers click-navigate."""
    html = html_path.read_text()
    if "deckInstance" not in html:
        raise RuntimeError(
            f"{html_path}: pydeck template no longer exposes 'deckInstance'"
        )
    html = html.replace("</html>", f"{_OVERVIEW_CLICK_SCRIPT}</html>")
    html_path.write_text(html)


# -- Per-site 3D view ---------------------------------------------------------

def _zone_from_depth(depth: float) -> int:
    """Map a depth in metres to a 1..4 zone (matches terrain.compute_depth_zones)."""
    if depth <= -40.0:
        return 4
    if depth <= -30.0:
        return 3
    if depth <= -18.0:
        return 2
    return 1


def downsample_bathymetry(
    tif_path: Path, grid_size: int = 50,
) -> list[dict]:
    """Read the bathymetry GeoTIFF, resample to grid_size × grid_size, and
    return a list of records with WGS84 coordinates, depth, and zone.

    Emerged (depth > 0) and NoData pixels are excluded from the output.
    """
    with rasterio.open(tif_path) as src:
        if src.crs is None or src.crs.to_epsg() != 25831:
            raise ValueError(
                f"{tif_path}: expected EPSG:25831, got {src.crs}"
            )
        nodata = src.nodata
        data = src.read(
            1, out_shape=(grid_size, grid_size),
            resampling=Resampling.average,
        )
        left, bottom, right, top = src.bounds

    dx = (right - left) / grid_size
    dy = (top - bottom) / grid_size

    records: list[dict] = []
    for r in range(grid_size):
        northing = top - (r + 0.5) * dy
        for c in range(grid_size):
            z = float(data[r, c])
            if nodata is not None and z == nodata:
                continue
            if not np.isfinite(z) or z > 0.0:
                continue
            easting = left + (c + 0.5) * dx
            lon, lat = _utm_to_wgs84.transform(easting, northing)
            zone = _zone_from_depth(z)
            records.append({
                "lon": lon,
                "lat": lat,
                "depth": round(z, 2),
                "zone": zone,
                "zone_label": ZONE_LABEL_BY_NUM[zone],
                "color": ZONE_RGB[zone - 1],
            })
    return records


def build_site_deck(
    site_row: SiteRow, records: list[dict], cell_metres: float,
) -> pdk.Deck:
    """Build the per-site 3D view: ColumnLayer + floating label.

    No basemap: the CARTO land map makes the Mediterranean show as featureless
    grey and makes the 3D columns look like they sit on top of land. We
    render on an ocean-blue canvas (see _inject_ocean_background) with
    native negative depths so the seabed extrudes downward from the
    (implicit) sea surface — a proper bathymetric viewpoint.
    """
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=records,
        get_position=["lon", "lat"],
        get_elevation="depth",  # depth is negative → columns extrude downward
        elevation_scale=VERTICAL_EXAGGERATION,
        radius=cell_metres / 2,
        get_fill_color="color",
        pickable=True,
        extruded=True,
        auto_highlight=True,
    )

    label_layer = pdk.Layer(
        "TextLayer",
        data=[{"lon": site_row.lon, "lat": site_row.lat, "text": site_row.name}],
        get_position=["lon", "lat"],
        get_text="text",
        get_size=18,
        get_color=[240, 240, 240],
        get_alignment_baseline="'bottom'",
        billboard=True,
    )

    view = pdk.ViewState(
        longitude=site_row.lon,
        latitude=site_row.lat,
        zoom=14,
        pitch=55,
        bearing=0,
    )

    tooltip = {
        "html": (
            "<b>Depth:</b> {depth} m<br/>"
            "<b>{zone_label}</b><br/>"
            f"<i style='color:#666;'>×{VERTICAL_EXAGGERATION:g} vertical "
            "exaggeration</i>"
        ),
        "style": {"backgroundColor": "white", "color": "#222",
                  "fontFamily": "system-ui, sans-serif", "fontSize": "12px"},
    }

    return pdk.Deck(
        layers=[column_layer, label_layer],
        initial_view_state=view,
        map_style=None,
        map_provider=None,  # both are required to fully disable the basemap
        tooltip=tooltip,
    )


# Injected into per-site HTML so the canvas has an ocean-blue background
# instead of deck.gl's default black. #0a3d62 reads as deep Mediterranean.
_SITE_BACKGROUND_STYLE = """
<style>
  body, #deck-container { background: #0a3d62 !important; }
  #deck-container canvas { background: #0a3d62 !important; }
</style>
"""


def _inject_ocean_background(html_path: Path) -> None:
    """Give per-site HTML an ocean-blue canvas background."""
    html = html_path.read_text()
    html = html.replace("</head>", f"{_SITE_BACKGROUND_STYLE}</head>")
    html_path.write_text(html)


# -- Orchestration ------------------------------------------------------------

@dataclass
class ViewerSummary:
    overview: Path
    sites: list[tuple[str, Path, int]]  # (slug, path, cell_count)
    skipped: list[tuple[str, str]]


def write_viewer(
    rows: list[SiteRow],
    output_dir: Path,
    sites_dir: Path,
    grid_size: int = 50,
) -> ViewerSummary:
    """Write the overview HTML and per-site 3D HTML for every row with a
    bathymetry raster on disk. Returns a summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

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
            records = downsample_bathymetry(tif, grid_size=grid_size)
        except Exception as exc:
            log.exception("Downsample failed for %s", row.slug)
            skipped.append((row.slug, str(exc)))
            continue
        if not records:
            skipped.append((row.slug, "no submerged cells after downsampling"))
            continue

        # Cell edge length in metres (site bbox is 2 * half_size in each axis,
        # but we don't have half_size on SiteRow; derive from window_size).
        cell_metres = row.window_size / grid_size

        site_path = output_dir / f"{row.slug}.html"
        build_site_deck(row, records, cell_metres).to_html(
            str(site_path), iframe_height=800, notebook_display=False,
        )
        _inject_ocean_background(site_path)
        sites_out.append((row.slug, site_path, len(records)))

    return ViewerSummary(
        overview=overview_path, sites=sites_out, skipped=skipped,
    )
