"""Thesis-quality figure generation: terrain analysis panels + depth profiles."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from sotamar.profile import DIVE_THRESHOLDS

ZONE_COLORS = ["#a6cee3", "#1f78b4", "#08519c", "#000000"]
ZONE_LABELS = [
    "Zone 1: OWD (0 to \u221218 m)",
    "Zone 2: AOWD (\u221218 to \u221230 m)",
    "Zone 3: Deep / rec limit (\u221230 to \u221240 m)",
    "Zone 4: Technical (< \u221240 m)",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def plot_terrain_analysis(
    elevation: np.ndarray,
    slope: np.ndarray,
    bpi_fine: np.ndarray,
    bpi_broad: np.ndarray,
    vrm: np.ndarray,
    depth_zones: np.ndarray,
    bounds: tuple[float, float, float, float],
    site_name: str,
    output_dir: Path,
    markers: list[tuple[float, float, str]] | None = None,
) -> None:
    """Generate 6-panel terrain analysis figure (2x3 grid). Saves PNG + PDF.

    Axes show metres offset from the site centre, so distances read directly.
    Optional `markers` are (easting, northing, label) in EPSG:25831 and are
    overlaid on every panel.
    """
    left, bottom, right, top = bounds
    cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
    extent = (left - cx, right - cx, bottom - cy, top - cy)
    marker_pts = [(e - cx, n - cy, lbl) for (e, n, lbl) in (markers or [])]

    # (title, array, cmap, vmax_mode): "auto" | "symmetric" | "p99"
    datasets = [
        ("Depth (m)",              elevation, "viridis", "auto"),
        ("Slope (degrees)",        slope,     "YlOrRd",  "auto"),
        ("Fine BPI (r=3\u20135 m)",    bpi_fine,  "RdBu_r",  "symmetric"),
        ("Broad BPI (r=25\u201350 m)", bpi_broad, "RdBu_r",  "symmetric"),
        ("VRM (3\u00d73)",             vrm,       "inferno", "p99"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, (title, arr, cmap, mode) in enumerate(datasets):
        ax = axes_flat[idx]
        kw = dict(cmap=cmap, extent=extent, origin="upper", aspect="equal")
        if mode == "symmetric":
            vmax = float(np.nanpercentile(np.abs(arr), 99))
            im = ax.imshow(arr, vmin=-vmax, vmax=vmax, **kw)
        elif mode == "p99":
            vmax = float(np.nanpercentile(arr, 99))
            im = ax.imshow(arr, vmin=0.0, vmax=vmax, **kw)
        else:
            im = ax.imshow(arr, **kw)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("E offset (m)")
        ax.set_ylabel("N offset (m)")
        _apply_axis_ticks(ax)
        _draw_markers(ax, marker_pts)

    _plot_depth_zones_panel(axes_flat[5], depth_zones, extent, marker_pts)

    fig.suptitle(
        f"Bathymetric Terrain Analysis \u2014 {site_name}  "
        f"(centre: {cx:.0f} E, {cy:.0f} N, EPSG:25831)",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, "terrain_analysis", output_dir)
    plt.close(fig)


def _draw_markers(ax, marker_pts: list[tuple[float, float, str]]) -> None:
    """Overlay labelled markers on a panel (coords already in offset metres)."""
    for mx, my, lbl in marker_pts:
        ax.plot(mx, my, marker="+", color="red", markersize=12, mew=1.8,
                zorder=5)
        ax.annotate(
            lbl, (mx, my), xytext=(6, 6), textcoords="offset points",
            fontsize=8, color="red", fontweight="bold",
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=2, foreground="white"),
            ],
            zorder=6,
        )


def _plot_depth_zones_panel(
    ax, depth_zones: np.ndarray, extent: tuple[float, float, float, float],
    marker_pts: list[tuple[float, float, str]] | None = None,
) -> None:
    """Render the discrete depth-zones panel with legend."""
    cmap = ListedColormap(ZONE_COLORS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    ax.imshow(
        depth_zones, cmap=cmap, norm=norm, extent=extent,
        origin="upper", aspect="equal",
    )
    ax.set_title("Depth zones")
    ax.set_xlabel("E offset (m)")
    ax.set_ylabel("N offset (m)")
    handles = [Patch(color=c, label=lbl) for c, lbl in zip(ZONE_COLORS, ZONE_LABELS)]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9)
    _apply_axis_ticks(ax)
    _draw_markers(ax, marker_pts or [])


def _apply_axis_ticks(ax) -> None:
    """Limit tick count and rotate easting labels to prevent overlap."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")


def plot_depth_profile(
    distances: np.ndarray,
    depths: np.ndarray,
    site_name: str,
    output_dir: Path,
) -> bool:
    """Depth profile with dive threshold annotations. Saves PNG + PDF.

    Returns True if a figure was written, False if the transect has no
    valid depth data (all-NaN). Drawing the all-NaN case is skipped because
    matplotlib's tight-bbox PNG backend hangs on undefined data limits.
    """
    if not np.isfinite(depths).any():
        return False

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(distances, depths, "steelblue", linewidth=1.5)
    ax.fill_between(
        distances, depths, 0, alpha=0.15, color="steelblue",
        where=(~np.isnan(depths)) & (depths <= 0),  # pyright: ignore[reportArgumentType]
    )
    ax.axhline(0, color="black", linewidth=0.5)

    for depth_val, label, color in DIVE_THRESHOLDS:
        ax.axhline(depth_val, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(
            distances[-1] * 0.98, depth_val + 0.8, label,
            ha="right", fontsize=8, color=color,
        )

    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Depth Profile \u2014 {site_name}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, "depth_profile", output_dir)
    plt.close(fig)
    return True


def _make_extent(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert (left, bottom, right, top) to matplotlib imshow extent."""
    left, bottom, right, top = bounds
    return (left, right, bottom, top)


def _save_figure(fig: Figure, name: str, output_dir: Path) -> None:
    """Save figure as both PDF and PNG at 300 DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
