"""
Tasks 7-9: Bathymetry visualization, multi-panel terrain analysis, depth profile.

Run: uv run python scripts/visualization.py
Requires: data/processed/medes_*.tif (from data_access.py + terrain_metrics.py)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import rasterio
from rasterio.transform import rowcol
from pathlib import Path

DATA_DIR = Path("data/processed")
FIG_DIR = Path("figures")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

NODATA = -9999.0


# --- Helpers -----------------------------------------------------------------

def load_raster(filename):
    """Load a single-band raster, return (data, bounds, transform)."""
    with rasterio.open(DATA_DIR / filename) as src:
        data = src.read(1)
        return data, src.bounds, src.transform


def mask_nodata(arr):
    """Replace nodata with NaN."""
    return np.where(arr == NODATA, np.nan, arr)


def make_extent(bounds):
    """Convert rasterio BoundingBox to matplotlib imshow extent."""
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def save_fig(fig, name):
    """Save figure as PDF + PNG."""
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved: figures/{name}.pdf + .png")


# --- Task 7: Bathymetry + hillshade -----------------------------------------

def plot_bathymetry():
    data, bounds, transform = load_raster("medes_bathy.tif")
    extent = make_extent(bounds)
    masked = mask_nodata(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Bathymetry with contours
    ax = axes[0]
    im = ax.imshow(masked, cmap="viridis", extent=extent, origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.8)

    # Contour lines at 5m intervals
    rows, cols = masked.shape
    x = np.linspace(bounds.left, bounds.right, cols)
    y = np.linspace(bounds.top, bounds.bottom, rows)
    depth_min, depth_max = np.nanmin(masked), np.nanmax(masked)
    levels = np.arange(np.ceil(depth_min / 5) * 5, depth_max, 5)
    cs = ax.contour(x, y, masked, levels=levels, colors="black", linewidths=0.3)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")

    ax.set_title("Bathymetry — Illes Medes")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Panel 2: Hillshade
    ax = axes[1]
    # Fill NaN for hillshade computation, then re-mask
    filled = np.where(np.isnan(masked), 0, masked)
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(filled, vert_exag=2, dx=1, dy=1)
    hillshade = np.where(np.isnan(masked), np.nan, hillshade)

    ax.imshow(hillshade, cmap="gray", extent=extent, origin="upper", aspect="equal")
    ax.set_title("Hillshade — Illes Medes")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    fig.tight_layout()
    save_fig(fig, "medes_bathymetry")
    plt.close(fig)


# --- Task 8: Multi-panel terrain analysis ------------------------------------

def plot_terrain_analysis():
    bathy, bounds, _ = load_raster("medes_bathy.tif")
    slope, _, _ = load_raster("medes_slope.tif")
    bpi_fine, _, _ = load_raster("medes_bpi_fine.tif")
    bpi_broad, _, _ = load_raster("medes_bpi_broad.tif")
    vrm, _, _ = load_raster("medes_vrm.tif")
    extent = make_extent(bounds)

    datasets = [
        ("Depth (m)", mask_nodata(bathy), "viridis", False),
        ("Slope (degrees)", mask_nodata(slope), "YlOrRd", False),
        ("Fine BPI (r=3–5 m)", mask_nodata(bpi_fine), "RdBu_r", True),
        ("Broad BPI (r=25–50 m)", mask_nodata(bpi_broad), "RdBu_r", True),
        ("VRM (3×3)", mask_nodata(vrm), "inferno", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, (title, arr, cmap, symmetric) in enumerate(datasets):
        ax = axes_flat[idx]
        if symmetric:
            vmax = np.nanpercentile(np.abs(arr), 99)
            im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper",
                           aspect="equal", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper",
                           aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    # Hide empty 6th subplot
    axes_flat[5].set_visible(False)

    fig.suptitle("Bathymetric Terrain Analysis — Illes Medes, Costa Brava",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "medes_terrain_analysis")
    plt.close(fig)


# --- Task 9: Depth profile --------------------------------------------------

def plot_depth_profile():
    data, bounds, transform = load_raster("medes_bathy.tif")

    # E-W transect crossing from deep water to coast, south of Medes Islands
    # Adjust these coordinates after inspecting the bathymetry figure
    start = (515900, 4655600)   # (easting, northing) — deep water (west)
    end = (517700, 4655600)     # coast/shallow (east)

    n_points = 500
    xs = np.linspace(start[0], end[0], n_points)
    ys = np.linspace(start[1], end[1], n_points)

    rows, cols = rowcol(transform, xs, ys)
    rows = np.array(rows)
    cols = np.array(cols)

    # Clip to valid array bounds
    valid = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    rows, cols, xs, ys = rows[valid], cols[valid], xs[valid], ys[valid]

    depths = data[rows, cols].astype(np.float64)
    depths[depths == NODATA] = np.nan
    distances = np.sqrt((xs - xs[0])**2 + (ys - ys[0])**2)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(distances, depths, "steelblue", linewidth=1.5)
    ax.fill_between(distances, depths, 0, alpha=0.15, color="steelblue",
                    where=~np.isnan(depths))
    ax.axhline(0, color="black", linewidth=0.5)

    # Dive certification thresholds (negative = submerged)
    thresholds = [
        (-12, "OWD (12 m)", "green"),
        (-18, "AOWD (18 m)", "orange"),
        (-30, "Deep (30 m)", "red"),
        (-40, "Rec. limit (40 m)", "darkred"),
    ]
    for depth_val, label, color in thresholds:
        ax.axhline(depth_val, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(distances[-1] * 0.98, depth_val + 0.8, label, ha="right",
                fontsize=8, color=color)

    ax.set_xlabel("Distance along transect (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Depth Profile — Illes Medes Transect")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "medes_depth_profile")
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main():
    print("Generating figures...\n")

    print("1. Bathymetry + hillshade...")
    plot_bathymetry()

    print("2. Multi-panel terrain analysis...")
    plot_terrain_analysis()

    print("3. Depth profile...")
    plot_depth_profile()

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
