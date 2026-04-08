"""
Coast-wide visualizations: overview map, per-site hillshades, metric comparison.

Run: uv run python scripts/coast_figures.py
Requires: data/processed/ outputs from coast_analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
import numpy as np
import rasterio
from pathlib import Path

DATA_DIR = Path("data/processed")
FIG_DIR = Path("figures")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

NODATA = -9999.0

SITES = [
    ("roses",     "Roses",            520000, 4686500),
    ("medes",     "Illes Medes",      516800, 4655800),
    ("formigues", "Illes Formigues",  515500, 4636000),
    ("tossa",     "Tossa de Mar",     494500, 4619500),
    ("garraf",    "Costa del Garraf",  406500, 4565000),
    ("salou",     "Cap de Salou",     345500, 4547500),
]
HALF = 1000


# --- Helpers -----------------------------------------------------------------

def load(filename):
    with rasterio.open(DATA_DIR / filename) as src:
        return src.read(1), src.bounds, src.transform


def mask_nd(arr):
    return np.where(arr == NODATA, np.nan, arr)


def extent_of(bounds):
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  Saved: figures/{name}.pdf + .png")


# --- Figure 1: Coast overview with site markers ------------------------------

def plot_coast_overview():
    print("1. Coast overview map...")
    data, bounds, transform = load("coast_overview.tif")
    masked = mask_nd(data)

    # Compute hillshade for the overview
    filled = np.where(np.isnan(masked), 0, masked)
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(filled, vert_exag=3,
                             dx=transform.a, dy=-transform.e)
    hillshade = np.where(np.isnan(masked), np.nan, hillshade)

    fig, ax = plt.subplots(figsize=(8, 14))
    ext = extent_of(bounds)

    # Hillshade background
    ax.imshow(hillshade, cmap="gray", extent=ext, origin="upper",
              aspect="equal", vmin=0.2, vmax=1.0)
    # Bathymetry overlay with transparency
    im = ax.imshow(masked, cmap="viridis", extent=ext, origin="upper",
                   aspect="equal", alpha=0.6, vmin=-60, vmax=10)
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.4, pad=0.02)

    # Site markers
    colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]
    for i, (key, label, cx, cy) in enumerate(SITES):
        rect = Rectangle((cx - HALF, cy - HALF), 2 * HALF, 2 * HALF,
                          linewidth=2, edgecolor=colors[i], facecolor="none")
        ax.add_patch(rect)
        ax.annotate(label, (cx + HALF + 1500, cy),
                    fontsize=8, fontweight="bold", color=colors[i],
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i],
                              alpha=0.85))

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("ICGC Coastal Bathymetry — Catalan Coast\nStudy Sites",
                 fontsize=13, fontweight="bold")

    # Crop to area of interest (trim empty margins)
    ax.set_xlim(320000, 540000)
    ax.set_ylim(4530000, 4700000)

    fig.tight_layout()
    save(fig, "coast_overview")
    plt.close(fig)


# --- Figure 2: Per-site hillshade comparison ---------------------------------

def plot_site_hillshades():
    print("2. Per-site hillshade comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for i, (key, label, cx, cy) in enumerate(SITES):
        ax = axes_flat[i]
        filepath = DATA_DIR / f"{key}_bathy.tif"
        if not filepath.exists():
            ax.set_visible(False)
            continue

        data, bounds, _ = load(f"{key}_bathy.tif")
        masked = mask_nd(data)
        ext = extent_of(bounds)

        filled = np.where(np.isnan(masked), 0, masked)
        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(filled, vert_exag=2, dx=1, dy=1)
        hs = np.where(np.isnan(masked), np.nan, hs)

        ax.imshow(hs, cmap="gray", extent=ext, origin="upper", aspect="equal")

        # Overlay depth as semi-transparent color
        ax.imshow(masked, cmap="viridis", extent=ext, origin="upper",
                  aspect="equal", alpha=0.35, vmin=-50, vmax=10)

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("E (m)")
        ax.set_ylabel("N (m)")
        # Simplify tick labels (show relative coordinates)
        ax.ticklabel_format(useOffset=True, style="plain")

    fig.suptitle("Terrain Character — Six Dive Areas, Catalan Coast",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "coast_site_hillshades")
    plt.close(fig)


# --- Figure 3: Comparative box plots ----------------------------------------

def plot_metric_comparison():
    print("3. Metric comparison box plots...")

    metrics = [
        ("slope", "Slope (degrees)"),
        ("vrm", "VRM"),
        ("bpi_fine", "Fine BPI (m)"),
        ("bpi_broad", "Broad BPI (m)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    labels = []
    for key, label, _, _ in SITES:
        if (DATA_DIR / f"{key}_bathy.tif").exists():
            labels.append(label)

    for ax_idx, (metric, ylabel) in enumerate(metrics):
        ax = axes_flat[ax_idx]
        box_data = []

        for key, label, _, _ in SITES:
            filepath = DATA_DIR / f"{key}_{metric}.tif"
            if not filepath.exists():
                continue
            with rasterio.open(filepath) as src:
                arr = src.read(1).flatten()
            valid = arr[(arr != NODATA) & np.isfinite(arr)]
            # Subsample for box plot performance (max 50k points)
            if valid.size > 50000:
                valid = np.random.default_rng(42).choice(valid, 50000, replace=False)
            box_data.append(valid)

        bp = ax.boxplot(box_data, tick_labels=[l.replace(" ", "\n") for l in labels],
                        patch_artist=True, showfliers=False, widths=0.6)
        colors = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]
        for patch, color in zip(bp["boxes"], colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(ylabel, fontweight="bold")

    fig.suptitle("Terrain Metric Distributions — Catalan Coast Dive Sites",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "coast_metric_comparison")
    plt.close(fig)


# --- Figure 4: Multi-metric terrain panels per site --------------------------

def plot_site_terrain_panels():
    print("4. Per-site terrain panels...")
    # One row per site, columns: depth, slope, VRM
    n_sites = sum(1 for k, *_ in SITES if (DATA_DIR / f"{k}_bathy.tif").exists())
    fig, axes = plt.subplots(n_sites, 3, figsize=(15, 4 * n_sites))

    row = 0
    for key, label, cx, cy in SITES:
        if not (DATA_DIR / f"{key}_bathy.tif").exists():
            continue

        bathy, bounds, _ = load(f"{key}_bathy.tif")
        slope, _, _ = load(f"{key}_slope.tif")
        vrm, _, _ = load(f"{key}_vrm.tif")
        ext = extent_of(bounds)

        datasets = [
            (mask_nd(bathy), "viridis", "Depth (m)", False),
            (mask_nd(slope), "YlOrRd", "Slope (\u00b0)", False),
            (mask_nd(vrm), "inferno", "VRM", False),
        ]

        for col, (arr, cmap, clabel, sym) in enumerate(datasets):
            ax = axes[row, col] if n_sites > 1 else axes[col]
            im = ax.imshow(arr, cmap=cmap, extent=ext, origin="upper", aspect="equal")
            plt.colorbar(im, ax=ax, shrink=0.8)
            if col == 0:
                ax.set_ylabel(f"{label}\nNorthing (m)", fontweight="bold")
            else:
                ax.set_ylabel("Northing (m)")
            ax.set_xlabel("Easting (m)")
            if row == 0:
                ax.set_title(clabel, fontweight="bold")

        row += 1

    fig.suptitle("Terrain Metrics — Catalan Coast Dive Sites",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "coast_terrain_panels")
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating coast-wide figures...\n")
    plot_coast_overview()
    plot_site_hillshades()
    plot_metric_comparison()
    plot_site_terrain_panels()
    print("\nAll coast figures saved to figures/")


if __name__ == "__main__":
    main()
