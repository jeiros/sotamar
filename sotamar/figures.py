"""Thesis-quality figure generation: terrain analysis panels + depth profiles."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sotamar.profile import DIVE_THRESHOLDS

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
    bounds: tuple[float, float, float, float],
    site_name: str,
    output_dir: Path,
) -> None:
    """Generate 5-panel terrain analysis figure (2x3 grid). Saves PNG + PDF."""
    extent = _make_extent(bounds)

    datasets = [
        ("Depth (m)", elevation, "viridis", False),
        ("Slope (degrees)", slope, "YlOrRd", False),
        ("Fine BPI (r=3\u20135 m)", bpi_fine, "RdBu_r", True),
        ("Broad BPI (r=25\u201350 m)", bpi_broad, "RdBu_r", True),
        ("VRM (3\u00d73)", vrm, "inferno", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, (title, arr, cmap, symmetric) in enumerate(datasets):
        ax = axes_flat[idx]
        if symmetric:
            vmax = np.nanpercentile(np.abs(arr), 99)
            im = ax.imshow(
                arr, cmap=cmap, extent=extent, origin="upper",
                aspect="equal", vmin=-vmax, vmax=vmax,
            )
        else:
            im = ax.imshow(
                arr, cmap=cmap, extent=extent, origin="upper",
                aspect="equal",
            )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    # Hide empty 6th subplot
    axes_flat[5].set_visible(False)

    fig.suptitle(
        f"Bathymetric Terrain Analysis \u2014 {site_name}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, "terrain_analysis", output_dir)
    plt.close(fig)


def plot_depth_profile(
    distances: np.ndarray,
    depths: np.ndarray,
    site_name: str,
    output_dir: Path,
) -> None:
    """Depth profile with dive threshold annotations. Saves PNG + PDF."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(distances, depths, "steelblue", linewidth=1.5)
    ax.fill_between(
        distances, depths, 0, alpha=0.15, color="steelblue",
        where=~np.isnan(depths),
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


def _make_extent(bounds: tuple[float, float, float, float]) -> list[float]:
    """Convert (left, bottom, right, top) to matplotlib imshow extent."""
    left, bottom, right, top = bounds
    return [left, right, bottom, top]


def _save_figure(fig: plt.Figure, name: str, output_dir: Path) -> None:
    """Save figure as both PDF and PNG at 300 DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
