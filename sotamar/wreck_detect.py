"""Bathymetry-based wreck detection: find compact positive elevation
anomalies likely to be sunken vessels.

The methodology was first applied successfully to the Boreas wreck during
the project's coordinate-verification work — the hull, rising from an
otherwise smooth ~22 m sandy seabed, produced a clean ~4 m residual
(slightly below the raw relief, since the 301 m low-pass background
absorbs part of the anomaly's base). This module generalises that procedure for
the Costa Daurada wrecks whose exact GPS coordinates are deliberately
withheld to protect the dive sites.

Algorithm (per call to `detect_wrecks_near`):

1. Read a (2 × radius) × (2 × radius) window from the COG centred on
   the seed lat/lon.
2. Smooth with a 301 m uniform-box low-pass — preserves features ≤150 m.
3. Residual = bathymetry − smoothed.
4. Mask: residual ≥ `min_height_m` AND bathymetry < −3 m (submerged).
5. Connected-component labelling.
6. Per blob: footprint, peak residual, bounding-box length/width, elongation.
7. Filter to plausibility window: 10–2000 m², peak ≥ 0.5 m, elongation ≤ 8.
8. Sort descending by peak_residual × log10(footprint + 10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.ndimage import label, uniform_filter

from sotamar.io import find_cog

log = logging.getLogger(__name__)


_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
_to_wgs = Transformer.from_crs("EPSG:25831", "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class WreckCandidate:
    source_poi_id: str
    rank: int
    peak_lat: float
    peak_lon: float
    peak_easting: float
    peak_northing: float
    peak_residual_m: float
    seabed_depth_m: float
    footprint_m2: int
    length_m: int
    width_m: int
    plausibility: float

    def asdict(self) -> dict:
        return asdict(self)


# -- Detection ---------------------------------------------------------------

def detect_wrecks_near(
    seed_lat: float,
    seed_lon: float,
    cog_path: Path | None = None,
    search_radius_m: float = 1500,
    smooth_m: int = 301,
    min_height_m: float = 0.5,
    min_size_m2: int = 10,
    max_size_m2: int = 2000,
    max_elongation: float = 8.0,
    source_poi_id: str = "",
) -> tuple[list[WreckCandidate], dict]:
    """Search for wreck-shaped residual anomalies near a seed point.

    Returns (candidates, debug) where `debug` carries the raw arrays the
    plotter needs (bathymetry, residual, extent, seed coords) so callers
    don't have to re-read the COG.
    """
    cog = find_cog(cog_path)
    seed_e, seed_n = _to_utm.transform(seed_lon, seed_lat)
    win = from_bounds(
        seed_e - search_radius_m, seed_n - search_radius_m,
        seed_e + search_radius_m, seed_n + search_radius_m,
        transform=_open_transform(cog),
    )
    with rasterio.open(cog) as src:
        if src.crs is None or src.crs.to_epsg() != 25831:
            raise ValueError(f"{cog}: expected EPSG:25831, got {src.crs}")
        bathy = src.read(1, window=win, boundless=True,
                         fill_value=src.nodata).astype(np.float32)
        win_transform = src.window_transform(win)
        nodata = src.nodata

    # Mask NoData and emerged terrain
    invalid = (bathy == nodata) | (bathy > -3) | ~np.isfinite(bathy)
    bathy_clean = np.where(invalid, np.nan, bathy)

    # Low-pass smooth (fill NaN with mean for the smoothing pass)
    fill_value = float(np.nanmean(bathy_clean)) if np.isfinite(bathy_clean).any() else 0.0
    filled = np.where(invalid, fill_value, bathy_clean)
    smoothed = uniform_filter(filled, size=smooth_m)
    residual = bathy_clean - smoothed
    residual = np.where(invalid, np.nan, residual)

    # Candidate mask + connected components
    candidate_mask = (
        np.isfinite(residual)
        & (residual >= min_height_m)
        & (bathy_clean < -3)
    )
    labels, n_blobs = cast(tuple[np.ndarray, int], label(candidate_mask))

    candidates: list[WreckCandidate] = []
    for blob_id in range(1, n_blobs + 1):
        ys, xs = np.where(labels == blob_id)
        if ys.size == 0:
            continue
        size = int(ys.size)
        if size < min_size_m2 or size > max_size_m2:
            continue
        length = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        elongation = max(length, width) / max(min(length, width), 1)
        if elongation > max_elongation:
            continue

        # Peak of the blob
        blob_residual = np.where(labels == blob_id, residual, -np.inf)
        peak_y, peak_x = np.unravel_index(np.nanargmax(blob_residual), residual.shape)
        peak_residual = float(residual[peak_y, peak_x])
        if peak_residual < min_height_m:
            continue

        # Convert pixel → UTM → WGS84
        peak_easting, peak_northing = win_transform * (peak_x + 0.5, peak_y + 0.5)
        peak_lon, peak_lat = _to_wgs.transform(peak_easting, peak_northing)

        plausibility = float(peak_residual * np.log10(size + 10.0))
        candidates.append(WreckCandidate(
            source_poi_id=source_poi_id,
            rank=0,                   # set after sorting
            peak_lat=float(peak_lat),
            peak_lon=float(peak_lon),
            peak_easting=float(peak_easting),
            peak_northing=float(peak_northing),
            peak_residual_m=round(peak_residual, 2),
            seabed_depth_m=round(float(bathy_clean[peak_y, peak_x]), 2),
            footprint_m2=size,
            length_m=length,
            width_m=width,
            plausibility=round(plausibility, 2),
        ))

    candidates.sort(key=lambda c: -c.plausibility)
    candidates = [replace(c, rank=i + 1) for i, c in enumerate(candidates)]

    extent = (
        float(seed_e - search_radius_m), float(seed_e + search_radius_m),
        float(seed_n - search_radius_m), float(seed_n + search_radius_m),
    )
    debug = {
        "bathy": bathy_clean,
        "residual": residual,
        "extent_utm": extent,
        "seed_easting": float(seed_e),
        "seed_northing": float(seed_n),
    }
    return candidates, debug


def _open_transform(cog_path: Path):
    with rasterio.open(cog_path) as src:
        return src.transform


# -- Visualisation -----------------------------------------------------------

def plot_wreck_candidates(
    candidates: list[WreckCandidate],
    debug: dict,
    output_path: Path,
    source_name: str,
    seed_lat: float,
    seed_lon: float,
    radius_m: float,
) -> None:
    """Two-panel figure: bathymetry + residual, both with seed and candidates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bathy = debug["bathy"]
    residual = debug["residual"]
    cx = debug["seed_easting"]
    cy = debug["seed_northing"]

    # Display in offset metres from seed (as elsewhere in this project)
    extent = (-radius_m, radius_m, -radius_m, radius_m)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    ax = axes[0]
    if np.isfinite(bathy).any():
        v_lo = float(np.nanmin(bathy))
        v_hi = min(0.0, float(np.nanmax(bathy)))
        im = ax.imshow(bathy, cmap="viridis", extent=extent, origin="upper",
                       vmin=v_lo, vmax=v_hi, interpolation="none")
        plt.colorbar(im, ax=ax, label="Depth (m)", shrink=0.8)
    else:
        ax.text(0.5, 0.5, "No valid bathymetry", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_title("Bathymetry")
    ax.set_xlabel("E offset from seed (m)")
    ax.set_ylabel("N offset from seed (m)")

    ax = axes[1]
    if np.isfinite(residual).any():
        v = float(np.nanpercentile(np.abs(residual[np.isfinite(residual)]), 99))
        im = ax.imshow(residual, cmap="RdBu_r", extent=extent, origin="upper",
                       vmin=-v, vmax=v, interpolation="none")
        plt.colorbar(im, ax=ax, label="Residual elevation (m)", shrink=0.8)
    ax.set_title("Residual (bathymetry − 301 m smoothed)")
    ax.set_xlabel("E offset from seed (m)")

    # Mark seed + candidates on both panels
    for ax in axes:
        ax.plot(0, 0, marker="x", color="red", markersize=10, mew=2,
                label="Seed coord")
        for c in candidates[:10]:
            mx = c.peak_easting - cx
            my = c.peak_northing - cy
            ax.plot(mx, my, marker="o", markersize=8, mfc="none",
                    mec="black", mew=1.4)
            ax.annotate(
                f"#{c.rank}",
                (mx, my), xytext=(8, 8), textcoords="offset points",
                fontsize=8, color="black", fontweight="bold",
            )
        if candidates:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Wreck-detection candidates — {source_name}  "
        f"(seed: {seed_lat:.5f}°N, {seed_lon:.5f}°E, "
        f"radius {int(radius_m)} m)",
        fontsize=12, fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
