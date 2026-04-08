"""Depth profile extraction along transect lines."""

from __future__ import annotations

import numpy as np
from rasterio.transform import rowcol

# Recreational diving depth thresholds (negative = submerged)
DIVE_THRESHOLDS = [
    (-12, "OWD (12 m)", "green"),
    (-18, "AOWD (18 m)", "orange"),
    (-30, "Deep (30 m)", "red"),
    (-40, "Rec. limit (40 m)", "darkred"),
]


def extract_depth_profile(
    elevation: np.ndarray,
    transform,
    start: tuple[float, float],
    end: tuple[float, float],
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract depth values along a transect line.

    Parameters
    ----------
    elevation : 2D array with NaN for nodata
    transform : rasterio Affine transform for the window
    start, end : (easting, northing) transect endpoints in EPSG:25831
    n_points : number of sample points

    Returns
    -------
    distances : 1D array of distances from start (metres)
    depths : 1D array of depth values (NaN where invalid)
    """
    xs = np.linspace(start[0], end[0], n_points)
    ys = np.linspace(start[1], end[1], n_points)

    rows, cols = rowcol(transform, xs, ys)
    rows = np.array(rows)
    cols = np.array(cols)

    # Clip to valid array bounds
    valid = (
        (rows >= 0)
        & (rows < elevation.shape[0])
        & (cols >= 0)
        & (cols < elevation.shape[1])
    )
    rows_v = rows[valid]
    cols_v = cols[valid]

    depths = np.full(n_points, np.nan, dtype=np.float64)
    depths[valid] = elevation[rows_v, cols_v].astype(np.float64)

    distances = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)

    return distances, depths
