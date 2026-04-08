"""Terrain metric computation: slope, hillshade, BPI, VRM.

All functions take (elevation, mask) arrays and return a float32 array
with NaN where nodata. Resolution is assumed 1 m x 1 m.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LightSource
from scipy import ndimage


def compute_slope(elevation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Slope in degrees from 1 m elevation grid (central differences)."""
    elev = elevation.astype(np.float64)
    elev[mask] = np.nan

    dz_dy, dz_dx = np.gradient(elev, 1.0, 1.0)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    slope_deg[mask | np.isnan(slope_deg)] = np.nan
    return slope_deg


def compute_hillshade(
    elevation: np.ndarray,
    mask: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    vert_exag: float = 2.0,
) -> np.ndarray:
    """Analytical hillshade via matplotlib LightSource. Returns [0, 1] float32."""
    filled = np.where(mask, 0.0, elevation.astype(np.float64))
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    hs = ls.hillshade(filled, vert_exag=vert_exag, dx=1, dy=1).astype(np.float32)
    hs[mask] = np.nan
    return hs


def compute_bpi(
    elevation: np.ndarray,
    mask: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """Bathymetric Position Index = z - annular_mean(z), nodata-aware."""
    elev_work = elevation.astype(np.float64)
    elev_work[mask] = 0.0

    annulus = _make_annular_kernel(inner_radius, outer_radius).astype(np.float64)

    elev_sum = ndimage.convolve(elev_work, annulus, mode="constant", cval=0.0)
    valid = (~mask).astype(np.float64)
    valid_count = ndimage.convolve(valid, annulus, mode="constant", cval=0.0)
    valid_count = np.maximum(valid_count, 1.0)

    annular_mean = elev_sum / valid_count
    bpi = (elevation.astype(np.float64) - annular_mean).astype(np.float32)

    bpi[mask] = np.nan
    return bpi


def compute_vrm(
    elevation: np.ndarray,
    mask: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Vector Ruggedness Measure (Sappington et al. 2007), 3x3 window."""
    elev = elevation.astype(np.float64)
    elev[mask] = np.nan

    dz_dy, dz_dx = np.gradient(elev, 1.0, 1.0)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dx, dz_dy)

    sin_slope = np.sin(slope)
    x = sin_slope * np.sin(aspect)
    y = sin_slope * np.cos(aspect)
    z = np.cos(slope)

    nan_mask = mask | np.isnan(x)
    x[nan_mask] = 0.0
    y[nan_mask] = 0.0
    z[nan_mask] = 0.0

    n = window_size**2
    x_sum = ndimage.uniform_filter(x, size=window_size, mode="constant", cval=0.0) * n
    y_sum = ndimage.uniform_filter(y, size=window_size, mode="constant", cval=0.0) * n
    z_sum = ndimage.uniform_filter(z, size=window_size, mode="constant", cval=0.0) * n

    resultant = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
    vrm = (1.0 - (resultant / n)).astype(np.float32)

    vrm[nan_mask] = np.nan
    return vrm


def _make_annular_kernel(inner_radius: int, outer_radius: int) -> np.ndarray:
    """Boolean annular kernel: True where inner_r <= dist <= outer_r."""
    y, x = np.ogrid[
        -outer_radius : outer_radius + 1,
        -outer_radius : outer_radius + 1,
    ]
    dist = np.sqrt(x**2 + y**2)
    return (dist >= inner_radius) & (dist <= outer_radius)
