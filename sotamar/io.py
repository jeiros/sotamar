"""Bathymetry I/O: COG windowed reads, GeoTIFF writing, statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

NODATA_GEOTIFF = -9999.0
DEFAULT_COG_PATH = Path("data/icgc/batimetria-v2r1-elevacions-2021-2025.tif")


def find_cog(cog_path: str | Path | None = None) -> Path:
    """Resolve the COG path. Raises FileNotFoundError if not found."""
    if cog_path is not None:
        p = Path(cog_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"COG not found at: {p}")

    # Try relative to CWD
    if DEFAULT_COG_PATH.exists():
        return DEFAULT_COG_PATH

    raise FileNotFoundError(
        f"COG not found at {DEFAULT_COG_PATH}. "
        "Download from ICGC or pass --cog <path>."
    )


def read_bathymetry_window(
    bounds: tuple[float, float, float, float],
    cog_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a rectangular window from the bathymetry COG.

    Parameters
    ----------
    bounds : (left, bottom, right, top) in EPSG:25831
    cog_path : override COG location

    Returns
    -------
    elevation : 2D float32 with NaN for nodata
    mask : boolean array (True where nodata)
    profile : rasterio-compatible dict for writing outputs
    """
    path = find_cog(cog_path)
    left, bottom, right, top = bounds

    with rasterio.open(path) as src:
        # Clip bounds to dataset extent
        b = src.bounds
        left = max(left, b.left)
        bottom = max(bottom, b.bottom)
        right = min(right, b.right)
        top = min(top, b.top)

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)

        nodata_val = src.nodata
        if nodata_val is not None:
            mask = data == nodata_val
        else:
            mask = np.isnan(data)

        # Convert to float32 with NaN for nodata
        elevation = data.astype(np.float32)
        elevation[mask] = np.nan

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "crs": src.crs,
            "transform": win_transform,
            "nodata": NODATA_GEOTIFF,
            "compress": "lzw",
        }

    return elevation, mask, profile


def save_geotiff(
    array: np.ndarray,
    profile: dict,
    path: Path,
) -> None:
    """Save a 2D array as single-band float32 LZW GeoTIFF."""
    out = array.astype(np.float32).copy()
    out[np.isnan(out)] = NODATA_GEOTIFF

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=NODATA_GEOTIFF, compress="lzw")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)


def compute_stats(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    """Compute summary statistics for multiple metric arrays.

    Returns dict with per-metric {min, max, mean, std}, overall nodata_pct,
    and vrm_pct_above_003 if VRM is present.
    """
    stats: dict = {
        "nodata_pct": round(float(mask.sum() / mask.size * 100), 2),
    }

    for name, arr in arrays.items():
        valid = arr[~mask & np.isfinite(arr)]
        if valid.size == 0:
            stats[name] = {"min": None, "max": None, "mean": None, "std": None}
            continue
        stats[name] = {
            "min": round(float(valid.min()), 3),
            "max": round(float(valid.max()), 3),
            "mean": round(float(valid.mean()), 3),
            "std": round(float(valid.std()), 3),
        }

    # VRM threshold metric
    if "vrm" in arrays:
        vrm = arrays["vrm"]
        valid_vrm = vrm[~mask & np.isfinite(vrm)]
        if valid_vrm.size > 0:
            stats["vrm_pct_above_003"] = round(
                float((valid_vrm > 0.003).sum() / valid_vrm.size * 100), 2
            )

    return stats


def compute_depth_zone_pcts(zones: np.ndarray) -> dict:
    """Percentages of submerged area in each dive zone (NaN pixels excluded).

    Returns dict with owd_pct, aowd_pct, deep_pct, tech_pct. If the zones
    array is entirely NaN, all values are None.
    """
    keys = ("owd_pct", "aowd_pct", "deep_pct", "tech_pct")
    valid = zones[~np.isnan(zones)]
    if valid.size == 0:
        return {k: None for k in keys}
    total = valid.size
    return {
        "owd_pct":  round(float((valid == 1).sum() / total * 100), 2),
        "aowd_pct": round(float((valid == 2).sum() / total * 100), 2),
        "deep_pct": round(float((valid == 3).sum() / total * 100), 2),
        "tech_pct": round(float((valid == 4).sum() / total * 100), 2),
    }


def save_stats(stats: dict, path: Path) -> None:
    """Write stats dict as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
