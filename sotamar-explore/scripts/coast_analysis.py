"""
Coast-wide bathymetric analysis of the Catalan coast.

Reads a coast overview at reduced resolution, then extracts and analyzes
six dive sites at full 1m resolution: slope, BPI (fine + broad), VRM.

Run: uv run python scripts/coast_analysis.py
"""

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.windows import from_bounds
from scipy import ndimage

# --- Configuration -----------------------------------------------------------

COG_URL = (
    "https://datacloud.icgc.cat/datacloud/batimetria/"
    "tif_unzip/batimetria-v2r1-elevacions-2021-2025.tif"
)

OUTPUT_DIR = Path("data/processed")
NODATA = -9999.0
HALF = 1000  # 1 km each direction → 2 km × 2 km windows

# Sites: (key, label, center_easting, center_northing)
SITES = [
    ("roses",     "Roses",             520100, 4677000),
    ("medes",     "Illes Medes",       518400, 4655100),
    ("formigues", "Illes Formigues",   515400, 4634500),
    ("tossa",     "Tossa de Mar",      495100, 4618900),
    ("garraf",    "Costa del Garraf",   403200, 4565300),
    ("salou",     "Cap de Salou",      344800, 4547200),
]

GDAL_ENV: dict[str, Any] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": True,
    "VSI_CACHE_SIZE": 10_000_000,
}


# --- Terrain metric functions ------------------------------------------------

def compute_slope(elev, mask):
    elev_f = elev.astype(np.float64)
    elev_f[mask] = np.nan
    dz_dy, dz_dx = np.gradient(elev_f, 1.0, 1.0)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope_deg[mask | np.isnan(slope_deg)] = NODATA
    return slope_deg.astype(np.float32)


def make_annular_mask(inner_radius, outer_radius):
    y, x = np.ogrid[-outer_radius:outer_radius + 1, -outer_radius:outer_radius + 1]
    dist = np.sqrt(x**2 + y**2)
    return (dist >= inner_radius) & (dist <= outer_radius)


def compute_bpi(elev, mask, inner_radius, outer_radius):
    elev_work = elev.astype(np.float64)
    elev_work[mask] = 0.0
    annulus = make_annular_mask(inner_radius, outer_radius).astype(np.float64)
    elev_sum = ndimage.convolve(elev_work, annulus, mode="constant", cval=0.0)
    valid = (~mask).astype(np.float64)
    valid_count = ndimage.convolve(valid, annulus, mode="constant", cval=0.0)
    valid_count = np.maximum(valid_count, 1.0)
    annular_mean = elev_sum / valid_count
    bpi = elev.astype(np.float64) - annular_mean
    bpi[mask] = NODATA
    return bpi.astype(np.float32)


def compute_vrm(elev, mask, window_size=3):
    elev_f = elev.astype(np.float64)
    elev_f[mask] = np.nan
    dz_dy, dz_dx = np.gradient(elev_f, 1.0, 1.0)
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
    vrm = 1.0 - (resultant / n)
    vrm[nan_mask] = NODATA
    return vrm.astype(np.float32)


# --- I/O helpers -------------------------------------------------------------

def save_raster(array, profile, filepath):
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=NODATA, compress="lzw")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(filepath, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def valid_stats(arr, nodata_val=NODATA):
    """Return dict of stats for valid (non-nodata, finite) pixels."""
    valid = arr[(arr != nodata_val) & np.isfinite(arr)]
    if valid.size == 0:
        return None
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "valid_pct": float(valid.size / arr.size * 100),
    }


# --- Site processing ---------------------------------------------------------

def process_site(src, key, label, cx, cy):
    """Extract window, compute all metrics, save GeoTIFFs, return stats."""
    left, bottom = cx - HALF, cy - HALF
    right, top = cx + HALF, cy + HALF

    # Check bounds are within dataset
    b = src.bounds
    if left < b.left or right > b.right or bottom < b.bottom or top > b.top:
        print(f"  WARNING: Window extends outside dataset bounds — clipping")
        left = max(left, b.left)
        bottom = max(bottom, b.bottom)
        right = min(right, b.right)
        top = min(top, b.top)

    window = from_bounds(left, bottom, right, top, transform=src.transform)
    data = src.read(1, window=window)
    win_transform = src.window_transform(window)
    mask = (data == src.nodata) if src.nodata is not None else np.isnan(data)

    nodata_pct = mask.sum() / mask.size * 100
    print(f"  Shape: {data.shape}, NoData: {nodata_pct:.1f}%")

    if nodata_pct > 95:
        print(f"  SKIPPING: insufficient data ({nodata_pct:.0f}% nodata)")
        return None

    # Build profile for saving
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": src.crs,
        "transform": win_transform,
        "nodata": NODATA,
        "compress": "lzw",
    }

    # Save bathymetry
    save_raster(data, profile, OUTPUT_DIR / f"{key}_bathy.tif")

    # Compute and save metrics
    t0 = time.time()
    slope = compute_slope(data, mask)
    save_raster(slope, profile, OUTPUT_DIR / f"{key}_slope.tif")
    print(f"  Slope: {time.time() - t0:.1f}s")

    t0 = time.time()
    bpi_fine = compute_bpi(data, mask, 3, 5)
    save_raster(bpi_fine, profile, OUTPUT_DIR / f"{key}_bpi_fine.tif")
    print(f"  Fine BPI: {time.time() - t0:.1f}s")

    t0 = time.time()
    bpi_broad = compute_bpi(data, mask, 25, 50)
    save_raster(bpi_broad, profile, OUTPUT_DIR / f"{key}_bpi_broad.tif")
    print(f"  Broad BPI: {time.time() - t0:.1f}s")

    t0 = time.time()
    vrm = compute_vrm(data, mask, window_size=3)
    save_raster(vrm, profile, OUTPUT_DIR / f"{key}_vrm.tif")
    print(f"  VRM: {time.time() - t0:.1f}s")

    # Collect stats
    stats = {
        "key": key,
        "label": label,
        "nodata_pct": round(nodata_pct, 1),
    }
    for name, arr in [("depth", data), ("slope", slope),
                      ("bpi_fine", bpi_fine), ("bpi_broad", bpi_broad),
                      ("vrm", vrm)]:
        s = valid_stats(arr)
        if s:
            for k, v in s.items():
                stats[f"{name}_{k}"] = round(v, 3)
    return stats


# --- Main --------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Coast overview
    print("=" * 60)
    print("PHASE 1: Coast overview (~64m resolution)")
    print("=" * 60)

    t0 = time.time()
    with Env(**GDAL_ENV):
        # overview_level=5 → 64× decimation → ~64m resolution
        with rasterio.open(COG_URL, overview_level=5) as src:
            overview = src.read(1)
            ov_profile = src.profile.copy()
            ov_nodata = src.nodata
            print(f"  Shape: {overview.shape}")
            print(f"  Resolution: {src.res}")
            mask = overview == ov_nodata
            valid = overview[~mask]
            print(f"  Valid pixels: {valid.size:,} ({valid.size / overview.size * 100:.1f}%)")
            if valid.size > 0:
                print(f"  Depth range: {valid.min():.1f} to {valid.max():.1f} m")

        save_raster(overview, ov_profile, OUTPUT_DIR / "coast_overview.tif")
        print(f"  Time: {time.time() - t0:.0f}s")

        # Also compute coast-wide slope at overview resolution
        print("\n  Computing coast-wide slope...")
        t0 = time.time()
        ov_elev = overview.astype(np.float64)
        ov_elev[mask] = np.nan
        res = ov_profile["transform"].a  # pixel size in meters
        dz_dy, dz_dx = np.gradient(ov_elev, res, res)
        ov_slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        ov_slope[mask | np.isnan(ov_slope)] = NODATA
        save_raster(ov_slope.astype(np.float32), ov_profile,
                    OUTPUT_DIR / "coast_overview_slope.tif")
        print(f"  Time: {time.time() - t0:.1f}s")

    # Phase 2: Per-site analysis
    print("\n" + "=" * 60)
    print("PHASE 2: Per-site analysis (1m resolution)")
    print("=" * 60)

    all_stats = []
    with Env(**GDAL_ENV):
        with rasterio.open(COG_URL) as src:
            for key, label, cx, cy in SITES:
                print(f"\n--- {label} ({key}) ---")
                t_site = time.time()

                # Skip if all files exist
                files = [OUTPUT_DIR / f"{key}_{m}.tif"
                         for m in ("bathy", "slope", "bpi_fine", "bpi_broad", "vrm")]
                if all(f.exists() for f in files):
                    print("  All files exist — loading stats from existing data")
                    bathy = rasterio.open(files[0]).read(1)
                    cached: dict[str, Any] = {"key": key, "label": label}
                    mask = bathy == NODATA
                    cached["nodata_pct"] = round(mask.sum() / mask.size * 100, 1)
                    for name, fp in zip(
                        ("depth", "slope", "bpi_fine", "bpi_broad", "vrm"), files
                    ):
                        with rasterio.open(fp) as f:
                            arr = f.read(1)
                        s = valid_stats(arr)
                        if s:
                            for k, v in s.items():
                                cached[f"{name}_{k}"] = round(v, 3)
                    all_stats.append(cached)
                    continue

                stats = process_site(src, key, label, cx, cy)
                if stats:
                    all_stats.append(stats)
                print(f"  Total site time: {time.time() - t_site:.1f}s")

    # Phase 3: Summary
    print("\n" + "=" * 60)
    print("PHASE 3: Comparative summary")
    print("=" * 60)

    if not all_stats:
        print("No sites with sufficient data!")
        return

    # Print table
    print(f"\n{'Site':<20} {'NoData%':>8} {'Depth range':>16} "
          f"{'Mean slope':>11} {'Mean VRM':>10} {'BPI broad σ':>12}")
    print("-" * 80)
    for s in all_stats:
        depth_range = f"{s.get('depth_min', '?'):.0f} to {s.get('depth_max', '?'):.0f}"
        print(f"{s['label']:<20} {s['nodata_pct']:>7.1f}% {depth_range:>16} "
              f"{s.get('slope_mean', 0):>10.1f}° {s.get('vrm_mean', 0):>10.4f} "
              f"{s.get('bpi_broad_std', 0):>11.2f} m")

    # Save as CSV
    csv_path = OUTPUT_DIR / "coast_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\nSaved: {csv_path}")

    print("\nDone! Next: uv run python scripts/coast_figures.py")


if __name__ == "__main__":
    main()
