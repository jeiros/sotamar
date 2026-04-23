"""
Tasks 4-6: Compute slope, BPI (fine + broad), and VRM from local bathymetry GeoTIFF.

Run: uv run python scripts/terrain_metrics.py
Requires: data/processed/medes_bathy.tif (from data_access.py)
"""

import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

INPUT_PATH = Path("data/processed/medes_bathy.tif")
OUTPUT_DIR = Path("data/processed")
NODATA_OUT = -9999.0


# --- I/O helpers -------------------------------------------------------------

def load_bathymetry():
    """Load the Medes bathymetry, return (data, nodata_mask, rasterio_profile)."""
    with rasterio.open(INPUT_PATH) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
    mask = (data == nodata) if nodata is not None else np.isnan(data)
    return data, mask, profile


def save_raster(array, profile, filename):
    """Save a 2D float32 array as a single-band GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=NODATA_OUT, compress="lzw")
    out_path = OUTPUT_DIR / filename
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved: {out_path} ({size_kb:.0f} KB)")


def print_stats(name, array, mask):
    """Print basic statistics for a metric array, ignoring nodata and NaN."""
    valid = array[~mask]
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        print(f"  {name}: no valid pixels")
        return
    print(f"  {name}: min={valid.min():.3f}, max={valid.max():.3f}, "
          f"mean={valid.mean():.3f}, std={valid.std():.3f}")


# --- Task 4: Slope -----------------------------------------------------------

def compute_slope(elev, mask):
    """Slope in degrees from elevation grid (dx=dy=1m)."""
    elev_f = elev.astype(np.float64)
    elev_f[mask] = np.nan

    dz_dy, dz_dx = np.gradient(elev_f, 1.0, 1.0)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    slope_deg[mask | np.isnan(slope_deg)] = NODATA_OUT
    return slope_deg.astype(np.float32)


# --- Task 5: BPI (Bathymetric Position Index) --------------------------------

def make_annular_mask(inner_radius, outer_radius):
    """Create a boolean annular kernel: True where inner_r <= dist <= outer_r."""
    y, x = np.ogrid[-outer_radius:outer_radius + 1, -outer_radius:outer_radius + 1]
    dist = np.sqrt(x**2 + y**2)
    return (dist >= inner_radius) & (dist <= outer_radius)


def compute_bpi(elev, mask, inner_radius, outer_radius):
    """BPI = elevation - annular_mean(elevation), nodata-aware."""
    elev_work = elev.astype(np.float64)
    elev_work[mask] = 0.0

    annulus = make_annular_mask(inner_radius, outer_radius).astype(np.float64)

    # Sum of elevation values in annulus
    elev_sum = ndimage.convolve(elev_work, annulus, mode="constant", cval=0.0)

    # Count of valid pixels in annulus
    valid = (~mask).astype(np.float64)
    valid_count = ndimage.convolve(valid, annulus, mode="constant", cval=0.0)
    valid_count = np.maximum(valid_count, 1.0)

    annular_mean = elev_sum / valid_count
    bpi = elev.astype(np.float64) - annular_mean

    bpi[mask] = NODATA_OUT
    return bpi.astype(np.float32)


# --- Task 6: VRM (Vector Ruggedness Measure) ---------------------------------

def compute_vrm(elev, mask, window_size=3):
    """VRM (Sappington et al. 2007): 1 - (|resultant| / n) in a moving window."""
    elev_f = elev.astype(np.float64)
    elev_f[mask] = np.nan

    # Slope and aspect
    dz_dy, dz_dx = np.gradient(elev_f, 1.0, 1.0)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dx, dz_dy)

    # 3D unit normal vectors
    sin_slope = np.sin(slope)
    x = sin_slope * np.sin(aspect)
    y = sin_slope * np.cos(aspect)
    z = np.cos(slope)

    # Zero out nodata and NaN (gradient propagates NaN to nodata neighbors)
    nan_mask = mask | np.isnan(x)
    x[nan_mask] = 0.0
    y[nan_mask] = 0.0
    z[nan_mask] = 0.0

    # Sum components in moving window via uniform_filter (separable, O(N))
    n = window_size**2
    x_sum = ndimage.uniform_filter(x, size=window_size, mode="constant", cval=0.0) * n
    y_sum = ndimage.uniform_filter(y, size=window_size, mode="constant", cval=0.0) * n
    z_sum = ndimage.uniform_filter(z, size=window_size, mode="constant", cval=0.0) * n

    resultant = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
    vrm = 1.0 - (resultant / n)

    vrm[nan_mask] = NODATA_OUT
    return vrm.astype(np.float32)


# --- Main --------------------------------------------------------------------

def main():
    print(f"Loading bathymetry from {INPUT_PATH}...")
    elev, mask, profile = load_bathymetry()
    print(f"  Shape: {elev.shape}, NoData pixels: {mask.sum()} ({mask.sum() / mask.size * 100:.1f}%)")

    # Slope
    print("\nComputing slope...")
    t0 = time.time()
    slope = compute_slope(elev, mask)
    print(f"  Time: {time.time() - t0:.1f}s")
    print_stats("Slope (deg)", slope, slope == NODATA_OUT)
    save_raster(slope, profile, "medes_slope.tif")

    # Fine BPI
    print("\nComputing fine BPI (inner=3, outer=5)...")
    t0 = time.time()
    bpi_fine = compute_bpi(elev, mask, inner_radius=3, outer_radius=5)
    print(f"  Time: {time.time() - t0:.1f}s")
    print_stats("Fine BPI", bpi_fine, bpi_fine == NODATA_OUT)
    save_raster(bpi_fine, profile, "medes_bpi_fine.tif")

    # Broad BPI
    print("\nComputing broad BPI (inner=25, outer=50)...")
    t0 = time.time()
    bpi_broad = compute_bpi(elev, mask, inner_radius=25, outer_radius=50)
    print(f"  Time: {time.time() - t0:.1f}s")
    print_stats("Broad BPI", bpi_broad, bpi_broad == NODATA_OUT)
    save_raster(bpi_broad, profile, "medes_bpi_broad.tif")

    # VRM
    print("\nComputing VRM (window=3)...")
    t0 = time.time()
    vrm = compute_vrm(elev, mask, window_size=3)
    print(f"  Time: {time.time() - t0:.1f}s")
    print_stats("VRM", vrm, vrm == NODATA_OUT)
    save_raster(vrm, profile, "medes_vrm.tif")

    print("\nAll terrain metrics computed.")
    print("Next step: uv run python scripts/visualization.py")


if __name__ == "__main__":
    main()
