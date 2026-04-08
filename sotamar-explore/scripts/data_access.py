"""
Task 1-3: Discover ICGC bathymetry COG, read metadata, extract Illes Medes window.

Run: uv run python scripts/data_access.py
"""

import re
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.windows import from_bounds

# --- Configuration -----------------------------------------------------------
# Override this if you already know the URL or have a local file:
COG_URL = "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/batimetria-v2r1-elevacions-2021-2025.tif"

# Illes Medes extraction window (EPSG:25831 UTM coordinates)
MEDES_LEFT = 515800
MEDES_BOTTOM = 4654800
MEDES_RIGHT = 517800
MEDES_TOP = 4656800

OUTPUT_PATH = Path("data/processed/medes_bathy.tif")


# --- Task 1: Discover COG URL -----------------------------------------------

CANDIDATE_URLS = [
    # Most likely patterns based on other ICGC products
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/batimetria-litoral-v2r1-2021-2025.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/batimetria-litoral-v2r0-2021-2025.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/batimetria-litoral-v1r0-2021-2025.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/batimetria-litoral-v2r1-2024.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/batimetria-litoral-v2r1-2023.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/batimetria-litoral-v2r1-2022.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/batimetria-v2r1-2021-2025.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/batimetria-v2r0-2021-2025.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/batimetria-v1r0-2021-2025.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/batimetria-v2r1-2024.tif",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/batimetria-v2r1-2023.tif",
    "https://datacloud.icgc.cat/datacloud/elevacions-batimetria/tif_unzip/elevacions-batimetria-v2r1-2021-2025.tif",
]

DIRECTORY_URLS = [
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif_unzip/",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/tif/",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif_unzip/",
    "https://datacloud.icgc.cat/datacloud/batimetria/tif/",
    "https://datacloud.icgc.cat/datacloud/elevacions-batimetria/tif_unzip/",
    "https://datacloud.icgc.cat/datacloud/batimetria-litoral/",
    "https://datacloud.icgc.cat/datacloud/batimetria/",
]

WMS_CAPABILITIES_URL = (
    "https://geoserveis.icgc.cat/servei/catalunya/batimetria/wms"
    "?service=WMS&request=GetCapabilities"
)


def try_head(url, timeout=10):
    """Send a HEAD request; return response if 200, else None."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "SotaMar-Explorer/1.0")
        resp = urllib.request.urlopen(req, timeout=timeout)
        if resp.status == 200:
            return resp
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        pass
    return None


def try_get(url, timeout=15):
    """GET request; return response body as string if 200, else None."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "SotaMar-Explorer/1.0")
        resp = urllib.request.urlopen(req, timeout=timeout)
        if resp.status == 200:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        pass
    return None


def discover_cog_url():
    """Try to find the ICGC bathymetry COG URL."""

    # Phase 1: Probe known URL patterns
    print("Phase 1: Probing candidate URLs...")
    for url in CANDIDATE_URLS:
        print(f"  Trying: {url}")
        resp = try_head(url)
        if resp:
            size = resp.headers.get("Content-Length", "unknown")
            print(f"  >>> FOUND! Content-Length: {size}")
            return url

    # Phase 2: Browse directory listings
    print("\nPhase 2: Browsing directory listings...")
    for dir_url in DIRECTORY_URLS:
        print(f"  Trying: {dir_url}")
        html = try_get(dir_url)
        if html:
            tif_files = re.findall(r'href="([^"]*\.tif)"', html)
            if tif_files:
                found = dir_url + tif_files[0]
                print(f"  >>> FOUND via directory listing: {found}")
                return found
            # Also look for subdirectories that might contain tifs
            subdirs = re.findall(r'href="([^"]*/)"\s*>', html)
            print(f"  Directory exists, subdirs: {subdirs[:10]}")

    # Phase 3: Parse WMS GetCapabilities
    print("\nPhase 3: Checking WMS GetCapabilities...")
    print(f"  Fetching: {WMS_CAPABILITIES_URL}")
    xml = try_get(WMS_CAPABILITIES_URL, timeout=20)
    if xml:
        # Look for OnlineResource, DataURL, or file path hints
        data_urls = re.findall(r'<OnlineResource[^>]*href="([^"]*)"', xml)
        file_refs = re.findall(r'(datacloud[^"<\s]*\.tif)', xml)
        print(f"  OnlineResource URLs found: {len(data_urls)}")
        for u in data_urls[:5]:
            print(f"    {u}")
        if file_refs:
            print(f"  TIF references: {file_refs}")
            return "https://" + file_refs[0] if not file_refs[0].startswith("http") else file_refs[0]
        # Print a snippet of the XML for debugging
        print(f"  XML length: {len(xml)} chars")
        # Look for layer names that hint at the data
        layers = re.findall(r'<Name>([^<]*)</Name>', xml)
        print(f"  Layer names: {layers[:10]}")
    else:
        print("  Could not fetch WMS capabilities")

    # Phase 4: Fail gracefully
    print("\n" + "=" * 60)
    print("Could not discover COG URL automatically.")
    print("Options:")
    print("  1. Set COG_URL at the top of this script to the correct URL")
    print("  2. Download manually from visors.icgc.cat/appdownloads/")
    print("     and place the file in data/raw/")
    print("  3. Set COG_URL = 'data/raw/<filename>.tif'")
    print("=" * 60)
    return None


# --- Task 2: Read and print metadata ----------------------------------------

def print_metadata(src):
    """Print comprehensive raster metadata."""
    print("\n--- Raster Metadata ---")
    print(f"  Driver:       {src.driver}")
    print(f"  CRS:          {src.crs}")
    print(f"  Bounds:       {src.bounds}")
    print(f"  Resolution:   {src.res} (x, y)")
    print(f"  Shape:        {src.shape} (height, width)")
    print(f"  Dtype:        {src.dtypes}")
    print(f"  NoData:       {src.nodata}")
    print(f"  Band count:   {src.count}")
    print(f"  Transform:    {src.transform}")
    print(f"  Compression:  {src.compression}")
    overviews = src.overviews(1)
    print(f"  Overviews:    {overviews} ({len(overviews)} levels)")


# --- Task 3: Extract Medes window and save -----------------------------------

def extract_medes(src):
    """Read 2km×2km window around Illes Medes and save as local GeoTIFF."""
    window = from_bounds(
        MEDES_LEFT, MEDES_BOTTOM, MEDES_RIGHT, MEDES_TOP,
        transform=src.transform,
    )
    data = src.read(1, window=window)
    win_transform = src.window_transform(window)

    # Mask nodata
    nodata_val = src.nodata
    if nodata_val is not None:
        mask = data == nodata_val
    else:
        mask = np.isnan(data)

    valid = data[~mask]

    print("\n--- Illes Medes Window Statistics ---")
    print(f"  Window:    col_off={window.col_off:.0f}, row_off={window.row_off:.0f}, "
          f"width={window.width:.0f}, height={window.height:.0f}")
    print(f"  Shape:     {data.shape}")
    print(f"  Dtype:     {data.dtype}")
    if valid.size > 0:
        print(f"  Min:       {valid.min():.2f} m")
        print(f"  Max:       {valid.max():.2f} m")
        print(f"  Mean:      {valid.mean():.2f} m")
        print(f"  Std:       {valid.std():.2f} m")
    print(f"  NoData:    {mask.sum()} pixels ({mask.sum() / data.size * 100:.1f}%)")

    # Sanity check
    if valid.size > 0 and (valid.min() < -100 or valid.max() > 50):
        print("  WARNING: Values outside expected range [-100, +50] m")

    # Save local GeoTIFF
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": src.crs,
        "transform": win_transform,
        "nodata": nodata_val,
        "compress": "lzw",
    }
    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(data, 1)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n  Saved: {OUTPUT_PATH} ({size_kb:.0f} KB)")
    return data


# --- Main --------------------------------------------------------------------

def main():
    # Task 1: Discover or use provided URL
    url = COG_URL
    if url is None:
        url = discover_cog_url()
    if url is None:
        return

    print(f"\nUsing COG source: {url}")

    # Determine if local or remote
    is_remote = url.startswith("http://") or url.startswith("https://")

    # Task 2 & 3: Open raster and work with it
    gdal_env = {}
    if is_remote:
        gdal_env = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "VSI_CACHE": True,
            "VSI_CACHE_SIZE": 5_000_000,
        }

    with Env(**gdal_env):
        # Try opening directly first, then with /vsicurl/ prefix
        raster_path = url
        try:
            src = rasterio.open(raster_path)
        except Exception:
            if is_remote:
                print("Direct open failed, trying /vsicurl/ prefix...")
                raster_path = f"/vsicurl/{url}"
                src = rasterio.open(raster_path)
            else:
                raise

        with src:
            print_metadata(src)

            # Quick sanity read: 100×100 pixel test window
            print("\n--- Test read (100×100 pixels from top-left) ---")
            test_data = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
            print(f"  Shape: {test_data.shape}, dtype: {test_data.dtype}")
            print(f"  Sample values: {test_data[50, 40:50]}")

            # Task 3: Extract Medes
            extract_medes(src)

    print("\nDone! Next step: uv run python scripts/terrain_metrics.py")


if __name__ == "__main__":
    main()
