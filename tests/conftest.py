"""Shared test fixtures: synthetic rasters and temporary GeoTIFFs."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from pathlib import Path

from sotamar.sites import Site


# -- Synthetic elevation surfaces ---------------------------------------------

@pytest.fixture
def flat_surface():
    """100x100 flat surface at -20 m, no nodata."""
    elev = np.full((100, 100), -20.0, dtype=np.float32)
    mask = np.zeros_like(elev, dtype=bool)
    return elev, mask


@pytest.fixture
def ramp_surface():
    """100x100 surface with a 45-degree E-W slope (rises 1 m per pixel eastward)."""
    cols = np.arange(100, dtype=np.float32)
    elev = np.broadcast_to(cols, (100, 100)).copy()
    mask = np.zeros_like(elev, dtype=bool)
    return elev, mask


@pytest.fixture
def peaked_surface():
    """100x100 surface with a Gaussian peak — high local relief."""
    y, x = np.mgrid[-50:50, -50:50].astype(np.float32)
    elev = 30.0 * np.exp(-(x**2 + y**2) / (2 * 15**2))
    elev -= 25.0  # shift so most is submerged
    mask = np.zeros_like(elev, dtype=bool)
    return elev, mask


@pytest.fixture
def surface_with_nodata():
    """100x100 surface with a 20-pixel nodata border on all sides."""
    elev = np.full((100, 100), -15.0, dtype=np.float32)
    mask = np.zeros_like(elev, dtype=bool)
    # Nodata border
    mask[:20, :] = True
    mask[-20:, :] = True
    mask[:, :20] = True
    mask[:, -20:] = True
    elev[mask] = np.nan
    return elev, mask


@pytest.fixture
def sample_site():
    """A test site with known parameters."""
    return Site(
        slug="test_site",
        name="Test Site",
        easting=500000.0,
        northing=4600000.0,
        region="Test Region",
        character="Synthetic test surface",
        half_size=50,
    )


@pytest.fixture
def sample_site_with_transect():
    """A test site with explicit transect."""
    return Site(
        slug="test_transect",
        name="Test Transect Site",
        easting=500000.0,
        northing=4600000.0,
        region="Test Region",
        character="Synthetic",
        half_size=50,
        transect=((499960, 4600000), (500040, 4600000)),
    )


# -- GeoTIFF helpers ----------------------------------------------------------

def make_test_profile(shape=(100, 100), origin=(500000.0, 4600050.0)):
    """Create a rasterio-compatible profile for a synthetic raster.

    Origin is the top-left corner (easting, northing) at 1 m resolution.
    """
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": "EPSG:25831",
        "transform": Affine(1.0, 0.0, origin[0], 0.0, -1.0, origin[1]),
        "nodata": -9999.0,
        "compress": "lzw",
    }


@pytest.fixture
def test_profile():
    """Rasterio profile for a 100x100 test raster."""
    return make_test_profile()


@pytest.fixture
def synthetic_cog(tmp_path, flat_surface):
    """Write a small synthetic GeoTIFF that acts as a COG for I/O tests."""
    path = tmp_path / "test_bathy.tif"
    elev, mask = flat_surface
    out = elev.copy()
    out[mask] = -9999.0

    profile = make_test_profile()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    return path


# -- DB fixtures (skip cleanly when PostGIS is not running) -------------------

@pytest.fixture(scope="session")
def db_url():
    """Return the PostGIS URL — env var wins, default points at compose."""
    import os
    return os.environ.get(
        "SOTAMAR_DB_URL",
        "postgresql+psycopg://sotamar:sotamar@localhost:5432/sotamar",
    )


@pytest.fixture(scope="session")
def db_engine(db_url):
    """Yield a SQLAlchemy engine; skip the test if PostGIS is unreachable."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(db_url, future=True)
    try:
        with engine.connect() as c:
            c.execute(sqlalchemy.text("SELECT 1"))
            c.execute(sqlalchemy.text("SELECT PostGIS_Version()"))
    except Exception as exc:
        pytest.skip(f"PostGIS not reachable at {db_url}: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture
def clean_db(db_engine):
    """Truncate the three catalogue tables before each integration test."""
    import sqlalchemy
    with db_engine.begin() as c:
        c.execute(sqlalchemy.text(
            "TRUNCATE site_rasters, site_terrain_stats, dive_sites "
            "RESTART IDENTITY CASCADE"
        ))
    yield db_engine


@pytest.fixture
def varied_cog(tmp_path, peaked_surface):
    """Write a synthetic COG with terrain variation for integration tests."""
    path = tmp_path / "varied_bathy.tif"
    elev, mask = peaked_surface
    out = elev.copy()
    out[mask] = -9999.0

    profile = make_test_profile()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    return path
