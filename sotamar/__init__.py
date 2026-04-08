"""SotaMar: Bathymetric terrain analysis for Catalan coast dive sites."""

__version__ = "0.1.0"

from sotamar.sites import Site, get_site, list_sites, all_sites
from sotamar.io import read_bathymetry_window, save_geotiff
from sotamar.terrain import compute_slope, compute_hillshade, compute_bpi, compute_vrm
from sotamar.profile import extract_depth_profile
