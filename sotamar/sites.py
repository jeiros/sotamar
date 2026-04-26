"""Dive site registry: CSV-driven analysis windows + coordinate verification.

Every verified POI in `data/dive_sites.csv` becomes an analysis Site at
runtime. Site slugs equal POI ids (e.g. ``pal_boreas``, ``med_meda_gran``).
Window size (`half_size`) is determined by the POI's `site_type` via
`_HALF_SIZE_BY_TYPE`. To add or modify a site, edit the CSV — there is no
static Python registry.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from sotamar.pois import POI, load_pois


# Default analysis-window half-size by feature type. The window edge length
# is 2 × half_size in metres, so half_size=200 yields a 400 × 400 m window.
# Sized to match recreational dive footprints (100–300 m radius) while
# keeping ~50 m of edge context for the broad-BPI kernel.
_HALF_SIZE_BY_TYPE: dict[str, int] = {
    "wreck":    75,
    "pinnacle": 100,
    "cave":     100,
    "island":   150,
    "wall":     200,
    "cove":     200,
    "headland": 500,
}


@dataclass(frozen=True)
class Site:
    """A registered dive site with its extraction window parameters."""

    slug: str
    name: str
    easting: float
    northing: float
    region: str
    character: str
    half_size: int = 200
    transect: tuple[tuple[float, float], tuple[float, float]] | None = None
    description: str | None = None
    max_depth: float | None = None
    markers: tuple[tuple[float, float, str], ...] = field(default_factory=tuple)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (left, bottom, right, top) in EPSG:25831."""
        return (
            self.easting - self.half_size,
            self.northing - self.half_size,
            self.easting + self.half_size,
            self.northing + self.half_size,
        )

    @property
    def transect_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return custom transect if set, else default E-W through center."""
        if self.transect is not None:
            return self.transect
        margin = min(100, self.half_size // 4)
        return (
            (self.easting - self.half_size + margin, self.northing),
            (self.easting + self.half_size - margin, self.northing),
        )


# -- POI → Site conversion ----------------------------------------------------

def poi_to_site(poi: POI) -> Site:
    """Build a Site from a verified POI using the per-type half_size policy.

    Coordinates are snapped to a 100 m UTM grid (matches the CSV's
    declared verified precision and keeps figures stable across edits).
    Raises KeyError if the POI's site_type is unknown.
    """
    half_size = _HALF_SIZE_BY_TYPE[poi.site_type]
    easting = round(poi.easting / 100.0) * 100
    northing = round(poi.northing / 100.0) * 100
    character = (poi.description or poi.name)[:120]
    return Site(
        slug=poi.id,
        name=poi.name,
        easting=easting,
        northing=northing,
        region=poi.region,
        character=character,
        half_size=half_size,
        description=poi.description,
        max_depth=poi.depth_max_m,
    )


# -- Registry (CSV-backed, cached) --------------------------------------------

@functools.lru_cache(maxsize=1)
def _verified_sites() -> tuple[Site, ...]:
    """Load, filter, and convert every verified POI to a Site.

    Cached for the lifetime of the process; tests that mutate the CSV
    must call _verified_sites.cache_clear() between runs.
    """
    pois = load_pois()
    return tuple(
        poi_to_site(p) for p in pois if p.coord_confidence == "verified"
    )


def all_sites() -> list[Site]:
    """Return every Site derived from a verified CSV POI."""
    return list(_verified_sites())


def list_sites() -> list[str]:
    """Return every registered site slug (POI id)."""
    return [s.slug for s in _verified_sites()]


def get_site(slug: str) -> Site:
    """Look up a Site by slug (POI id). Raises KeyError if not found."""
    for s in _verified_sites():
        if s.slug == slug:
            return s
    raise KeyError(slug)


# -- Coordinate verification --------------------------------------------------

def verify_coordinates(site: Site) -> dict:
    """Geocode a site name and compare with registered UTM coordinates.

    Returns a dict with geocoded lat/lon, converted UTM, and distance in metres.
    Requires network access for Nominatim geocoding.
    """
    from geopy.geocoders import Nominatim
    from geopy.location import Location
    from pyproj import Transformer

    geolocator = Nominatim(user_agent="sotamar-coord-check/0.1", timeout=10)  # pyright: ignore[reportArgumentType]

    # Search with geographic context for better results
    query = f"{site.name}, Catalunya, Spain"
    try:
        location: Location | None = geolocator.geocode(query)  # pyright: ignore[reportAssignmentType]
    except Exception as exc:
        return {"site": site.slug, "error": f"Geocoding failed: {exc}"}
    if location is None:
        return {"site": site.slug, "error": f"Could not geocode '{query}'"}

    # Convert WGS84 → EPSG:25831
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
    geocoded_e, geocoded_n = transformer.transform(location.longitude, location.latitude)

    # Distance between registered and geocoded coordinates
    de = site.easting - geocoded_e
    dn = site.northing - geocoded_n
    distance = (de**2 + dn**2) ** 0.5

    return {
        "site": site.slug,
        "name": site.name,
        "geocoded_address": location.address,
        "geocoded_lat": location.latitude,
        "geocoded_lon": location.longitude,
        "geocoded_easting": round(geocoded_e, 1),
        "geocoded_northing": round(geocoded_n, 1),
        "registered_easting": site.easting,
        "registered_northing": site.northing,
        "delta_easting": round(de, 1),
        "delta_northing": round(dn, 1),
        "distance_m": round(distance, 1),
    }


def verify_all_coordinates() -> list[dict]:
    """Verify coordinates for all registered sites. Returns list of result dicts."""
    import time

    results = []
    for site in all_sites():
        result = verify_coordinates(site)
        results.append(result)
        time.sleep(1.1)  # Nominatim rate limit: 1 req/sec
    return results
