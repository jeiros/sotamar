"""Dive site catalog: dataclass registry with coordinate verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    """A registered dive site with its extraction window parameters."""

    slug: str
    name: str
    easting: float
    northing: float
    region: str
    character: str
    half_size: int = 1000
    transect: tuple[tuple[float, float], tuple[float, float]] | None = None

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
        margin = 100
        return (
            (self.easting - self.half_size + margin, self.northing),
            (self.easting + self.half_size - margin, self.northing),
        )


# -- Registry -----------------------------------------------------------------

_SITES: dict[str, Site] = {}


def _register(site: Site) -> Site:
    _SITES[site.slug] = site
    return site


_register(Site(
    slug="roses",
    name="Roses",
    easting=520000,
    northing=4677000,
    region="Costa Brava",
    character="Rocky cape, steep cliffs",
))

_register(Site(
    slug="illes_medes",
    name="Illes Medes",
    easting=518400,
    northing=4655100,
    region="Costa Brava",
    character="Island MPA, walls & tunnels",
    transect=((517500, 4655100), (519300, 4655100)),
))

_register(Site(
    slug="illes_formigues",
    name="Illes Formigues",
    easting=515400,
    northing=4634500,
    region="Costa Brava",
    character="Rocky islets, gentle shelf",
))

_register(Site(
    slug="tossa_de_mar",
    name="Tossa de Mar",
    easting=495100,
    northing=4618900,
    region="Costa Brava",
    character="Medieval coast, steep cliffs",
))

_register(Site(
    slug="costa_del_garraf",
    name="Costa del Garraf",
    easting=403200,
    northing=4565300,
    region="Costa del Garraf",
    character="Smooth continental shelf",
))

_register(Site(
    slug="cap_de_salou",
    name="Cap de Salou",
    easting=344800,
    northing=4547200,
    region="Costa Daurada",
    character="Rocky cape, moderate relief",
))


# -- Accessors ----------------------------------------------------------------

def get_site(slug: str) -> Site:
    """Look up a site by slug. Raises KeyError if not found."""
    return _SITES[slug]


def list_sites() -> list[str]:
    """Return all registered site slugs."""
    return list(_SITES.keys())


def all_sites() -> list[Site]:
    """Return all registered Site objects."""
    return list(_SITES.values())


# -- Coordinate verification --------------------------------------------------

def verify_coordinates(site: Site) -> dict:
    """Geocode a site name and compare with registered UTM coordinates.

    Returns a dict with geocoded lat/lon, converted UTM, and distance in metres.
    Requires network access for Nominatim geocoding.
    """
    from geopy.geocoders import Nominatim
    from pyproj import Transformer

    geolocator = Nominatim(user_agent="sotamar-coord-check/0.1", timeout=10)

    # Search with geographic context for better results
    query = f"{site.name}, Catalunya, Spain"
    try:
        location = geolocator.geocode(query)
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
