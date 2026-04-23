"""Dive site catalog: dataclass registry with coordinate verification."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    easting=409200,
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

# -- Session 2 additions: expanded catalogue ---------------------------------
# Coordinates below are approximate — cross-check with check-coords and
# correct any that drift more than a few hundred metres from expected.

_register(Site(
    slug="cap_de_creus",
    name="Cap de Creus",
    easting=525800,
    northing=4688900,
    region="Costa Brava",
    character="Exposed granite headland, steep walls",
))

_register(Site(
    slug="massa_dor",
    name="Massa d'Or",
    easting=527100,
    northing=4687200,
    region="Costa Brava",
    character="Offshore pinnacle, high relief",
))

_register(Site(
    slug="el_gat",
    name="El Gat",
    easting=524300,
    northing=4690100,
    region="Costa Brava",
    character="Small rocky islet, vertical relief",
))

_register(Site(
    slug="els_farallons",
    name="Els Farallons",
    easting=525500,
    northing=4686600,
    region="Costa Brava",
    character="Twin rocks, swim-throughs",
))

_register(Site(
    slug="lescala_empuries",
    name="L'Escala – Empúries",
    easting=513000,
    northing=4663400,
    region="Costa Brava",
    character="Sandy shelf near Greco-Roman ruins",
))

_register(Site(
    slug="ullastres",
    name="Ullastres",
    easting=516700,
    northing=4637100,
    region="Costa Brava",
    character="Three pinnacles off Llafranc",
    description=(
        "Three submerged pinnacles (Ullastre I, II, III) at roughly 15–50 m "
        "depth off Llafranc. One of the best-known recreational dive sites "
        "on the Costa Brava."
    ),
    max_depth=50.0,
))

_register(Site(
    slug="els_canyers",
    name="Els Canyers",
    easting=510800,
    northing=4637200,
    region="Costa Brava",
    character="Rocky coastal drop-off near Palamós",
))

_register(Site(
    slug="cap_de_planes",
    name="Cap de Planes",
    easting=513000,
    northing=4636000,
    region="Costa Brava",
    character="Rocky cape near Palamós, moderate relief",
))

_register(Site(
    slug="boreas",
    name="Boreas (wreck)",
    easting=510000,
    northing=4631400,
    region="Costa Brava",
    character="40 m tugboat wreck, top ~22 m, seabed ~32 m, off Palamós",
    description=(
        "Wreck of the Boreas (ex-Pellworm, 40 m deep-sea tugboat) at "
        "approximately 41.83432° N, 3.12065° E, resting on a sandy "
        "seabed between ~22 m (wheelhouse) and ~32 m (propeller). "
        "Used to test whether individual wrecks are resolvable in "
        "the ICGC v2r1 1 m bathymetry."
    ),
    max_depth=32.0,
    markers=((510018, 4631388, "Boreas"),),
))

_register(Site(
    slug="garraf_falconera",
    name="La Falconera",
    easting=408500,
    northing=4567700,
    region="Costa del Garraf",
    character="Karstic cliffs with freshwater seeps",
    description=(
        "Submerged karstic spring in the Garraf massif where freshwater "
        "discharges into the Mediterranean through underwater cave systems."
    ),
))

_register(Site(
    slug="illa_de_la_plana",
    name="Illa de la Plana",
    easting=411900,
    northing=4562800,
    region="Costa del Garraf",
    character="Small rocky islet, moderate relief",
))

_register(Site(
    slug="el_biotop_torredembarra",
    name="El Biotop (Torredembarra)",
    easting=371800,
    northing=4552400,
    region="Costa Daurada",
    character="Artificial reef; bathymetry-gap case study",
    description=(
        "Artificial reef module near Torredembarra at approximately "
        "41.130° N, 1.430° E. Used as a case study of structures absent "
        "from the ICGC v2r1 bathymetry coverage."
    ),
    max_depth=20.0,
))

_register(Site(
    slug="lametlla_de_mar",
    name="L'Ametlla de Mar",
    easting=306700,
    northing=4531800,
    region="Costa Daurada",
    character="Rocky coastline with caves, near Ebre delta",
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
