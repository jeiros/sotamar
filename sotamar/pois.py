"""POI catalogue: CSV-driven points of interest along the Catalan coast.

POIs (points of interest) are named features — wrecks, pinnacles, coves,
islets — sourced from `data/dive_sites.csv`. They're distinct from `Site`
analysis windows: a single Site (e.g. `illes_medes`) may contain dozens
of POIs that get auto-overlaid as markers on its terrain figures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pyproj import Transformer


DEFAULT_CSV_PATH = Path("data/dive_sites.csv")
VALID_CONFIDENCE = {"verified", "approximate", "unverified"}

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)


@dataclass(frozen=True)
class POI:
    """One named feature from the CSV catalogue."""

    id: str
    name: str
    region: str
    municipality: str | None
    site_type: str
    latitude: float
    longitude: float
    coord_confidence: str
    depth_min_m: float | None
    depth_max_m: float | None
    description: str | None
    sources: str | None

    @property
    def easting(self) -> float:
        e, _ = _to_utm.transform(self.longitude, self.latitude)
        return e

    @property
    def northing(self) -> float:
        _, n = _to_utm.transform(self.longitude, self.latitude)
        return n


def _opt_str(value: str) -> str | None:
    return value.strip() or None


def _opt_float(value: str) -> float | None:
    s = value.strip()
    return float(s) if s else None


def load_pois(path: Path = DEFAULT_CSV_PATH) -> list[POI]:
    """Read POIs from the CSV. Raises FileNotFoundError if missing."""
    pois: list[POI] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            confidence = row["coord_confidence"].strip()
            if confidence not in VALID_CONFIDENCE:
                raise ValueError(
                    f"{path}: row id={row['id']!r} has invalid "
                    f"coord_confidence={confidence!r}"
                )
            pois.append(POI(
                id=row["id"].strip(),
                name=row["name"].strip(),
                region=row["region"].strip(),
                municipality=_opt_str(row["municipality"]),
                site_type=row["site_type"].strip(),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                coord_confidence=confidence,
                depth_min_m=_opt_float(row["depth_min_m"]),
                depth_max_m=_opt_float(row["depth_max_m"]),
                description=_opt_str(row["description"]),
                sources=_opt_str(row["sources"]),
            ))
    return pois


def pois_in_bounds(
    pois: Iterable[POI],
    bounds: tuple[float, float, float, float],
) -> list[POI]:
    """Return POIs whose UTM (easting, northing) falls inside `bounds`.

    `bounds` is (left, bottom, right, top) in EPSG:25831, matching the
    Site.bounds convention.
    """
    left, bottom, right, top = bounds
    return [
        p for p in pois
        if left <= p.easting <= right and bottom <= p.northing <= top
    ]


def pois_to_markers(pois: Iterable[POI]) -> list[tuple[float, float, str]]:
    """Convert POIs to (easting, northing, label) tuples for figures.py."""
    return [(p.easting, p.northing, p.name) for p in pois]
