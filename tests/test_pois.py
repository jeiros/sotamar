"""Tests for sotamar.pois: CSV-driven POI catalogue."""

from __future__ import annotations

from pathlib import Path

import pytest

from sotamar.pois import (
    POI,
    VALID_CONFIDENCE,
    load_pois,
    pois_in_bounds,
    pois_to_markers,
)


REPO_CSV = Path(__file__).resolve().parents[1] / "data" / "dive_sites.csv"


# -- POI dataclass ------------------------------------------------------------

class TestPOI:
    def test_easting_northing_round_trip(self):
        # Coordinates that should round-trip via pyproj. Boreas wreck UTM is
        # E=510018, N=4631388 per earlier investigation.
        poi = POI(
            id="t", name="t", region="r", municipality=None, site_type="wreck",
            latitude=41.83432, longitude=3.12065,
            coord_confidence="verified",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )
        assert abs(poi.easting - 510018) < 5
        assert abs(poi.northing - 4631388) < 5


# -- load_pois ----------------------------------------------------------------

class TestLoadPois:
    def test_loads_real_catalogue(self):
        """The repo CSV must load cleanly and have known landmarks."""
        if not REPO_CSV.exists():
            pytest.skip(f"{REPO_CSV} not present")
        pois = load_pois(REPO_CSV)
        assert len(pois) > 50
        ids = {p.id for p in pois}
        assert "pal_boreas" in ids
        assert "med_meda_gran" in ids

    def test_all_confidences_valid(self):
        if not REPO_CSV.exists():
            pytest.skip(f"{REPO_CSV} not present")
        for poi in load_pois(REPO_CSV):
            assert poi.coord_confidence in VALID_CONFIDENCE

    def test_synthetic_minimal_csv(self, tmp_path):
        csv_path = tmp_path / "pois.csv"
        csv_path.write_text(
            "id,name,region,municipality,site_type,latitude,longitude,"
            "coord_confidence,depth_min_m,depth_max_m,description,sources\n"
            "t1,Test One,test,Town,pinnacle,41.83,3.12,verified,,,desc,src\n"
            "t2,Test Two,test,,wreck,41.84,3.13,approximate,5,30,,\n",
            encoding="utf-8",
        )
        pois = load_pois(csv_path)
        assert len(pois) == 2
        assert pois[0].municipality == "Town"
        assert pois[1].municipality is None
        assert pois[0].depth_min_m is None
        assert pois[1].depth_min_m == 5.0
        assert pois[1].depth_max_m == 30.0
        assert pois[1].description is None

    def test_rejects_invalid_confidence(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "id,name,region,municipality,site_type,latitude,longitude,"
            "coord_confidence,depth_min_m,depth_max_m,description,sources\n"
            "t1,Test,test,,wreck,41.0,3.0,bogus,,,,\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="coord_confidence"):
            load_pois(csv_path)


# -- pois_in_bounds -----------------------------------------------------------

class TestPoisInBounds:
    @pytest.fixture
    def sample(self):
        # Approx UTM 510000, 4631400 (around Boreas)
        inside = POI(
            id="in", name="in", region="r", municipality=None, site_type="wreck",
            latitude=41.83432, longitude=3.12065,
            coord_confidence="verified",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )
        # Far away: Garraf area
        outside = POI(
            id="out", name="out", region="r", municipality=None, site_type="cove",
            latitude=41.265, longitude=1.91,
            coord_confidence="approximate",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )
        return [inside, outside]

    def test_filters_to_window(self, sample):
        # 1 km half-size around the inside POI
        bounds = (509018, 4630388, 511018, 4632388)
        result = pois_in_bounds(sample, bounds)
        assert len(result) == 1
        assert result[0].id == "in"

    def test_empty_when_window_disjoint(self, sample):
        bounds = (0, 0, 1000, 1000)
        assert pois_in_bounds(sample, bounds) == []


# -- pois_to_markers ----------------------------------------------------------

class TestPoisToMarkers:
    def test_shape(self):
        pois = [
            POI(id="a", name="A", region="r", municipality=None,
                site_type="wreck", latitude=41.0, longitude=3.0,
                coord_confidence="verified",
                depth_min_m=None, depth_max_m=None,
                description=None, sources=None),
        ]
        markers = pois_to_markers(pois)
        assert len(markers) == 1
        e, n, label = markers[0]
        assert isinstance(e, float)
        assert isinstance(n, float)
        assert label == "A"
