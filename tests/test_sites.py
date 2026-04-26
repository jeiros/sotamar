"""Tests for sotamar.sites: CSV-driven Site loader + coordinate verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sotamar.pois import POI
from sotamar.sites import (
    Site,
    _HALF_SIZE_BY_TYPE,
    all_sites,
    get_site,
    list_sites,
    poi_to_site,
    verify_coordinates,
)


# -- Site dataclass -----------------------------------------------------------

class TestSiteDataclass:
    def test_frozen(self):
        site = Site(
            slug="x", name="X", easting=1.0, northing=2.0,
            region="R", character="C",
        )
        with pytest.raises(AttributeError):
            site.slug = "y"  # pyright: ignore[reportAttributeAccessIssue]

    def test_bounds_default_half_size(self):
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C",
        )
        left, bottom, right, top = site.bounds
        # Default half_size=200 → 400 × 400 m window
        assert left == 499800
        assert bottom == 4599800
        assert right == 500200
        assert top == 4600200

    def test_bounds_custom_half_size(self):
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C", half_size=500,
        )
        left, bottom, right, top = site.bounds
        assert right - left == 1000
        assert top - bottom == 1000

    def test_bounds_is_centered(self):
        site = Site(
            slug="a", name="A", easting=100, northing=200,
            region="R", character="C", half_size=10,
        )
        left, bottom, right, top = site.bounds
        assert (left + right) / 2 == 100
        assert (bottom + top) / 2 == 200

    def test_default_transect_through_center(self):
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C", half_size=500,
        )
        start, end = site.transect_endpoints
        # E-W: same northing
        assert start[1] == end[1] == 4600000
        # Inset min(100, half_size//4) from edges → 100 m here
        assert start[0] == 499600
        assert end[0] == 500400

    def test_default_transect_small_window(self):
        """Tight wreck windows shouldn't have a margin larger than half_size//4."""
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C", half_size=75,
        )
        start, end = site.transect_endpoints
        # margin = min(100, 75//4) = 18
        assert start[0] == 500000 - 75 + 18
        assert end[0] == 500000 + 75 - 18

    def test_custom_transect_overrides_default(self):
        custom = ((1.0, 2.0), (3.0, 4.0))
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C", transect=custom,
        )
        assert site.transect_endpoints == custom


# -- poi_to_site -------------------------------------------------------------

class TestPoiToSite:
    def _poi(self, **kw) -> POI:
        defaults = dict(
            id="t", name="Test", region="r", municipality=None,
            site_type="wreck",
            latitude=41.83432, longitude=3.12065,
            coord_confidence="verified",
            depth_min_m=None, depth_max_m=None,
            description=None, sources=None,
        )
        defaults.update(kw)
        return POI(**defaults)  # type: ignore[arg-type]

    def test_wreck_uses_75(self):
        site = poi_to_site(self._poi(site_type="wreck"))
        assert site.half_size == 75

    def test_pinnacle_uses_100(self):
        site = poi_to_site(self._poi(site_type="pinnacle"))
        assert site.half_size == 100

    def test_island_uses_150(self):
        site = poi_to_site(self._poi(site_type="island"))
        assert site.half_size == 150

    def test_headland_uses_500(self):
        site = poi_to_site(self._poi(site_type="headland"))
        assert site.half_size == 500

    def test_unknown_type_raises(self):
        with pytest.raises(KeyError):
            poi_to_site(self._poi(site_type="bogus"))

    def test_coords_snap_to_100m(self):
        # 41.83432 / 3.12065 → about 510018 / 4631388
        site = poi_to_site(self._poi(site_type="wreck"))
        assert site.easting % 100 == 0
        assert site.northing % 100 == 0
        # Expected after rounding: 510000, 4631400
        assert site.easting == 510000
        assert site.northing == 4631400

    def test_slug_is_poi_id(self):
        site = poi_to_site(self._poi(id="custom_id"))
        assert site.slug == "custom_id"

    def test_max_depth_propagates(self):
        site = poi_to_site(self._poi(depth_max_m=42.0))
        assert site.max_depth == 42.0


# -- Registry (CSV-backed) ---------------------------------------------------

class TestRegistry:
    """Drives off the real data/dive_sites.csv. Don't mock the loader."""

    def test_at_least_50_sites(self):
        assert len(all_sites()) >= 50

    def test_every_slug_uses_region_prefix(self):
        valid_prefixes = ("cdc_", "med_", "mon_", "pal_", "sel_", "gar_", "dau_")
        for slug in list_sites():
            assert slug.startswith(valid_prefixes), (
                f"slug {slug!r} doesn't use a known region prefix"
            )

    def test_every_site_has_known_type(self):
        """Every registered site's half_size matches a known site_type."""
        valid_half_sizes = set(_HALF_SIZE_BY_TYPE.values())
        for site in all_sites():
            assert site.half_size in valid_half_sizes

    def test_get_site_known(self):
        # pal_boreas should always be in the verified catalogue
        site = get_site("pal_boreas")
        assert isinstance(site, Site)
        assert site.name.startswith("Boreas")

    def test_get_site_unknown_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_site("atlantis")

    def test_list_matches_all_sites_order(self):
        slugs = list_sites()
        sites = all_sites()
        assert [s.slug for s in sites] == slugs


# -- Known site properties (against real CSV) -------------------------------

class TestKnownSites:
    def test_all_sites_have_positive_coordinates(self):
        for site in all_sites():
            assert site.easting > 0, f"{site.slug} has non-positive easting"
            assert site.northing > 0, f"{site.slug} has non-positive northing"

    def test_all_sites_in_utm31n_range(self):
        """UTM zone 31N easting should be ~100k-900k, northing ~4M-5M."""
        for site in all_sites():
            assert 100_000 < site.easting < 900_000, (
                f"{site.slug} easting out of UTM31N range"
            )
            assert 4_000_000 < site.northing < 5_000_000, (
                f"{site.slug} northing out of range"
            )

    def test_half_size_policy_table_complete(self):
        """The seven canonical types after coalescing."""
        assert set(_HALF_SIZE_BY_TYPE.keys()) == {
            "wreck", "pinnacle", "cave", "island",
            "wall", "cove", "headland",
        }

    def test_bounds_match_half_size(self):
        for site in all_sites():
            left, bottom, right, top = site.bounds
            assert right - left == 2 * site.half_size
            assert top - bottom == 2 * site.half_size


# -- Coordinate verification (mocked) ----------------------------------------

class TestVerifyCoordinates:
    @patch("geopy.geocoders.Nominatim")
    @patch("pyproj.Transformer")
    def test_successful_verification(self, mock_transformer_cls, mock_nominatim_cls):
        mock_location = MagicMock()
        mock_location.address = "Boreas wreck, Palamós"
        mock_location.latitude = 41.8313
        mock_location.longitude = 3.1184

        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim_cls.return_value = mock_geolocator

        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (509800.0, 4631100.0)
        mock_transformer_cls.from_crs.return_value = mock_transformer

        site = get_site("pal_boreas")
        result = verify_coordinates(site)

        assert result["site"] == "pal_boreas"
        assert "distance_m" in result
        assert result["distance_m"] >= 0
        assert "geocoded_address" in result

    @patch("geopy.geocoders.Nominatim")
    def test_geocode_not_found(self, mock_nominatim_cls):
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = None
        mock_nominatim_cls.return_value = mock_geolocator

        site = get_site("pal_boreas")
        result = verify_coordinates(site)

        assert "error" in result
        assert "Could not geocode" in result["error"]

    @patch("geopy.geocoders.Nominatim")
    def test_geocode_exception_handled(self, mock_nominatim_cls):
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.side_effect = Exception("Network timeout")
        mock_nominatim_cls.return_value = mock_geolocator

        site = get_site("pal_boreas")
        result = verify_coordinates(site)

        assert "error" in result
        assert "Network timeout" in result["error"]
