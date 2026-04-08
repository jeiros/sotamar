"""Tests for sotamar.sites: Site dataclass, registry, coordinate verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sotamar.sites import (
    Site,
    all_sites,
    get_site,
    list_sites,
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
            site.slug = "y"

    def test_bounds_default_half_size(self):
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C",
        )
        left, bottom, right, top = site.bounds
        assert left == 499000
        assert bottom == 4599000
        assert right == 501000
        assert top == 4601000

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

    def test_default_transect_ew_through_center(self):
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C", half_size=1000,
        )
        start, end = site.transect_endpoints
        # E-W: same northing
        assert start[1] == end[1] == 4600000
        # Inset 100 m from edges
        assert start[0] == 499100
        assert end[0] == 500900

    def test_custom_transect_overrides_default(self):
        custom = ((1.0, 2.0), (3.0, 4.0))
        site = Site(
            slug="a", name="A", easting=500000, northing=4600000,
            region="R", character="C", transect=custom,
        )
        assert site.transect_endpoints == custom

    def test_default_values(self):
        site = Site(
            slug="a", name="A", easting=0, northing=0,
            region="R", character="C",
        )
        assert site.half_size == 1000
        assert site.transect is None


# -- Registry -----------------------------------------------------------------

class TestRegistry:
    def test_six_sites_registered(self):
        assert len(list_sites()) == 6

    def test_all_slugs_present(self):
        expected = {
            "roses", "illes_medes", "illes_formigues",
            "tossa_de_mar", "costa_del_garraf", "cap_de_salou",
        }
        assert set(list_sites()) == expected

    def test_get_site_returns_correct_type(self):
        site = get_site("illes_medes")
        assert isinstance(site, Site)

    def test_get_site_unknown_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_site("atlantis")

    def test_all_sites_returns_list_of_sites(self):
        sites = all_sites()
        assert all(isinstance(s, Site) for s in sites)

    def test_list_sites_matches_all_sites(self):
        slugs = list_sites()
        sites = all_sites()
        assert [s.slug for s in sites] == slugs


# -- Known site properties ----------------------------------------------------

class TestKnownSites:
    def test_medes_has_custom_transect(self):
        site = get_site("illes_medes")
        assert site.transect is not None
        start, end = site.transect
        assert start == (517500, 4655100)
        assert end == (519300, 4655100)

    def test_medes_coordinates(self):
        site = get_site("illes_medes")
        assert site.easting == 518400
        assert site.northing == 4655100

    def test_all_sites_have_positive_coordinates(self):
        for site in all_sites():
            assert site.easting > 0, f"{site.slug} has non-positive easting"
            assert site.northing > 0, f"{site.slug} has non-positive northing"

    def test_all_sites_in_utm31n_range(self):
        """UTM zone 31N easting should be ~100k-900k, northing ~0-10M."""
        for site in all_sites():
            assert 100_000 < site.easting < 900_000, f"{site.slug} easting out of UTM31N"
            assert 4_000_000 < site.northing < 5_000_000, f"{site.slug} northing out of range"

    def test_sites_without_custom_transect_use_default(self):
        for site in all_sites():
            if site.transect is None:
                start, end = site.transect_endpoints
                # Should be E-W through center
                assert start[1] == end[1] == site.northing

    def test_bounds_produce_2km_window(self):
        for site in all_sites():
            left, bottom, right, top = site.bounds
            assert right - left == 2000
            assert top - bottom == 2000


# -- Coordinate verification (mocked) ----------------------------------------

class TestVerifyCoordinates:
    @patch("geopy.geocoders.Nominatim")
    @patch("pyproj.Transformer")
    def test_successful_verification(self, mock_transformer_cls, mock_nominatim_cls):
        # Set up mock geocoder
        mock_location = MagicMock()
        mock_location.address = "Illes Medes, Torroella de Montgrí"
        mock_location.latitude = 42.047
        mock_location.longitude = 3.222

        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim_cls.return_value = mock_geolocator

        # Set up mock transformer
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (518390.0, 4655090.0)
        mock_transformer_cls.from_crs.return_value = mock_transformer

        site = get_site("illes_medes")
        result = verify_coordinates(site)

        assert result["site"] == "illes_medes"
        assert "distance_m" in result
        assert result["distance_m"] >= 0
        assert "geocoded_address" in result
        assert result["geocoded_easting"] == 518390.0

    @patch("geopy.geocoders.Nominatim")
    def test_geocode_not_found(self, mock_nominatim_cls):
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = None
        mock_nominatim_cls.return_value = mock_geolocator

        site = get_site("roses")
        result = verify_coordinates(site)

        assert "error" in result
        assert "Could not geocode" in result["error"]

    @patch("geopy.geocoders.Nominatim")
    def test_geocode_exception_handled(self, mock_nominatim_cls):
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.side_effect = Exception("Network timeout")
        mock_nominatim_cls.return_value = mock_geolocator

        site = get_site("roses")
        result = verify_coordinates(site)

        assert "error" in result
        assert "Network timeout" in result["error"]

    @patch("geopy.geocoders.Nominatim")
    @patch("pyproj.Transformer")
    def test_distance_calculation(self, mock_transformer_cls, mock_nominatim_cls):
        mock_location = MagicMock()
        mock_location.address = "Test"
        mock_location.latitude = 42.0
        mock_location.longitude = 3.0
        mock_geolocator = MagicMock()
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim_cls.return_value = mock_geolocator

        # Return exact same coords → distance should be 0
        site = get_site("illes_medes")
        mock_transformer = MagicMock()
        mock_transformer.transform.return_value = (site.easting, site.northing)
        mock_transformer_cls.from_crs.return_value = mock_transformer

        result = verify_coordinates(site)
        assert result["distance_m"] == 0.0
