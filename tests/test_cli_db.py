"""Smoke tests for the new CLI commands: load-db and export-geojson."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sotamar.cli import cli


class TestExportGeojsonDbUnreachable:
    """Without --from-files, a DB error must exit non-zero with a hint."""

    def test_unreachable_db_fails_loudly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["export-geojson", "-o", str(tmp_path / "out.geojson"),
             "--db-url", "postgresql+psycopg://nope:nope@127.0.0.1:1/nodb"],
        )
        assert result.exit_code != 0
        assert "could not read from PostGIS" in result.output
        assert "--from-files" in result.output
        assert not (tmp_path / "out.geojson").exists()


class TestExportGeojsonFromFiles:
    """The --from-files path must work without a database."""

    def test_produces_valid_geojson(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "catalog" / "sites.geojson"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["export-geojson", "--from-files", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

        fc = json.loads(out.read_text())
        assert fc["type"] == "FeatureCollection"
        assert fc["name"] == "sotamar_dive_sites"
        assert len(fc["features"]) >= 18

        feat = fc["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Point"
        lon, lat = feat["geometry"]["coordinates"]
        # Sanity bounds for Catalan coast.
        assert 0.0 < lon < 4.0
        assert 40.0 < lat < 43.0

        props = feat["properties"]
        for key in ("slug", "name", "region", "character",
                    "window_size", "easting_25831", "northing_25831",
                    "rasters"):
            assert key in props


class TestLoadDbCli:
    """DB-gated: runs only when PostGIS is reachable."""

    def test_load_db_smoke(self, clean_db, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["load-db", "--sites-dir", str(tmp_path),
             "--db-url", str(clean_db.url)],
        )
        assert result.exit_code == 0, result.output
        assert "Loaded" in result.output
