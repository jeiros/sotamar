"""CLI entry point: sotamar list | analyze | check-coords."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from sotamar.sites import get_site, list_sites, all_sites, Site
from sotamar.io import (
    read_bathymetry_window,
    save_geotiff,
    compute_stats,
    compute_depth_zone_pcts,
    save_stats,
    find_cog,
)
from sotamar.terrain import (
    compute_slope,
    compute_hillshade,
    compute_bpi,
    compute_vrm,
    compute_depth_zones,
)
from sotamar.profile import extract_depth_profile
from sotamar.figures import plot_terrain_analysis, plot_depth_profile

DEFAULT_OUTPUT_BASE = Path("data/sites")


@click.group()
def cli():
    """SotaMar: Bathymetric terrain analysis for Catalan coast dive sites."""


@cli.command("list")
def list_cmd():
    """List all registered dive sites."""
    sites = all_sites()
    click.echo(f"{'Slug':<20} {'Name':<20} {'Region':<20} {'Easting':>10} {'Northing':>10}")
    click.echo("-" * 82)
    for s in sites:
        click.echo(
            f"{s.slug:<20} {s.name:<20} {s.region:<20} "
            f"{s.easting:>10.0f} {s.northing:>10.0f}"
        )
    click.echo(f"\n{len(sites)} sites registered.")


@cli.command()
@click.argument("slug", required=False)
@click.option("--all", "run_all", is_flag=True, help="Analyze all registered sites.")
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output base directory (default: data/sites/).",
)
@click.option(
    "--cog", type=click.Path(exists=True), default=None,
    help="Path to the ICGC bathymetry COG.",
)
def analyze(slug, run_all, output, cog):
    """Run full analysis pipeline for a dive site.

    Provide a site SLUG or use --all for batch processing.
    """
    if not slug and not run_all:
        raise click.UsageError("Provide a site SLUG or use --all.")

    output_base = Path(output) if output else DEFAULT_OUTPUT_BASE
    cog_path = Path(cog) if cog else None

    # Validate COG exists early
    resolved_cog = find_cog(cog_path)
    click.echo(f"COG: {resolved_cog}")

    if run_all:
        sites = all_sites()
    else:
        try:
            sites = [get_site(slug)]
        except KeyError:
            available = ", ".join(list_sites())
            raise click.UsageError(f"Unknown site '{slug}'. Available: {available}")

    for site in sites:
        _analyze_site(site, output_base, cog_path)


def _analyze_site(site: Site, output_base: Path, cog_path: Path | None) -> None:
    """Full pipeline for one site."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {site.name} ({site.slug})")
    click.echo(f"{'=' * 60}")

    output_dir = output_base / site.slug
    t_start = time.time()

    # 1. Read bathymetry
    click.echo("  Reading bathymetry window...")
    elevation, mask, profile = read_bathymetry_window(site.bounds, cog_path)
    click.echo(f"  Shape: {elevation.shape}, NoData: {mask.sum() / mask.size * 100:.1f}%")

    # 2. Check data sufficiency
    nodata_pct = mask.sum() / mask.size * 100
    if nodata_pct > 95:
        click.echo(f"  SKIPPING: {nodata_pct:.0f}% nodata — insufficient data.")
        return

    # 3. Compute terrain metrics
    click.echo("  Computing slope...")
    t0 = time.time()
    slope = compute_slope(elevation, mask)
    click.echo(f"    {time.time() - t0:.1f}s")

    click.echo("  Computing hillshade...")
    t0 = time.time()
    hillshade = compute_hillshade(elevation, mask)
    click.echo(f"    {time.time() - t0:.1f}s")

    click.echo("  Computing fine BPI (r=3-5)...")
    t0 = time.time()
    bpi_fine = compute_bpi(elevation, mask, inner_radius=3, outer_radius=5)
    click.echo(f"    {time.time() - t0:.1f}s")

    click.echo("  Computing broad BPI (r=25-50)...")
    t0 = time.time()
    bpi_broad = compute_bpi(elevation, mask, inner_radius=25, outer_radius=50)
    click.echo(f"    {time.time() - t0:.1f}s")

    click.echo("  Computing VRM (3x3)...")
    t0 = time.time()
    vrm = compute_vrm(elevation, mask, window_size=3)
    click.echo(f"    {time.time() - t0:.1f}s")

    click.echo("  Classifying depth zones...")
    t0 = time.time()
    depth_zones = compute_depth_zones(elevation, mask)
    click.echo(f"    {time.time() - t0:.1f}s")

    # 4. Save GeoTIFFs
    click.echo("  Saving GeoTIFFs...")
    for name, array in [
        ("bathymetry", elevation),
        ("slope", slope),
        ("hillshade", hillshade),
        ("bpi_fine", bpi_fine),
        ("bpi_broad", bpi_broad),
        ("vrm", vrm),
        ("depth_zones", depth_zones),
    ]:
        save_geotiff(array, profile, output_dir / f"{name}.tif")

    # 5. Compute and save statistics
    click.echo("  Computing statistics...")
    stats = compute_stats(
        {"depth": elevation, "slope": slope, "bpi_fine": bpi_fine,
         "bpi_broad": bpi_broad, "vrm": vrm},
        mask,
    )
    stats["depth_zones"] = compute_depth_zone_pcts(depth_zones)
    stats["slug"] = site.slug
    stats["name"] = site.name
    stats["easting"] = site.easting
    stats["northing"] = site.northing
    stats["half_size"] = site.half_size
    stats["region"] = site.region
    stats["character"] = site.character
    save_stats(stats, output_dir / "stats.json")

    # 6. Extract depth profile
    click.echo("  Extracting depth profile...")
    start, end = site.transect_endpoints
    distances, depths = extract_depth_profile(
        elevation, profile["transform"], start, end,
    )

    # 7. Generate figures
    click.echo("  Generating terrain analysis figure...")
    plot_terrain_analysis(
        elevation, slope, bpi_fine, bpi_broad, vrm, depth_zones,
        site.bounds, site.name, output_dir,
    )

    click.echo("  Generating depth profile figure...")
    plot_depth_profile(distances, depths, site.name, output_dir)

    elapsed = time.time() - t_start
    click.echo(f"  Done in {elapsed:.1f}s → {output_dir}/")


@cli.command("check-coords")
def check_coords():
    """Geocode all site names and compare with registered UTM coordinates."""
    from sotamar.sites import verify_all_coordinates

    click.echo("Verifying coordinates via Nominatim geocoding...")
    click.echo("(1 request/sec rate limit — this takes a few seconds)\n")

    results = verify_all_coordinates()

    click.echo(
        f"{'Site':<20} {'Distance':>10} {'Geocoded address'}"
    )
    click.echo("-" * 80)

    for r in results:
        if "error" in r:
            click.echo(f"{r['site']:<20} {'ERROR':>10} {r['error']}")
        else:
            dist = r["distance_m"]
            marker = " !!!" if dist > 2000 else ""
            click.echo(
                f"{r['site']:<20} {dist:>8.0f} m {r['geocoded_address'][:50]}{marker}"
            )

    click.echo(
        "\nDistances > 2000 m are flagged (!!!) — coordinates may need correction."
    )


@cli.command("load-db")
@click.option(
    "--db-url", envvar="SOTAMAR_DB_URL", default=None,
    help="PostgreSQL URL (default from SOTAMAR_DB_URL or localhost:5432/sotamar).",
)
@click.option(
    "--sites-dir", type=click.Path(file_okay=False), default="data/sites",
    help="Per-site output directory (default: data/sites).",
)
@click.option(
    "--only", "slugs", multiple=True,
    help="Restrict to specific slugs (repeatable).",
)
def load_db(db_url, sites_dir, slugs):
    """Load the site registry, stats.json, and rasters into PostGIS."""
    from sotamar import db as dbmod

    engine = dbmod.get_engine(db_url)
    slug_filter = list(slugs) if slugs else None
    summary = dbmod.load_all_sites(
        engine, sites_dir=Path(sites_dir), slugs=slug_filter,
    )

    click.echo(f"\nLoaded {summary.sites} sites, {summary.stats} stats rows, "
               f"{summary.rasters} raster rows.")
    if summary.sites_without_zones:
        click.echo(
            f"  note: {len(summary.sites_without_zones)} site(s) loaded without "
            f"depth_zones — stale stats.json. Re-run `analyze --all`:"
        )
        for slug in summary.sites_without_zones:
            click.echo(f"    - {slug}")
    if summary.skipped:
        click.echo(f"\n  skipped {len(summary.skipped)} site(s):")
        for slug, reason in summary.skipped:
            click.echo(f"    - {slug}: {reason}")
        if not slugs:
            raise SystemExit(1)


@cli.command("export-geojson")
@click.option(
    "--db-url", envvar="SOTAMAR_DB_URL", default=None,
    help="PostgreSQL URL (default from SOTAMAR_DB_URL or localhost:5432/sotamar).",
)
@click.option(
    "--output", "-o", type=click.Path(dir_okay=False),
    default="data/catalog/sites.geojson",
    help="Output path (default: data/catalog/sites.geojson).",
)
@click.option(
    "--from-files", is_flag=True,
    help="Bypass DB and build from sites.py + stats.json directly.",
)
@click.option("--pretty/--compact", default=True)
def export_geojson(db_url, output, from_files, pretty):
    """Export all dive sites as a GeoJSON FeatureCollection (WGS84)."""
    from sotamar import db as dbmod

    rows = None
    if not from_files:
        try:
            engine = dbmod.get_engine(db_url)
            with engine.connect() as conn:
                conn.execute(sqlalchemy_text("SELECT 1"))
            rows = dbmod.fetch_sites_with_stats(engine)
            click.echo(f"Read {len(rows)} sites from database.")
        except Exception as exc:
            click.echo(f"Database unreachable ({exc}); falling back to files.")
            rows = None

    if rows is None:
        rows = dbmod.site_rows_from_files()
        click.echo(f"Read {len(rows)} sites from file registry.")

    fc = dbmod.site_rows_to_geojson(rows)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    out_path.write_text(json.dumps(fc, indent=indent, default=_json_default))
    click.echo(f"Wrote {len(rows)} features to {out_path}")


def _json_default(obj):
    """JSON fallback for decimal/datetime-like values coming back from pg."""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    try:
        return obj.isoformat()
    except AttributeError:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serialisable")


def sqlalchemy_text(sql):
    import sqlalchemy
    return sqlalchemy.text(sql)


if __name__ == "__main__":
    cli()
