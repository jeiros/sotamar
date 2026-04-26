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
from sotamar.pois import (
    DEFAULT_CSV_PATH as DEFAULT_POIS_CSV,
    load_pois,
    pois_in_bounds,
    pois_to_markers,
)

DEFAULT_OUTPUT_BASE = Path("data/sites")


def _markers_for_site(site: Site) -> list[tuple[float, float, str]]:
    """Combine Site.markers with POIs from the CSV that fall in the window.

    POIs are loaded from data/dive_sites.csv if present (silently skipped if
    not). The POI whose id equals the site slug is dropped — the figure
    already represents that POI, a self-marker is redundant. Remaining
    duplicates between manual Site.markers and POI catalogue are de-duped
    by rounding coordinates to the nearest 10 m.
    """
    markers = list(site.markers)
    if DEFAULT_POIS_CSV.exists():
        in_window = [p for p in pois_in_bounds(load_pois(), site.bounds)
                     if p.id != site.slug]
        markers += pois_to_markers(in_window)
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[float, float, str]] = []
    for e, n, label in markers:
        key = (round(e, -1), round(n, -1))
        if key in seen:
            continue
        seen.add(key)
        unique.append((e, n, label))
    return unique


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
    elevation, nodata_mask, profile = read_bathymetry_window(site.bounds, cog_path)

    # 2. Clip emerged terrain (above sea level). This is a dive-site
    # analysis; subaerial topography isn't informative for dive planning
    # and would skew slope/BPI/VRM statistics. Emerged pixels become NaN
    # and are folded into the "not analysed" mask.
    import numpy as _np
    emerged_mask = (~nodata_mask) & (elevation > 0)
    emerged_pct = round(float(emerged_mask.sum() / emerged_mask.size * 100), 2)
    nodata_pct_true = round(
        float(nodata_mask.sum() / nodata_mask.size * 100), 2,
    )
    elevation[emerged_mask] = _np.nan
    mask = nodata_mask | emerged_mask
    click.echo(
        f"  Shape: {elevation.shape}, NoData: {nodata_pct_true:.1f}%, "
        f"Emerged: {emerged_pct:.1f}%"
    )

    # 3. Check data sufficiency
    not_analysed_pct = mask.sum() / mask.size * 100
    if not_analysed_pct > 95:
        click.echo(
            f"  SKIPPING: only {100 - not_analysed_pct:.0f}% submerged "
            f"— insufficient data."
        )
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
    # Override mask-derived nodata_pct with the split (true gap vs emerged).
    stats["nodata_pct"] = nodata_pct_true
    stats["emerged_pct"] = emerged_pct
    save_stats(stats, output_dir / "stats.json")

    # 6. Extract depth profile
    click.echo("  Extracting depth profile...")
    start, end = site.transect_endpoints
    distances, depths = extract_depth_profile(
        elevation, profile["transform"], start, end,
    )

    # 7. Generate figures
    click.echo("  Generating terrain analysis figure...")
    markers = _markers_for_site(site)
    plot_terrain_analysis(
        elevation, slope, bpi_fine, bpi_broad, vrm, depth_zones,
        site.bounds, site.name, output_dir,
        markers=markers or None,
    )

    click.echo("  Generating depth profile figure...")
    wrote = plot_depth_profile(distances, depths, site.name, output_dir)
    if not wrote:
        click.echo("    (skipped: transect crosses only NoData/emerged terrain)")

    elapsed = time.time() - t_start
    click.echo(f"  Done in {elapsed:.1f}s → {output_dir}/")


@cli.command("detect-wrecks")
@click.option(
    "--cog", type=click.Path(exists=True), default=None,
    help="Path to the ICGC bathymetry COG.",
)
@click.option(
    "--radius", type=int, default=1500,
    help="Search radius in metres around each seed coord (default 1500).",
)
@click.option(
    "-o", "--output", type=click.Path(file_okay=False), default="data",
    help="Output base directory; CSV goes here, figures under data/sites/<id>/.",
)
@click.option(
    "--all-wrecks", is_flag=True,
    help="Also run detection on verified wreck POIs (sanity check).",
)
def detect_wrecks(cog, radius, output, all_wrecks):
    """Scan bathymetry for wreck-shaped anomalies near unverified wreck POIs.

    Iterates over every wreck POI in data/dive_sites.csv whose
    coord_confidence is 'unverified', searches a `--radius`-metre window
    around each approximate coord, and reports compact positive elevation
    anomalies as wreck candidates. With --all-wrecks, also runs against
    verified wrecks (useful for calibration).
    """
    import csv as _csv
    from sotamar.pois import load_pois
    from sotamar.wreck_detect import detect_wrecks_near, plot_wreck_candidates

    cog_path = Path(cog) if cog else None
    output_base = Path(output)
    sites_dir = output_base / "sites"
    csv_path = output_base / "wreck_candidates.csv"

    confidences = (
        ("verified", "unverified") if all_wrecks else ("unverified",)
    )
    targets = [
        p for p in load_pois()
        if p.site_type == "wreck" and p.coord_confidence in confidences
    ]
    if not targets:
        click.echo("No wreck POIs match the filter.")
        return

    click.echo(f"Scanning {len(targets)} wreck POI(s) at radius {radius} m\n")
    all_candidates: list = []
    for poi in targets:
        click.echo(f"=== {poi.name} ({poi.id}) [{poi.coord_confidence}] ===")
        candidates, debug = detect_wrecks_near(
            seed_lat=poi.latitude,
            seed_lon=poi.longitude,
            cog_path=cog_path,
            search_radius_m=radius,
            source_poi_id=poi.id,
        )
        click.echo(f"  {len(candidates)} candidate(s) above threshold")
        for c in candidates[:3]:
            click.echo(
                f"    #{c.rank}  ({c.peak_lat:.5f},{c.peak_lon:.5f})  "
                f"peak {c.peak_residual_m:+.2f} m  size {c.footprint_m2} m²  "
                f"{c.length_m}×{c.width_m} m  plausibility {c.plausibility}"
            )
        plot_wreck_candidates(
            candidates, debug,
            sites_dir / poi.id / "wreck_candidates.png",
            source_name=poi.name,
            seed_lat=poi.latitude, seed_lon=poi.longitude,
            radius_m=radius,
        )
        all_candidates.extend(candidates)

    # Write the combined candidates CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_candidates[0].asdict().keys()) if all_candidates \
        else ["source_poi_id"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for c in all_candidates:
            writer.writerow(c.asdict())
    click.echo(f"\nWrote {len(all_candidates)} candidate(s) → {csv_path}")
    click.echo(f"Figures → {sites_dir}/<poi_id>/wreck_candidates.png")


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
               f"{summary.rasters} raster rows, {summary.pois} POIs.")
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

    if from_files:
        rows = dbmod.site_rows_from_files()
        click.echo(f"Read {len(rows)} sites from file registry.")
    else:
        try:
            engine = dbmod.get_engine(db_url)
            rows = dbmod.fetch_sites_with_stats(engine)
        except Exception as exc:
            raise click.ClickException(
                f"could not read from PostGIS: {exc}\n"
                "Hint: pass --from-files to bypass the database."
            )
        click.echo(f"Read {len(rows)} sites from database.")

    fc = dbmod.site_rows_to_geojson(rows)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    out_path.write_text(json.dumps(fc, indent=indent, default=_json_default))
    click.echo(f"Wrote {len(rows)} features to {out_path}")


@cli.command()
@click.option(
    "--db-url", envvar="SOTAMAR_DB_URL", default=None,
    help="PostgreSQL URL (default from SOTAMAR_DB_URL or localhost:5432/sotamar).",
)
@click.option(
    "--sites-dir", type=click.Path(file_okay=False), default="data/sites",
    help="Per-site output directory (default: data/sites).",
)
@click.option(
    "--output", "-o", type=click.Path(file_okay=False),
    default="data/viewer",
    help="Directory for generated HTML (default: data/viewer).",
)
@click.option(
    "--from-files", is_flag=True,
    help="Bypass DB; read site registry + stats.json directly.",
)
@click.option(
    "--grid-size", type=int, default=100,
    help="Downsample grid per site (default 100×100 ≈ 20 m cells).",
)
@click.option(
    "--cog", type=click.Path(exists=True), default=None,
    help="Path to the ICGC bathymetry COG (used for regional pages).",
)
def viewer(db_url, sites_dir, output, from_files, grid_size, cog):
    """Generate a static HTML viewer: overview map + per-site 3D seabed."""
    from sotamar import db as dbmod
    from sotamar import viewer as viewermod

    if from_files:
        rows = dbmod.site_rows_from_files(Path(sites_dir))
        click.echo(f"Read {len(rows)} sites from file registry.")
    else:
        try:
            engine = dbmod.get_engine(db_url)
            rows = dbmod.fetch_sites_with_stats(engine)
        except Exception as exc:
            raise click.ClickException(
                f"could not read from PostGIS: {exc}\n"
                "Hint: pass --from-files to bypass the database."
            )
        click.echo(f"Read {len(rows)} sites from database.")

    summary = viewermod.write_viewer(
        rows, output_dir=Path(output), sites_dir=Path(sites_dir),
        grid_size=grid_size, cog_path=Path(cog) if cog else None,
    )

    click.echo(f"Wrote overview → {summary.overview}")
    for slug, site_dir, cells in summary.sites:
        click.echo(f"Wrote {len(viewermod.METRICS)} metric views → "
                   f"{site_dir}/ ({cells} cells)")
    for region, html_path, n in summary.regions:
        click.echo(f"Wrote regional 3D view → {html_path} ({n} POIs)")
    if summary.skipped:
        click.echo(f"\n  skipped {len(summary.skipped)} site(s):")
        for slug, reason in summary.skipped:
            click.echo(f"    - {slug}: {reason}")
    total = 1 + len(summary.sites) * len(viewermod.METRICS)
    click.echo(f"\nWrote {total} HTML file(s) to {output}/")


def _json_default(obj):
    """JSON fallback for decimal/datetime-like values coming back from pg."""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    try:
        return obj.isoformat()
    except AttributeError:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON-serialisable")


if __name__ == "__main__":
    cli()
