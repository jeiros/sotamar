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
    save_stats,
    find_cog,
)
from sotamar.terrain import (
    compute_slope,
    compute_hillshade,
    compute_bpi,
    compute_vrm,
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

    # 4. Save GeoTIFFs
    click.echo("  Saving GeoTIFFs...")
    for name, array in [
        ("bathymetry", elevation),
        ("slope", slope),
        ("hillshade", hillshade),
        ("bpi_fine", bpi_fine),
        ("bpi_broad", bpi_broad),
        ("vrm", vrm),
    ]:
        save_geotiff(array, profile, output_dir / f"{name}.tif")

    # 5. Compute and save statistics
    click.echo("  Computing statistics...")
    stats = compute_stats(
        {"depth": elevation, "slope": slope, "bpi_fine": bpi_fine,
         "bpi_broad": bpi_broad, "vrm": vrm},
        mask,
    )
    stats["slug"] = site.slug
    stats["name"] = site.name
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
        elevation, slope, bpi_fine, bpi_broad, vrm,
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


if __name__ == "__main__":
    cli()
