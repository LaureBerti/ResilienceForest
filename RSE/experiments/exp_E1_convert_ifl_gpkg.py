"""
E1 — Convert IFL GeoPackage files to GeoParquet.

Converts all IFL vintage files (2000, 2013, 2016, 2020, 2025) from .gpkg to
GeoParquet format.  GeoParquet loads ~10× faster than .gpkg and is the
preferred format for downstream spatial joins in E1–E8.

Input:   data/raw/ifl/IFL_<year>.gpkg  (WGS84 / EPSG:4326, MultiPolygon)
Output:  data/raw/ifl/IFL_<year>.parquet

Usage
-----
    python experiments/exp_E1_convert_ifl_gpkg.py
    python experiments/exp_E1_convert_ifl_gpkg.py --years 2020      # single year
    python experiments/exp_E1_convert_ifl_gpkg.py --years 2020 2025  # subset

Downstream use
--------------
    import geopandas as gpd
    ifl = gpd.read_parquet("data/raw/ifl/IFL_2020.parquet")
    # spatial join on pixel coordinates:
    pixels_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    ifl_pixels = gpd.sjoin(pixels_gdf, ifl[["geometry"]], how="inner", predicate="within")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_RAW = _REPO_ROOT / "data" / "raw" / "ifl"

ALL_YEARS = [2000, 2013, 2016, 2020, 2025]


def convert_one(year: int) -> bool:
    gpkg_path = _DATA_RAW / f"IFL_{year}.gpkg"
    parquet_path = _DATA_RAW / f"IFL_{year}.parquet"

    if not gpkg_path.exists():
        log.warning("Not found, skipping: %s", gpkg_path)
        return False

    log.info("Reading %s …", gpkg_path.name)
    gdf = gpd.read_file(gpkg_path)

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        log.info("Reprojecting to EPSG:4326 …")
        gdf = gdf.to_crs(epsg=4326)

    # Keep only geometry column — drop any QGIS/metadata columns to slim the file
    keep_cols = [c for c in gdf.columns if c == "geometry"]
    gdf = gdf[keep_cols]

    log.info(
        "Writing %s  [%d polygons, %.1f MB gpkg → parquet] …",
        parquet_path.name,
        len(gdf),
        gpkg_path.stat().st_size / 1e6,
    )
    gdf.to_parquet(parquet_path, index=False, compression="snappy")

    parquet_mb = parquet_path.stat().st_size / 1e6
    log.info("Done: %s  (%.1f MB)", parquet_path.name, parquet_mb)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IFL .gpkg to GeoParquet")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=ALL_YEARS,
        metavar="YEAR",
        help=f"IFL vintage years to convert (default: {ALL_YEARS})",
    )
    args = parser.parse_args()

    years = sorted(set(args.years))
    log.info("Converting IFL vintages: %s", years)

    ok = sum(convert_one(y) for y in years)
    log.info("Converted %d / %d files.", ok, len(years))

    if ok == 0:
        log.error("No files converted — check that .gpkg files are in %s", _DATA_RAW)
        sys.exit(1)


if __name__ == "__main__":
    main()
