"""
E1 — Download MODIS MOD13A3 via NASA Earthdata, apply local IFL mask,
     output time-aligned pixel-level kNDVI Parquet.

Replaces the GEE JavaScript approach.  No GEE account required.

IFL–MODIS period matching (time-varying mask)
---------------------------------------------
Each IFL vintage masks only the MODIS months that fall within its validity
window.  Results are concatenated into a single Parquet per region.

  IFL 2000 → MODIS 2000-01 – 2012-12
  IFL 2013 → MODIS 2013-01 – 2015-12
  IFL 2016 → MODIS 2016-01 – 2019-12
  IFL 2020 → MODIS 2020-01 – 2024-12

Time alignment
--------------
Output Parquet has one row per (pixel_id × month) for the full study period
(2000-01 to 2024-12).  Months with no valid data (cloud / quality masked)
are represented as NaN kndvi rows so downstream matrix profile code receives
a gapless monthly index.

Output
------
  data/raw/kndvi/kndvi_{region}_2000_2025.parquet
  Columns: region, date, pixel_id, latitude, longitude, kndvi

Requirements
------------
  pip install earthaccess pyhdf pyproj geopandas pyarrow pandas numpy

NASA Earthdata account
----------------------
  Register at https://urs.earthdata.nasa.gov
  First run calls earthaccess.login() which caches credentials in ~/.netrc.

Usage
-----
  python experiments/exp_E1_earthaccess.py
  python experiments/exp_E1_earthaccess.py --mvp           # Africa 2020 only
  python experiments/exp_E1_earthaccess.py --regions amazon congo
  python experiments/exp_E1_earthaccess.py --ifl-years 2020
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Region bounding boxes [lon_min, lat_min, lon_max, lat_max]
# Matches exp_E1_gee.js and resilience_forest.yaml
# ---------------------------------------------------------------------------

REGION_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "africa":               (-20, -35,  55,  38),
    "asia":                 ( 25, -10, 180,  77),
    "europe":               (-25,  34,  45,  72),
    "north_america":        (-170,  5, -50,  83),
    "south_america":        (-82, -57, -33,  13),
    "australia_oceania":    (110, -50, 180,  10),
    "amazon":               (-75, -15, -45,   5),
    "congo":                ( 14,  -6,  30,   6),
    "boreal_siberia":       ( 60,  58, 120,  72),
    "southeast_asia":       ( 95, -10, 142,  15),
    "canadian_boreal":      (-140, 55,  -90, 72),
    "scandinavian_boreal":  ( 14,  59,  32,  71),
    "papua_new_guinea":     (140, -10, 156,   0),
    "russian_far_east":     (130,  44, 145,  60),
}

# IFL vintage → (start, end) of its validity window (both inclusive)
IFL_PERIODS: Dict[int, Tuple[str, str]] = {
    2000: ("2000-01-01", "2012-12-31"),
    2013: ("2013-01-01", "2015-12-31"),
    2016: ("2016-01-01", "2019-12-31"),
    2020: ("2020-01-01", "2024-12-31"),
}

# MOD13A3 HDF4 band names and parameters
_NIR_SDS   = "1 km monthly NIR reflectance"
_RED_SDS   = "1 km monthly red reflectance"
_QC_SDS    = "1 km monthly pixel reliability"
_SCALE     = 0.0001      # reflectance scale factor
_FILL_REFL = -28672      # fill value for NIR/RED
_FILL_QC   = -1          # fill value for pixel_reliability

# MODIS sinusoidal projection (ESRI:53008 / custom proj)
_SINU_PROJ = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"

# Tile dimensions for MOD13A3 (1 km product): 1200×1200
_TILE_NROWS = 1200
_TILE_NCOLS = 1200
_TILE_SIZE_M = 1111950.5196296  # metres per tile edge


# ---------------------------------------------------------------------------
# MODIS sinusoidal → WGS84
# ---------------------------------------------------------------------------

def _tile_latlon(h: int, v: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) 2-D arrays for sampled pixel centres in tile (h, v).

    Parameters
    ----------
    h, v    : MODIS tile horizontal / vertical indices (0-based, 0-35 / 0-17)
    stride  : sample every `stride` pixels (5 → ~5 km at 1-km product)
    """
    psize   = _TILE_SIZE_M / _TILE_NROWS
    x_ul    = (h - 18) * _TILE_SIZE_M
    y_ul    = (9 - v) * _TILE_SIZE_M

    rows = np.arange(0, _TILE_NROWS, stride)
    cols = np.arange(0, _TILE_NCOLS, stride)
    c_grid, r_grid = np.meshgrid(cols, rows)

    x_sinu = x_ul + (c_grid + 0.5) * psize
    y_sinu = y_ul - (r_grid + 0.5) * psize

    transformer = Transformer.from_crs(_SINU_PROJ, "EPSG:4326", always_xy=True)
    lon_flat, lat_flat = transformer.transform(x_sinu.ravel(), y_sinu.ravel())
    n = len(rows)
    m = len(cols)
    return lat_flat.reshape(n, m), lon_flat.reshape(n, m)


# ---------------------------------------------------------------------------
# HDF4 reading via pyhdf
# ---------------------------------------------------------------------------

def _read_granule(
    filepath: Path,
    stride: int,
    sigma: float,
) -> Optional[pd.DataFrame]:
    """Read one MOD13A3 HDF4 granule; return DataFrame or None on error.

    Columns: latitude, longitude, kndvi, pixel_id
    """
    try:
        from pyhdf.SD import SD, SDC  # imported here so missing dep gives clear error
    except ImportError:
        log.error("pyhdf not installed. Run: pip install pyhdf")
        sys.exit(1)

    filename = filepath.name
    m = re.search(r"h(\d{2})v(\d{2})", filename)
    if not m:
        log.warning("Cannot parse h/v from filename: %s — skipping.", filename)
        return None

    h, v = int(m.group(1)), int(m.group(2))

    try:
        hdf = SD(str(filepath), SDC.READ)
    except Exception as exc:
        log.error("Cannot open HDF4 file %s: %s", filepath, exc)
        return None

    try:
        nir_raw = hdf.select(_NIR_SDS)[:].astype(float)
        red_raw = hdf.select(_RED_SDS)[:].astype(float)
        qc_raw  = hdf.select(_QC_SDS)[:]
    except Exception as exc:
        log.error("Cannot read bands from %s: %s", filepath, exc)
        hdf.end()
        return None
    finally:
        hdf.end()

    # Apply scale factor and mask fill/invalid values
    nir_raw[nir_raw <= _FILL_REFL] = np.nan
    red_raw[red_raw <= _FILL_REFL] = np.nan
    nir = nir_raw * _SCALE
    red = red_raw * _SCALE

    # Quality mask: keep pixel_reliability 0 (good) and 1 (marginal)
    bad_qc = (qc_raw < 0) | (qc_raw > 1)
    nir[bad_qc] = np.nan
    red[bad_qc] = np.nan

    # Compute kNDVI (Smith & Boers 2024 tanh form)
    ratio  = (nir - red) / (2.0 * sigma)
    kndvi  = np.tanh(ratio ** 2)

    # Sub-sample at stride
    nir_s   = nir[::stride, ::stride]
    kndvi_s = kndvi[::stride, ::stride]

    lat, lon = _tile_latlon(h, v, stride)

    flat_lat   = lat.ravel()
    flat_lon   = lon.ravel()
    flat_kndvi = kndvi_s.ravel()

    # Drop pixels where kNDVI is NaN (masked out by quality or fill)
    valid = np.isfinite(flat_kndvi)
    if valid.sum() == 0:
        return None

    pixel_ids = [f"{lon:.4f}_{lat:.4f}" for lon, lat in zip(flat_lon[valid], flat_lat[valid])]

    return pd.DataFrame({
        "latitude":  flat_lat[valid],
        "longitude": flat_lon[valid],
        "kndvi":     flat_kndvi[valid],
        "pixel_id":  pixel_ids,
    })


# ---------------------------------------------------------------------------
# IFL spatial masking
# ---------------------------------------------------------------------------

def _mask_to_ifl(
    df: pd.DataFrame,
    ifl_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Retain only pixels whose centroid falls within an IFL polygon."""
    if df.empty:
        return df
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, ifl_gdf[["geometry"]], how="inner", predicate="within")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def _assign_regions(
    df: pd.DataFrame,
    region_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Add a 'region' column based on which bounding boxes a pixel falls in.

    A pixel may match multiple regions (e.g., amazon is inside south_america).
    Duplicate rows are created intentionally so every relevant region is covered.
    """
    frames = []
    for rname, (lon_min, lat_min, lon_max, lat_max) in region_bboxes.items():
        mask = (
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) &
            (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max)
        )
        sub = df[mask].copy()
        sub["region"] = rname
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Time alignment
# ---------------------------------------------------------------------------

def _align_monthly(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Ensure every pixel has one row per month in [start, end].

    Missing months (cloud / quality masked) become NaN kndvi rows.
    """
    if df.empty:
        return df

    full_index = pd.date_range(start=start, end=end, freq="MS")
    pixel_ids  = df["pixel_id"].unique()

    # Keep one representative lat/lon per pixel_id
    coords = (
        df[["pixel_id", "latitude", "longitude", "region"]]
        .drop_duplicates("pixel_id")
        .set_index("pixel_id")
    )

    frames = []
    for pid in pixel_ids:
        sub = df[df["pixel_id"] == pid].set_index("date")["kndvi"]
        sub = sub[~sub.index.duplicated(keep="first")]
        aligned = sub.reindex(full_index)
        frame = pd.DataFrame({"date": full_index, "kndvi": aligned.values})
        frame["pixel_id"]  = pid
        frame["latitude"]  = coords.loc[pid, "latitude"]
        frame["longitude"] = coords.loc[pid, "longitude"]
        frame["region"]    = coords.loc[pid, "region"]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _granule_date(filename: str) -> Optional[datetime]:
    """Parse acquisition date from MOD13A3 filename (AYYYYDDD format)."""
    m = re.search(r"\.A(\d{7})\.", filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%j")
    except ValueError:
        return None


def run(
    ifl_years: List[int],
    regions: List[str],
    stride: int,
    sigma: float,
    download_dir: Path,
    output_dir: Path,
    mvp: bool,
) -> None:
    try:
        import earthaccess
    except ImportError:
        log.error("earthaccess not installed. Run: pip install earthaccess")
        sys.exit(1)

    log.info("Authenticating with NASA Earthdata …")
    earthaccess.login(persist=True)  # reads ~/.netrc; prompts interactively on first run and saves credentials

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    ifl_dir = _REPO_ROOT / "data" / "raw" / "ifl"

    # Restrict to requested regions
    target_bboxes = {r: REGION_BBOXES[r] for r in regions if r in REGION_BBOXES}
    unknown = set(regions) - set(REGION_BBOXES)
    if unknown:
        log.warning("Unknown regions ignored: %s", unknown)

    # Accumulate results per region across all IFL periods
    region_frames: Dict[str, List[pd.DataFrame]] = {r: [] for r in target_bboxes}

    for ifl_year, (period_start, period_end) in IFL_PERIODS.items():
        if ifl_year not in ifl_years:
            continue

        ifl_parquet = ifl_dir / f"IFL_{ifl_year}.parquet"
        if not ifl_parquet.exists():
            log.warning("IFL parquet not found: %s — run exp_E1_convert_ifl_gpkg.py first", ifl_parquet)
            continue

        log.info("Loading IFL %d mask …", ifl_year)
        ifl_gdf = gpd.read_parquet(ifl_parquet)

        if mvp:
            period_end = period_start[:4] + "-01-31"  # first month only
            log.warning("MVP mode: restricting to %s – %s", period_start, period_end)

        log.info("IFL %d  →  MODIS period %s – %s", ifl_year, period_start, period_end)

        # Global bbox covering all target regions
        all_lons = [b[0] for b in target_bboxes.values()] + [b[2] for b in target_bboxes.values()]
        all_lats = [b[1] for b in target_bboxes.values()] + [b[3] for b in target_bboxes.values()]
        global_bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))

        log.info("Searching MOD13A3 granules …")
        results = earthaccess.search_data(
            short_name="MOD13A3",
            version="061",
            temporal=(period_start[:10], period_end[:10]),
            bounding_box=global_bbox,
        )
        log.info("Found %d granules for IFL %d period.", len(results), ifl_year)

        if not results:
            log.warning("No granules found — check dates and bbox.")
            continue

        log.info("Downloading granules to %s …", download_dir)
        files = earthaccess.download(results, str(download_dir))
        log.info("Downloaded %d files.", len(files))

        # Process each granule
        for fpath in sorted(Path(f) for f in files):
            acq_date = _granule_date(fpath.name)
            if acq_date is None:
                log.warning("Cannot parse date from %s — skipping.", fpath.name)
                continue

            # MOD13A3 is monthly; use the 1st of the acquisition month
            month_date = pd.Timestamp(acq_date.year, acq_date.month, 1)

            log.debug("Processing %s (%s) …", fpath.name, month_date.strftime("%Y-%m"))
            pixel_df = _read_granule(fpath, stride, sigma)
            if pixel_df is None or pixel_df.empty:
                continue

            # Apply IFL mask
            masked = _mask_to_ifl(pixel_df, ifl_gdf)
            if masked.empty:
                continue

            # Assign to study regions
            assigned = _assign_regions(masked, target_bboxes)
            if assigned.empty:
                continue

            assigned["date"] = month_date

            for region, grp in assigned.groupby("region"):
                region_frames[region].append(grp)

    # Combine periods, align to full monthly index, save
    full_start = "2000-01-01"
    full_end   = "2024-12-31"

    for region, frames in region_frames.items():
        if not frames:
            log.warning("No data collected for region: %s", region)
            continue

        log.info("Aligning %s to monthly index …", region)
        combined = pd.concat(frames, ignore_index=True)
        aligned  = _align_monthly(combined, full_start, full_end)

        out_path = output_dir / f"kndvi_{region}_2000_2025.parquet"
        aligned.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        n_pixels = aligned["pixel_id"].nunique()
        n_months = aligned["date"].nunique()
        log.info("Saved %s  [%d pixels × %d months = %d rows]",
                 out_path.name, n_pixels, n_months, len(aligned))

    log.info("=== E1 earthaccess complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ifl-years", nargs="+", type=int, default=list(IFL_PERIODS.keys()),
        metavar="YEAR",
        help=f"IFL vintages to process (default: all {list(IFL_PERIODS)})",
    )
    p.add_argument(
        "--regions", nargs="+", default=list(REGION_BBOXES.keys()),
        metavar="REGION",
        help="Study regions to process (default: all 14)",
    )
    p.add_argument(
        "--stride", type=int, default=5,
        help="Pixel sampling stride (1=1 km, 5=~5 km; default: 5)",
    )
    p.add_argument(
        "--sigma", type=float, default=0.5,
        help="kNDVI kernel width σ (default: 0.5, per Smith & Boers 2024)",
    )
    p.add_argument(
        "--download-dir", type=Path,
        default=_REPO_ROOT / "data" / "raw" / "modis",
        help="Directory to cache downloaded HDF4 files",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "data" / "raw" / "kndvi",
        help="Directory for output Parquet files",
    )
    p.add_argument(
        "--mvp", action="store_true",
        help="Smoke test: first month only, Africa and Amazon only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mvp:
        args.regions = ["africa", "amazon"]
        log.warning("MVP mode: regions restricted to africa + amazon, first month of each period only.")

    run(
        ifl_years=args.ifl_years,
        regions=args.regions,
        stride=args.stride,
        sigma=args.sigma,
        download_dir=args.download_dir,
        output_dir=args.output_dir,
        mvp=args.mvp,
    )
