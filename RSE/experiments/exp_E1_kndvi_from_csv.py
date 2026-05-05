"""
E1 — kNDVI validation and reformatting from GEE pixel-level CSV exports.

Google Earth Engine exports one CSV per region to Google Drive.
Each row is one pixel × one month (NOT a region mean).

Expected GEE CSV columns (from exp_E1_gee.js):
    region     : region label
    date       : "YYYY-MM-DD" string
    pixel_id   : "lon_lat" string identifying the pixel
    latitude   : pixel centroid latitude (float)
    longitude  : pixel centroid longitude (float)
    kNDVI      : kNDVI value computed in GEE using tanh (Smith & Boers 2024)

This script:
1. Optionally fetches CSVs directly from a Google Drive shared folder (--gdrive-folder).
2. Reads each CSV per region from data/raw/kndvi/.
3. Validates that all kNDVI values lie in [-1, 1].
4. Parses dates, ensures pixel_id / lat / lon columns are present.
5. Computes per-pixel yearly summary statistics for QC.
6. Saves one Parquet file per region:
       data/raw/kndvi/kndvi_{region}_2000_2025.parquet

kNDVI formula (Smith & Boers 2024 — TANH, not TAN):
    kNDVI = np.tanh(((nir - red) / (2 * sigma)) ** 2)

Usage (Hydra)
-------------
    # Download from Google Drive then process
    python experiments/exp_E1_kndvi_from_csv.py \\
        --gdrive-folder https://drive.google.com/drive/folders/FOLDER_ID

    # Process local CSVs already in data/raw/kndvi/
    python experiments/exp_E1_kndvi_from_csv.py

    python experiments/exp_E1_kndvi_from_csv.py run_mvp_only=true
    python experiments/exp_E1_kndvi_from_csv.py --write-config
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Module-level storage for pre-Hydra CLI args (set in __main__ block)
# ---------------------------------------------------------------------------

_GDRIVE_FOLDER_ID: str = ""
_DELETE_FROM_DRIVE: bool = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONF_PATH = _REPO_ROOT / "conf" / "resilience_forest.yaml"

# ---------------------------------------------------------------------------
# kNDVI computation
# ---------------------------------------------------------------------------

# MODIS MOD13A3 reflectance scale factor.
_MODIS_SCALE = 0.0001


def compute_kndvi(nir: np.ndarray, red: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Compute kNDVI using the Smith & Boers (2024) TANH correction.

    Parameters
    ----------
    nir, red : array-like
        Near-infrared and red reflectance values.  Raw MOD13A3 integer
        values must be multiplied by 0.0001 before calling this function.
    sigma : float
        Kernel width parameter (default 0.5, per Smith & Boers 2024).

    Returns
    -------
    np.ndarray
        kNDVI values in [-1, 1] (tanh maps R → (-1, 1)).

    Notes
    -----
    Formula: kNDVI = tanh(((NIR - RED) / (2 * sigma)) ** 2)
    This is the TANH (hyperbolic tangent) form, NOT the tangent (tan) form.
    The distinction matters: tan is unbounded; tanh is bounded in (-1, 1).
    Reference: Smith & Boers (2024), Nature Communications.
    """
    ratio = (np.asarray(nir, dtype=float) - np.asarray(red, dtype=float)) / (2.0 * sigma)
    return np.tanh(ratio ** 2)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _find_input_files(kndvi_dir: Path, region: str) -> List[Path]:
    """Return all CSV / Parquet files matching a region prefix, sorted."""
    candidates: List[Path] = []
    for pattern in (f"kndvi_{region}*.csv", f"kndvi_{region}*.parquet"):
        candidates.extend(sorted(kndvi_dir.glob(pattern)))
    return candidates


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

_GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
_TOKEN_PATH    = Path.home() / ".config" / "gdrive_token.json"
_CREDS_PATH    = Path.home() / ".config" / "gdrive_credentials.json"


def _extract_folder_id(folder_arg: str) -> str:
    """Accept a full Drive URL or a bare folder ID; return the ID."""
    if not folder_arg:
        return ""
    m = re.search(r"folders/([A-Za-z0-9_-]+)", folder_arg)
    return m.group(1) if m else folder_arg.strip()


def _gdrive_auth():
    """Return an authenticated Drive v3 service, caching the token."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        log.error(
            "Google Drive packages missing. Run:\n"
            "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )
        sys.exit(1)

    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _GDRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDS_PATH.exists():
                log.error(
                    "Google OAuth credentials not found at %s.\n"
                    "Download OAuth 2.0 Desktop credentials from "
                    "https://console.cloud.google.com/apis/credentials "
                    "and save them there.",
                    _CREDS_PATH,
                )
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), _GDRIVE_SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")

    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds)


def _gdrive_list_csvs(service, folder_id: str) -> list[dict]:
    """Return list of {id, name} for all CSV files in a Drive folder."""
    files: list[dict] = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='text/csv' and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=200,
            pageToken=page_token,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _gdrive_stream_df(service, file_id: str) -> pd.DataFrame:
    """Stream a Drive CSV directly into a DataFrame without touching disk."""
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf)


def _gdrive_delete_file(service, file_id: str, name: str) -> None:
    """Permanently delete a file from Drive."""
    service.files().delete(fileId=file_id).execute()
    log.info("Deleted from Drive: %s", name)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_region_file(path: Path, mvp_nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Parameters
    ----------
    path : Path
        File path.
    mvp_nrows : int or None
        If set, load only this many rows (MVP smoke-test mode).
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=mvp_nrows)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        if mvp_nrows is not None:
            df = df.iloc[:mvp_nrows]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_kndvi_range(
    series: pd.Series,
    label: str,
) -> Tuple[int, int]:
    """Check that kNDVI values lie in [-1, 1].

    Returns (n_valid, n_out_of_range).  Logs a warning if any values are
    outside the valid range.
    """
    finite = series.dropna()
    n_total = len(finite)
    out_of_range_mask = (finite < -1.0 - 1e-6) | (finite > 1.0 + 1e-6)
    n_bad = int(out_of_range_mask.sum())
    if n_bad > 0:
        log.warning(
            "%s: %d / %d kNDVI values outside [-1, 1] — check scale factor.",
            label, n_bad, n_total,
        )
    else:
        log.info("%s: all %d kNDVI values within [-1, 1]. ✓", label, n_total)
    return n_total - n_bad, n_bad


# ---------------------------------------------------------------------------
# Processing pipeline per region
# ---------------------------------------------------------------------------


def _process_df(df: pd.DataFrame, region: str) -> Optional[pd.DataFrame]:
    """Validate, normalise, and QC a raw GEE kNDVI DataFrame.

    Shared by both the local-file path and the Google Drive streaming path.
    Returns None if the DataFrame is structurally invalid.
    """
    # --- Normalise column names ---
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: c.lower() for c in df.columns})

    # --- Validate required columns ---
    required_cols = {"date", "region", "pixel_id", "latitude", "longitude", "kndvi"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(
            "%s: missing columns %s. Available: %s",
            region, missing, list(df.columns),
        )
        return None

    # --- Parse dates ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad = int(df["date"].isna().sum())
    if n_bad:
        log.warning("%s: %d rows with unparseable dates dropped.", region, n_bad)
        df = df.dropna(subset=["date"])

    # --- Cast numeric columns ---
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["kndvi"]     = pd.to_numeric(df["kndvi"],     errors="coerce")
    df = df.dropna(subset=["kndvi", "latitude", "longitude"])

    # --- Validate kNDVI range [-1, 1] ---
    validate_kndvi_range(df["kndvi"], label=region)

    # --- Pixel count QC ---
    n_pixels = df["pixel_id"].nunique()
    n_dates  = df["date"].nunique()
    n_rows   = len(df)
    log.info("%s: %d pixels × %d dates = %d rows", region, n_pixels, n_dates, n_rows)
    completeness = n_rows / max(n_pixels * n_dates, 1)
    if completeness < 0.7:
        log.warning("%s: completeness %.1f%% — possible cloud gaps.", region, completeness * 100)

    # --- Yearly summary ---
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year")["kndvi"].agg(mean="mean", std="std", n="count")
    log.info("%s: yearly kNDVI (first 5 years):\n%s", region, yearly.head(5).to_string())

    df["region"] = region
    return df


def process_region(
    region: str,
    kndvi_dir: Path,
    sigma: float,
    mvp_nrows: Optional[int],
) -> Optional[pd.DataFrame]:
    """Load from local files, validate, and return processed DataFrame."""
    files = _find_input_files(kndvi_dir, region)
    if not files:
        log.warning("No input files found for region: %s  (dir: %s)", region, kndvi_dir)
        return None

    frames: List[pd.DataFrame] = []
    for fpath in files:
        log.info("Reading: %s", fpath)
        try:
            df = _read_region_file(fpath, mvp_nrows)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to read %s: %s", fpath, exc)
            continue
        frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    log.info("Region %s: %d rows from %d file(s).", region, len(df), len(files))
    return _process_df(df, region)


def process_from_drive(
    folder_id: str,
    all_regions: List[str],
    kndvi_dir: Path,
    delete_after: bool,
) -> Tuple[int, int]:
    """Stream CSVs from Drive, process in memory, save Parquet, optionally delete.

    Returns (processed_count, skipped_count).
    """
    log.info("Authenticating with Google Drive …")
    service = _gdrive_auth()

    log.info("Listing CSV files in folder %s …", folder_id)
    drive_files = _gdrive_list_csvs(service, folder_id)
    log.info("Found %d CSV file(s) on Drive.", len(drive_files))

    if not drive_files:
        log.warning("No CSV files found in the Drive folder.")
        return 0, 0

    kndvi_dir.mkdir(parents=True, exist_ok=True)
    processed, skipped = 0, 0

    for file_meta in drive_files:
        fid   = file_meta["id"]
        fname = file_meta["name"]

        # Match file to a known region by filename
        region = next(
            (r for r in all_regions if f"kndvi_{r}" in fname.lower()),
            None,
        )
        if region is None:
            log.warning("Cannot match %s to any region — skipping.", fname)
            skipped += 1
            continue

        log.info("Streaming %s (region: %s) …", fname, region)
        try:
            df_raw = _gdrive_stream_df(service, fid)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to stream %s: %s", fname, exc)
            skipped += 1
            continue

        df_proc = _process_df(df_raw, region)
        if df_proc is None:
            skipped += 1
            continue

        out_path = kndvi_dir / f"kndvi_{region}_2000_2025.parquet"
        df_proc.to_parquet(out_path, index=False, engine="pyarrow")
        log.info("Saved → %s  (%d rows)", out_path.name, len(df_proc))
        processed += 1

        if delete_after:
            try:
                _gdrive_delete_file(service, fid, fname)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to delete %s from Drive: %s", fname, exc)

    return processed, skipped


# ---------------------------------------------------------------------------
# Config write helper
# ---------------------------------------------------------------------------


def _write_default_config() -> None:
    if _CONF_PATH.exists():
        print(f"Config at {_CONF_PATH}")
        print(_CONF_PATH.read_text(encoding="utf-8"))
    else:
        print(f"Config not found at {_CONF_PATH}.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_CONF_PATH.parent),
    config_name=_CONF_PATH.stem,
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("=== E1 kNDVI Validation & Reformat ===")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    run_mvp = bool(cfg.run_mvp_only)
    mvp_nrows: Optional[int] = 24 if run_mvp else None
    if run_mvp:
        log.warning("MVP mode: loading first 24 rows per file only.")

    sigma: float = float(cfg.kndvi.sigma)
    kndvi_dir = _REPO_ROOT / cfg.data.kndvi_dir
    kndvi_dir.mkdir(parents=True, exist_ok=True)

    all_regions: List[str] = list(cfg.regions.continents) + list(cfg.regions.subregions)

    if _GDRIVE_FOLDER_ID:
        log.info("Google Drive mode: folder %s  (delete_after=%s)", _GDRIVE_FOLDER_ID, _DELETE_FROM_DRIVE)
        processed_count, skipped_count = process_from_drive(
            folder_id=_GDRIVE_FOLDER_ID,
            all_regions=all_regions,
            kndvi_dir=kndvi_dir,
            delete_after=_DELETE_FROM_DRIVE,
        )
    else:
        processed_count = 0
        skipped_count = 0
        for region in all_regions:
            df_region = process_region(region, kndvi_dir, sigma, mvp_nrows)
            if df_region is None:
                skipped_count += 1
                continue
            out_path = kndvi_dir / f"kndvi_{region}_2000_2025.parquet"
            df_region.to_parquet(out_path, index=False, engine="pyarrow")
            log.info("Saved → %s  (%d rows)", out_path, len(df_region))
            processed_count += 1

        if skipped_count > 0 and processed_count == 0:
            log.warning(
                "No input files found in %s. Pass --gdrive-folder or place GEE CSVs there.",
                kndvi_dir,
            )

    log.info(
        "=== E1 kNDVI complete: %d regions processed, %d skipped ===",
        processed_count, skipped_count,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--write-config", action="store_true",
                        help="Print the default config and exit.")
    parser.add_argument(
        "--gdrive-folder", default="",
        metavar="URL_OR_ID",
        help="Google Drive folder URL or ID containing GEE-exported CSVs. "
             "Streams each file directly without downloading to local disk.",
    )
    parser.add_argument(
        "--delete-from-drive", action="store_true",
        help="Delete each CSV from Drive after it has been processed and saved as Parquet.",
    )
    known, remaining = parser.parse_known_args()

    if known.write_config:
        _write_default_config()

    # Assignments here are at module level → update the globals that main() reads.
    _GDRIVE_FOLDER_ID  = _extract_folder_id(known.gdrive_folder)
    _DELETE_FROM_DRIVE = known.delete_from_drive

    sys.argv = [sys.argv[0]] + remaining
    main()
