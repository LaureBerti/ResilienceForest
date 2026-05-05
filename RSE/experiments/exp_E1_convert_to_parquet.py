"""
E1 — Download GEE-exported NDVI CSV from Google Drive, fix scale, compute kNDVI,
     and save as Parquet.

WHAT THIS SCRIPT DOES
---------------------
1. Downloads the CSV file from Google Drive (using the file ID).
2. Reads it in chunks (handles files larger than memory).
3. Auto-detects whether NDVI values are in raw MODIS integer scale (0–10000)
   or already in the physical range (–1 to 1):
     • Raw scale  → divide by 10000 → NDVI in [–1, 1]
     • Correct scale already → no change
4. Computes kNDVI from NDVI using the tanh approximation
   (valid when sigma = mean reflectance, i.e. the self-adaptive kernel):
     kNDVI_approx ≈ tanh(NDVI²)
   NOTE: if your CSV contains raw NIR and RED bands (sur_refl_b02, sur_refl_b01),
   the script uses the exact Camps-Valls (2021) formula instead:
     kNDVI_exact = tanh(((NIR − RED) / (2 × sigma))²)
5. Saves one Parquet file per region/batch to data/raw/kndvi/.

USAGE
-----
    # Install gdown first if not present:
    pip install gdown

    # Run with default Google Drive file ID (set GDRIVE_FILE_IDS below):
    python experiments/exp_E1_convert_to_parquet.py

    # Override via Hydra or command line:
    python experiments/exp_E1_convert_to_parquet.py run_mvp_only=true

GOOGLE DRIVE FILE ID
--------------------
The file ID is the string after `id=` in your share link:
    https://drive.google.com/open?id=1M01g5BWAkyuFHKsPPJJ2aQPDSGHxxCsp
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Set this in GDRIVE_FILE_IDS below, or pass on the command line.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these lines
# ---------------------------------------------------------------------------

# One entry per GEE export file; key = region name, value = Google Drive file ID.
# Get the file ID from your share link:  drive.google.com/open?id=<FILE_ID>
GDRIVE_FILE_IDS: dict[str, str] = {
    "africa":               "1M01g5BWAkyuFHKsPPJJ2aQPDSGHxxCsp",  # ← your file
    # "asia":               "<file_id>",
    # "europe":             "<file_id>",
    # "north_america":      "<file_id>",
    # "south_america":      "<file_id>",
    # "australia_oceania":  "<file_id>",
    # "amazon":             "<file_id>",
    # "congo":              "<file_id>",
}

SIGMA = 0.5          # kNDVI kernel width (Camps-Valls 2021)
CHUNK_SIZE = 500_000  # rows per chunk (reduce if RAM is limited)

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_KNDVI_DIR  = _REPO_ROOT / "data" / "raw" / "kndvi"
_CACHE_DIR  = _REPO_ROOT / "data" / "raw" / "kndvi" / "_downloads"

# ---------------------------------------------------------------------------
# SCALE DETECTION
# ---------------------------------------------------------------------------

MODIS_NDVI_SCALE = 10_000.0  # MODIS stores NDVI as integer × 10000

def _detect_and_fix_scale(series: pd.Series, col_name: str) -> pd.Series:
    """
    If values are clearly in raw MODIS integer scale (max > 2), divide by 10000.
    Logs the decision so the user can verify.
    """
    valid = series.dropna()
    if valid.empty:
        return series
    vmax = valid.abs().max()
    if vmax > 2.0:
        log.warning(
            "Column '%s': max |value| = %.1f — looks like raw MODIS integer scale. "
            "Dividing by %.0f to get physical reflectance / NDVI.",
            col_name, vmax, MODIS_NDVI_SCALE,
        )
        return series / MODIS_NDVI_SCALE
    log.info("Column '%s': max |value| = %.4f — already in physical scale.", col_name, vmax)
    return series

# ---------------------------------------------------------------------------
# kNDVI COMPUTATION
# ---------------------------------------------------------------------------

def compute_kndvi_exact(nir: np.ndarray, red: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Exact Camps-Valls (2021) formula: tanh(((NIR-RED)/(2σ))²).  TANH not TAN."""
    ratio = (nir - red) / (2.0 * sigma)
    return np.tanh(ratio ** 2)

def compute_kndvi_from_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Approximation: kNDVI ≈ tanh(NDVI²).
    Valid when the kernel width σ equals the mean of NIR and RED (self-adaptive kernel).
    Use this only when raw NIR/RED bands are not available.
    Reference: Camps-Valls et al. (2021), Eq. 8 with σ = (NIR+RED)/2.
    """
    return np.tanh(ndvi ** 2)

# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def download_gdrive(file_id: str, dest: Path, skip_if_exists: bool = True) -> Path:
    """Download a file from Google Drive using gdown.

    The file must be publicly shared ("Anyone with the link can view").
    If gdown fails, place the CSV manually at `dest` and re-run.
    """
    if skip_if_exists and dest.exists():
        log.info("Already downloaded: %s  (delete to re-download)", dest)
        return dest
    try:
        import gdown  # noqa: PLC0415
    except ImportError:
        log.error(
            "gdown is not installed. Run: pip install gdown\n"
            "Or place the CSV manually at: %s", dest,
        )
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading from Google Drive (id=%s) → %s", file_id, dest)

    # gdown 6.x: pass file ID directly; fuzzy parameter was removed in 6.0
    result = gdown.download(id=file_id, output=str(dest), quiet=False)

    if result is None or not dest.exists():
        log.error(
            "Download failed for file_id=%s.\n"
            "Make sure the file is shared as 'Anyone with the link can view'.\n"
            "Alternatively, download it manually from:\n"
            "  https://drive.google.com/open?id=%s\n"
            "and place it at:\n"
            "  %s",
            file_id, file_id, dest,
        )
        sys.exit(1)

    log.info("Downloaded: %s  (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest

# ---------------------------------------------------------------------------
# PROCESS ONE CSV
# ---------------------------------------------------------------------------

EXPECTED_COLS = {"date", "region", "pixel_id", "latitude", "longitude"}

def process_csv(csv_path: Path, region: str, mvp: bool = False) -> Optional[pd.DataFrame]:
    """
    Read, validate, scale-fix, and compute kNDVI for one GEE export CSV.
    Returns a DataFrame ready to save as Parquet.
    """
    log.info("Reading: %s", csv_path)
    size_mb = csv_path.stat().st_size / 1e6
    log.info("File size: %.1f MB", size_mb)

    chunks: list[pd.DataFrame] = []
    chunksize = CHUNK_SIZE if not mvp else 5_000

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)):
        # Normalise column names
        chunk.columns = [c.strip() for c in chunk.columns]
        col_lower = {c: c.lower() for c in chunk.columns}
        chunk = chunk.rename(columns=col_lower)

        # ---- Detect what NDVI/reflectance columns are present ----
        has_kndvi   = "kndvi" in chunk.columns
        has_ndvi    = "ndvi"  in chunk.columns
        has_nir     = "sur_refl_b02" in chunk.columns
        has_red     = "sur_refl_b01" in chunk.columns

        if not (has_kndvi or has_ndvi or (has_nir and has_red)):
            log.error(
                "Chunk %d: no recognised NDVI/reflectance columns. "
                "Found: %s", i, list(chunk.columns),
            )
            break

        # ---- Parse date ----
        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

        # ---- Fix scale and compute kNDVI ----
        if has_nir and has_red:
            # Best case: raw reflectance bands available → exact kNDVI formula
            nir = _detect_and_fix_scale(chunk["sur_refl_b02"], "sur_refl_b02").values
            red = _detect_and_fix_scale(chunk["sur_refl_b01"], "sur_refl_b01").values
            chunk["kndvi"] = compute_kndvi_exact(nir, red, sigma=SIGMA)
            log.info("Chunk %d: kNDVI computed from NIR+RED (exact formula).", i)

        elif has_kndvi:
            # kNDVI already in the file (our exp_E1_gee.js output)
            chunk["kndvi"] = _detect_and_fix_scale(chunk["kndvi"], "kndvi")
            log.info("Chunk %d: kNDVI loaded directly from CSV.", i)

        elif has_ndvi:
            # Only NDVI available → scale fix then tanh approximation
            ndvi_scaled = _detect_and_fix_scale(chunk["ndvi"], "ndvi").values
            chunk["ndvi"]  = ndvi_scaled
            chunk["kndvi"] = compute_kndvi_from_ndvi(ndvi_scaled)
            log.info(
                "Chunk %d: kNDVI approximated from NDVI using tanh(NDVI²). "
                "For exact kNDVI, re-run GEE script to export NIR+RED bands.", i,
            )

        # ---- Tag region ----
        chunk["region"] = region

        # ---- Validate kNDVI range ----
        bad = ((chunk["kndvi"] < -1 - 1e-6) | (chunk["kndvi"] > 1 + 1e-6)).sum()
        if bad > 0:
            log.warning("Chunk %d: %d kNDVI values outside [–1, 1].", i, bad)

        chunks.append(chunk)
        log.info("Chunk %d: %d rows processed.", i, len(chunk))

        if mvp and i >= 1:
            log.warning("MVP mode: stopping after 2 chunks.")
            break

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    log.info(
        "Region %s: %d total rows | %d unique pixels | %d dates",
        region,
        len(df),
        df["pixel_id"].nunique() if "pixel_id" in df.columns else 0,
        df["date"].nunique()      if "date"     in df.columns else 0,
    )
    return df

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    mvp = "--mvp" in sys.argv or "run_mvp_only=true" in sys.argv
    if mvp:
        log.warning("MVP mode: processing first 10 000 rows only.")

    _KNDVI_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for region, file_id in GDRIVE_FILE_IDS.items():
        log.info("=== Processing region: %s ===", region)

        # Download CSV from Google Drive
        csv_dest = _CACHE_DIR / f"kndvi_{region}_pixels_ifl2020.csv"
        csv_path = download_gdrive(file_id, csv_dest)

        # Read, fix scale, compute kNDVI
        df = process_csv(csv_path, region, mvp=mvp)
        if df is None:
            log.error("No data produced for region %s — skipping.", region)
            continue

        # Save as Parquet
        out_path = _KNDVI_DIR / f"kndvi_{region}_2000_2025.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        size_mb = out_path.stat().st_size / 1e6
        log.info("Saved → %s  (%.1f MB, was %.1f MB CSV)", out_path, size_mb,
                 csv_path.stat().st_size / 1e6)
        log.info("Compression ratio: %.1fx", csv_path.stat().st_size / out_path.stat().st_size)


if __name__ == "__main__":
    main()
