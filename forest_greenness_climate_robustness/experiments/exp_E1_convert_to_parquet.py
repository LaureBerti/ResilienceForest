from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
GDRIVE_FILE_IDS: dict[str, str] = {'africa': '1M01g5BWAkyuFHKsPPJJ2aQPDSGHxxCsp'}
SIGMA = 0.5
CHUNK_SIZE = 500000
_REPO_ROOT = Path(__file__).resolve().parent.parent
_KNDVI_DIR = _REPO_ROOT / 'data' / 'raw' / 'kndvi'
_CACHE_DIR = _REPO_ROOT / 'data' / 'raw' / 'kndvi' / '_downloads'
MODIS_NDVI_SCALE = 10000.0

def _detect_and_fix_scale(series: pd.Series, col_name: str) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series
    vmax = valid.abs().max()
    if vmax > 2.0:
        log.warning("Column '%s': max |value| = %.1f — looks like raw MODIS integer scale. Dividing by %.0f to get physical reflectance / NDVI.", col_name, vmax, MODIS_NDVI_SCALE)
        return series / MODIS_NDVI_SCALE
    log.info("Column '%s': max |value| = %.4f — already in physical scale.", col_name, vmax)
    return series

def compute_kndvi_exact(nir: np.ndarray, red: np.ndarray, sigma: float=0.5) -> np.ndarray:
    ratio = (nir - red) / (2.0 * sigma)
    return np.tanh(ratio ** 2)

def compute_kndvi_from_ndvi(ndvi: np.ndarray) -> np.ndarray:
    return np.tanh(ndvi ** 2)

def download_gdrive(file_id: str, dest: Path, skip_if_exists: bool=True) -> Path:
    if skip_if_exists and dest.exists():
        log.info('Already downloaded: %s  (delete to re-download)', dest)
        return dest
    try:
        import gdown
    except ImportError:
        log.error('gdown is not installed. Run: pip install gdown\nOr place the CSV manually at: %s', dest)
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info('Downloading from Google Drive (id=%s) → %s', file_id, dest)
    result = gdown.download(id=file_id, output=str(dest), quiet=False)
    if result is None or not dest.exists():
        log.error("Download failed for file_id=%s.\nMake sure the file is shared as 'Anyone with the link can view'.\nAlternatively, download it manually from:\n  https://drive.google.com/open?id=%s\nand place it at:\n  %s", file_id, file_id, dest)
        sys.exit(1)
    log.info('Downloaded: %s  (%.1f MB)', dest, dest.stat().st_size / 1000000.0)
    return dest
EXPECTED_COLS = {'date', 'region', 'pixel_id', 'latitude', 'longitude'}

def process_csv(csv_path: Path, region: str, mvp: bool=False) -> Optional[pd.DataFrame]:
    log.info('Reading: %s', csv_path)
    size_mb = csv_path.stat().st_size / 1000000.0
    log.info('File size: %.1f MB', size_mb)
    chunks: list[pd.DataFrame] = []
    chunksize = CHUNK_SIZE if not mvp else 5000
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)):
        chunk.columns = [c.strip() for c in chunk.columns]
        col_lower = {c: c.lower() for c in chunk.columns}
        chunk = chunk.rename(columns=col_lower)
        has_kndvi = 'kndvi' in chunk.columns
        has_ndvi = 'ndvi' in chunk.columns
        has_nir = 'sur_refl_b02' in chunk.columns
        has_red = 'sur_refl_b01' in chunk.columns
        if not (has_kndvi or has_ndvi or (has_nir and has_red)):
            log.error('Chunk %d: no recognised NDVI/reflectance columns. Found: %s', i, list(chunk.columns))
            break
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        if has_nir and has_red:
            nir = _detect_and_fix_scale(chunk['sur_refl_b02'], 'sur_refl_b02').values
            red = _detect_and_fix_scale(chunk['sur_refl_b01'], 'sur_refl_b01').values
            chunk['kndvi'] = compute_kndvi_exact(nir, red, sigma=SIGMA)
            log.info('Chunk %d: kNDVI computed from NIR+RED (exact formula).', i)
        elif has_kndvi:
            chunk['kndvi'] = _detect_and_fix_scale(chunk['kndvi'], 'kndvi')
            log.info('Chunk %d: kNDVI loaded directly from CSV.', i)
        elif has_ndvi:
            ndvi_scaled = _detect_and_fix_scale(chunk['ndvi'], 'ndvi').values
            chunk['ndvi'] = ndvi_scaled
            chunk['kndvi'] = compute_kndvi_from_ndvi(ndvi_scaled)
            log.info('Chunk %d: kNDVI approximated from NDVI using tanh(NDVI²). For exact kNDVI, re-run GEE script to export NIR+RED bands.', i)
        chunk['region'] = region
        bad = ((chunk['kndvi'] < -1 - 1e-06) | (chunk['kndvi'] > 1 + 1e-06)).sum()
        if bad > 0:
            log.warning('Chunk %d: %d kNDVI values outside [–1, 1].', i, bad)
        chunks.append(chunk)
        log.info('Chunk %d: %d rows processed.', i, len(chunk))
        if mvp and i >= 1:
            log.warning('MVP mode: stopping after 2 chunks.')
            break
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    log.info('Region %s: %d total rows | %d unique pixels | %d dates', region, len(df), df['pixel_id'].nunique() if 'pixel_id' in df.columns else 0, df['date'].nunique() if 'date' in df.columns else 0)
    return df

def main() -> None:
    mvp = '--mvp' in sys.argv or 'run_mvp_only=true' in sys.argv
    if mvp:
        log.warning('MVP mode: processing first 10 000 rows only.')
    _KNDVI_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for region, file_id in GDRIVE_FILE_IDS.items():
        log.info('=== Processing region: %s ===', region)
        csv_dest = _CACHE_DIR / f'kndvi_{region}_pixels_ifl2020.csv'
        csv_path = download_gdrive(file_id, csv_dest)
        df = process_csv(csv_path, region, mvp=mvp)
        if df is None:
            log.error('No data produced for region %s — skipping.', region)
            continue
        out_path = _KNDVI_DIR / f'kndvi_{region}_2000_2025.parquet'
        df.to_parquet(out_path, index=False, engine='pyarrow', compression='snappy')
        size_mb = out_path.stat().st_size / 1000000.0
        log.info('Saved → %s  (%.1f MB, was %.1f MB CSV)', out_path, size_mb, csv_path.stat().st_size / 1000000.0)
        log.info('Compression ratio: %.1fx', csv_path.stat().st_size / out_path.stat().st_size)
if __name__ == '__main__':
    main()
