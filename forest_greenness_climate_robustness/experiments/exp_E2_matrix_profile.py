from __future__ import annotations
import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hydra
import numpy as np
import pandas as pd
import stumpy
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('E2')
BASE_DIR = Path(__file__).resolve().parent.parent

def _resolve(cfg: DictConfig, relpath: str) -> Path:
    return BASE_DIR / relpath

def _load_kndvi(parquet_path: Path, region: str) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f'kNDVI parquet not found: {parquet_path}. Run E1 first.')
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    required = {'date', 'pixel_id', 'lat', 'lon', 'kndvi', 'region'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns {missing} in {parquet_path}. Check E1 output schema.')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'pixel_id']).reset_index(drop=True)
    return df

def _mean_ts(df: pd.DataFrame) -> np.ndarray:
    ts = df.groupby('date')['kndvi'].mean().sort_index().to_numpy(dtype=np.float64)
    return ts

def _deseasonalize(ts: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(ts, index=dates, dtype=np.float64)
    climatology = s.groupby(s.index.month).transform('mean')
    return (s - climatology).to_numpy(dtype=np.float64)

def _date_labels(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df['date'].unique()))

def _run_matrix_profile(ts: np.ndarray, m: int) -> np.ndarray:
    if len(ts) < 2 * m:
        raise ValueError(f'Time series length {len(ts)} is less than 2*m={2 * m}. Cannot compute matrix profile.')
    return stumpy.stump(ts, m)

def _extract_motifs(mp_array: np.ndarray, ts: np.ndarray, m: int, n_motifs: int) -> list[dict[str, Any]]:
    mp = mp_array[:, 0].astype(np.float64)
    motif_distances, motif_indices = stumpy.motifs(ts, mp, max_motifs=n_motifs)
    records: list[dict[str, Any]] = []
    for rank in range(len(motif_indices)):
        idx_row = np.asarray(motif_indices[rank])
        dist_row = np.asarray(motif_distances[rank], dtype=np.float64)
        keep = (idx_row >= 0) & np.isfinite(dist_row)
        idx_valid = idx_row[keep]
        dist_valid = dist_row[keep]
        if len(idx_valid) < 1:
            continue
        start_idx = int(idx_valid[0])
        nn_idx = int(idx_valid[1]) if len(idx_valid) > 1 else -1
        distance = float(dist_valid[1]) if len(dist_valid) > 1 else np.nan
        records.append({'motif_rank': rank + 1, 'start_idx': start_idx, 'nn_idx': nn_idx, 'distance': distance})
    return records

def _extract_discords(mp_array: np.ndarray, m: int, n_discords: int) -> list[dict[str, Any]]:
    mp = mp_array[:, 0].astype(np.float64)
    mp_indices = mp_array[:, 1].astype(np.int64)
    excl_zone = max(1, m // 4)
    working = mp.copy()
    working[np.isinf(working)] = np.nan
    records: list[dict[str, Any]] = []
    for rank in range(1, n_discords + 1):
        if np.all(np.isnan(working)):
            break
        idx = int(np.nanargmax(working))
        records.append({'discord_rank': rank, 'start_idx': idx, 'nn_idx': int(mp_indices[idx]), 'mp_value': float(mp[idx])})
        lo = max(0, idx - excl_zone)
        hi = min(len(working), idx + excl_zone + 1)
        working[lo:hi] = np.nan
    return records

def _add_dates(records: list[dict[str, Any]], dates: pd.DatetimeIndex, idx_cols: list[str]) -> list[dict[str, Any]]:
    for rec in records:
        for col in idx_cols:
            idx = rec.get(col)
            date_col = col.replace('_idx', '_date')
            if idx is not None and 0 <= idx < len(dates):
                rec[date_col] = dates[idx].strftime('%Y-%m-%d')
            else:
                rec[date_col] = None
    return records

def process_region(region: str, cfg: DictConfig, out_dir: Path) -> dict[str, Any] | None:
    m: int = cfg.matrix_profile.subsequence_length
    n_motifs: int = cfg.matrix_profile.n_motifs
    n_discords: int = cfg.matrix_profile.n_discords
    kndvi_dir = _resolve(cfg, cfg.data.kndvi_dir)
    parquet_path = kndvi_dir / f'kndvi_{region}_2000_2025.parquet'
    log.info('──── Region: %s ────', region)
    df = _load_kndvi(parquet_path, region)
    if cfg.run_mvp_only:
        dates_all = sorted(df['date'].unique())
        cutoff = dates_all[:60][-1]
        df = df[df['date'] <= cutoff].copy()
        log.info('[MVP] Truncated to first 60 months (up to %s)', cutoff)
    dates = _date_labels(df)
    ts = _mean_ts(df)
    ts_anom = _deseasonalize(ts, dates)
    n_pixels = df['pixel_id'].nunique()
    log.info('  Time series length: %d months | Pixels: %d | m=%d | anomaly std=%.4f', len(ts), n_pixels, m, float(np.std(ts_anom)))
    log.info('  Computing matrix profile (deseasonalized) …')
    mp_array = _run_matrix_profile(ts_anom, m)
    n_mp = len(mp_array)
    mp_df = pd.DataFrame({'date_idx': np.arange(n_mp), 'date': dates[:n_mp], 'mp_value': mp_array[:, 0].astype(np.float64), 'mp_index': mp_array[:, 1].astype(np.int64)})
    mp_parquet = out_dir / f'matrix_profile_{region}.parquet'
    mp_df.to_parquet(mp_parquet, index=False)
    log.info('  Saved: %s', mp_parquet)
    log.info('  Extracting top-%d motifs …', n_motifs)
    motif_records = _extract_motifs(mp_array, ts_anom, m, n_motifs)
    motif_records = _add_dates(motif_records, dates, ['start_idx', 'nn_idx'])
    _MOTIF_COLS = ['motif_rank', 'start_idx', 'start_date', 'nn_idx', 'nn_date', 'distance']
    if motif_records:
        motif_df = pd.DataFrame(motif_records).reindex(columns=_MOTIF_COLS)
    else:
        motif_df = pd.DataFrame(columns=_MOTIF_COLS)
        log.warning('  No motifs extracted for %s', region)
    motif_df['region'] = region
    motif_csv = out_dir / f'motifs_{region}.csv'
    motif_df.to_csv(motif_csv, index=False)
    log.info('  Saved: %s', motif_csv)
    log.info('  Extracting top-%d discords …', n_discords)
    discord_records = _extract_discords(mp_array, m, n_discords)
    discord_records = _add_dates(discord_records, dates, ['start_idx', 'nn_idx'])
    _DISCORD_COLS = ['discord_rank', 'start_idx', 'start_date', 'nn_idx', 'nn_date', 'mp_value']
    if discord_records:
        discord_df = pd.DataFrame(discord_records).reindex(columns=_DISCORD_COLS)
    else:
        discord_df = pd.DataFrame(columns=_DISCORD_COLS)
    discord_df['region'] = region
    discord_csv = out_dir / f'discords_{region}.csv'
    discord_df.to_csv(discord_csv, index=False)
    log.info('  Saved: %s', discord_csv)
    mp_vals = mp_array[:, 0].astype(np.float64)
    summary = {'region': region, 'n_pixels': n_pixels, 'ts_length': len(ts), 'mp_length': n_mp, 'mp_mean': float(np.nanmean(mp_vals)), 'mp_std': float(np.nanstd(mp_vals)), 'mp_min': float(np.nanmin(mp_vals)), 'mp_max': float(np.nanmax(mp_vals)), 'n_motifs_extracted': len(motif_df), 'n_discords_extracted': len(discord_df), 'best_motif_distance': float(motif_df['distance'].min()) if not motif_df.empty else None, 'mvp_mode': bool(cfg.run_mvp_only)}
    return summary

@hydra.main(version_base='1.3', config_path='../conf', config_name='resilience_forest')
def main(cfg: DictConfig) -> None:
    log.info('stumpy version: %s', stumpy.__version__)
    log.info('numpy version:  %s', np.__version__)
    log.info('Config:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    out_dir = _resolve(cfg, cfg.output.dir) / 'E2'
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('Output directory: %s', out_dir)
    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions
    if cfg.run_mvp_only:
        all_regions = ['africa']
        log.info('[MVP] Processing only: africa (first 60 months)')
    summaries: list[dict[str, Any]] = []
    for region in all_regions:
        try:
            stats = process_region(region, cfg, out_dir)
            if stats is not None:
                summaries.append(stats)
        except FileNotFoundError as exc:
            log.warning('Skipping %s — %s', region, exc)
        except Exception:
            log.exception('Error processing region %s', region)
            raise
    if summaries:
        stats_df = pd.DataFrame(summaries)
        stats_csv = out_dir / 'stats_summary_E2.csv'
        stats_df.to_csv(stats_csv, index=False)
        log.info('Saved summary: %s', stats_csv)
        log.info('\n%s', stats_df[['region', 'n_pixels', 'ts_length', 'best_motif_distance']].to_string(index=False))
    else:
        log.warning('No regions processed successfully. Check E1 outputs exist.')
    log.info('E2 complete.')

def _write_config() -> None:
    conf_dir = BASE_DIR / 'conf'
    conf_dir.mkdir(parents=True, exist_ok=True)
    target = conf_dir / 'resilience_forest.yaml'
    if target.exists():
        print(f'Config already exists at {target}. Not overwriting.')
    else:
        print(f'Config already managed by Hydra at {target}.')
    print('Use Hydra overrides to customise: e.g. matrix_profile.subsequence_length=12')
if __name__ == '__main__':
    if '--write-config' in sys.argv:
        sys.argv.remove('--write-config')
        _write_config()
        sys.exit(0)
    main()
