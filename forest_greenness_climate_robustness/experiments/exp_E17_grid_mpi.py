from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Any
import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import stumpy
from omegaconf import DictConfig, OmegaConf
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger('E17')
BASE_DIR = Path(__file__).resolve().parent.parent
_READ_COLS = ['date', 'latitude', 'longitude', 'kndvi']

def _resolve(relpath: str) -> Path:
    return BASE_DIR / relpath

def _deseasonalize(ts: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(ts, index=dates, dtype=np.float64)
    climatology = s.groupby(s.index.month).transform('mean')
    return (s - climatology).to_numpy(dtype=np.float64)

def _compute_mpi(mp_values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(mp_values, dtype=np.float64).rolling(window=window, min_periods=window // 2).min().to_numpy(dtype=np.float64)

def _mann_kendall(values: np.ndarray) -> dict[str, Any]:
    import pymannkendall as mk
    arr = pd.Series(values).dropna().to_numpy()
    if len(arr) < 4 or np.nanstd(arr) == 0.0:
        return {'trend': 'insufficient_data', 'p': np.nan, 'Tau': np.nan, 'slope': np.nan}
    res = mk.original_test(arr)
    return {'trend': res.trend, 'p': float(res.p), 'Tau': float(res.Tau), 'slope': float(res.slope)}

def _trend_tests(values: np.ndarray) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exp_E13_hamed_rao_mk import hamed_rao_mk
    res = _mann_kendall(values)
    arr = pd.Series(values).dropna().to_numpy()
    if len(arr) < 4 or np.nanstd(arr) == 0.0:
        res['p_hr'] = np.nan
        return res
    _, p_hr, _ = hamed_rao_mk(arr)
    res['p_hr'] = float(p_hr)
    return res

def _bh(p: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exp_E10_fdr_correction import _bh_fdr
    p = np.asarray(p, dtype=float)
    adj = np.full(p.shape, np.nan)
    rej = np.zeros(p.shape, dtype=bool)
    ok = np.isfinite(p)
    if ok.any():
        a, r = _bh_fdr(p[ok], q)
        adj[ok] = a
        rej[ok] = r
    return (adj, rej)

def _lattice_edges(rows: np.ndarray, cols: np.ndarray, contiguity: str) -> tuple[np.ndarray, np.ndarray]:
    if contiguity == 'queen':
        offs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
    elif contiguity == 'rook':
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError(f'grid.moran_contiguity must be queen or rook, got {contiguity!r}')
    index = {(int(r), int(c)): k for k, (r, c) in enumerate(zip(rows, cols))}
    ii: list[int] = []
    jj: list[int] = []
    for k, (r, c) in enumerate(zip(rows, cols)):
        for di, dj in offs:
            nb = index.get((int(r) + di, int(c) + dj))
            if nb is not None:
                ii.append(k)
                jj.append(nb)
    return (np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64))

def morans_i(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, contiguity: str, n_perm: int, seed: int) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    v, r, c = (v[ok], np.asarray(rows)[ok], np.asarray(cols)[ok])
    n_nonfinite = int((~ok).sum())
    ii, jj = _lattice_edges(r, c, contiguity)
    has_nb = np.zeros(len(v), dtype=bool)
    has_nb[ii] = True
    n_islands = int((~has_nb).sum())
    if n_islands:
        v, r, c = (v[has_nb], r[has_nb], c[has_nb])
        ii, jj = _lattice_edges(r, c, contiguity)
    n = len(v)
    out: dict[str, Any] = {'n_used': n, 'n_islands_dropped': n_islands, 'n_nonfinite_dropped': n_nonfinite, 'n_links': len(ii) // 2}
    if n < 3 or len(ii) == 0 or np.std(v) == 0.0:
        out.update({'I': np.nan, 'expected_I': np.nan, 'perm_mean': np.nan, 'perm_sd': np.nan, 'z_perm': np.nan, 'p_greater': np.nan, 'p_two_sided': np.nan})
        return out
    s0 = float(len(ii))
    z = v - v.mean()
    denom = float(np.sum(z * z))
    obs = n / s0 * float(np.sum(z[ii] * z[jj])) / denom
    rng = np.random.default_rng(seed)
    perm_i = np.empty(n_perm)
    for b in range(n_perm):
        zp = z[rng.permutation(n)]
        perm_i[b] = n / s0 * float(np.sum(zp[ii] * zp[jj])) / denom
    e_i = -1.0 / (n - 1)
    out.update({'I': obs, 'expected_I': e_i, 'perm_mean': float(perm_i.mean()), 'perm_sd': float(perm_i.std(ddof=1)), 'z_perm': float((obs - perm_i.mean()) / perm_i.std(ddof=1)), 'p_greater': float((1 + np.sum(perm_i >= obs)) / (n_perm + 1)), 'p_two_sided': float((1 + np.sum(np.abs(perm_i - e_i) >= abs(obs - e_i))) / (n_perm + 1))})
    return out

def aggregate_to_grid(parquet_path: Path, cell_size: float, batch_rows: int) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f'kNDVI parquet not found: {parquet_path}. Run E1 first.')
    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    log.info('  Streaming %s (%.1f M pixel-months, %d row groups) …', parquet_path.name, total_rows / 1000000.0, pf.metadata.num_row_groups)
    partials: list[pd.DataFrame] = []
    seen = 0
    n_nan = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=_READ_COLS):
        df = batch.to_pandas()
        seen += len(df)
        finite = np.isfinite(df['kndvi'].to_numpy())
        n_nan += int((~finite).sum())
        df = df.loc[finite]
        if df.empty:
            continue
        df['row_i'] = np.floor(df['latitude'].to_numpy() / cell_size).astype(np.int32)
        df['col_j'] = np.floor(df['longitude'].to_numpy() / cell_size).astype(np.int32)
        df['date'] = df['date'].values.astype('datetime64[M]')
        g = df.groupby(['row_i', 'col_j', 'date'], sort=False)['kndvi'].agg(kndvi_sum='sum', n_pix='count').reset_index()
        partials.append(g)
        del df, g
        if seen % (batch_rows * 5) == 0:
            log.info('    … %.1f M / %.1f M rows', seen / 1000000.0, total_rows / 1000000.0)
    if not partials:
        raise ValueError(f'No finite kNDVI values in {parquet_path}')
    agg = pd.concat(partials, ignore_index=True)
    del partials
    agg = agg.groupby(['row_i', 'col_j', 'date'], sort=False).agg(kndvi_sum=('kndvi_sum', 'sum'), n_pix=('n_pix', 'sum')).reset_index()
    log.info('  Aggregated to %d (cell, month) records; %d non-finite pixel-months dropped.', len(agg), n_nan)
    return agg

def process_region(region: str, cfg: DictConfig, out_dir: Path) -> dict[str, Any] | None:
    cell_size = float(cfg.grid.cell_size_deg)
    min_pix = int(cfg.grid.min_pixels_per_cell)
    overrides = OmegaConf.to_container(cfg.grid.get('min_completeness_overrides', {}) or {})
    min_comp = float(overrides.get(region, cfg.grid.min_completeness))
    if region in overrides:
        log.info('  Using relaxed completeness threshold %.2f for %s (high-latitude winter data gap).', min_comp, region)
    mpi_window = int(cfg.grid.mpi_window)
    m = int(cfg.matrix_profile.subsequence_length)
    alpha = float(cfg.analysis.alpha)
    kndvi_dir = _resolve(cfg.data.kndvi_dir)
    parquet_path = kndvi_dir / f'kndvi_{region}_2000_2025.parquet'
    log.info('──── Region: %s (cell=%.2f deg) ────', region, cell_size)
    agg = aggregate_to_grid(parquet_path, cell_size, int(cfg.grid.batch_rows))
    all_dates = pd.DatetimeIndex(sorted(agg['date'].unique()))
    n_months_region = len(all_dates)
    log.info('  Regional month grid: %d months (%s … %s)', n_months_region, all_dates[0].date(), all_dates[-1].date())
    agg['cell_mean'] = agg['kndvi_sum'] / agg['n_pix']
    agg['cell_id'] = agg['row_i'].astype(str) + '_' + agg['col_j'].astype(str)
    per_cell = agg.groupby(['row_i', 'col_j'], sort=False).agg(n_months=('date', 'size'), median_pix=('n_pix', 'median'), mean_pix=('n_pix', 'mean'), max_pix=('n_pix', 'max')).reset_index()
    per_cell['completeness'] = per_cell['n_months'] / n_months_region
    per_cell['lat_c'] = (per_cell['row_i'] + 0.5) * cell_size
    per_cell['lon_c'] = (per_cell['col_j'] + 0.5) * cell_size
    ok_pix = per_cell['median_pix'] >= min_pix
    ok_comp = per_cell['completeness'] >= min_comp
    ok_len = per_cell['n_months'] >= 2 * m
    reason = np.where(~ok_pix, 'insufficient_pixels', np.where(~ok_comp, 'insufficient_completeness', np.where(~ok_len, 'series_too_short', 'kept')))
    per_cell['status'] = reason
    n_candidate = len(per_cell)
    keep = per_cell[per_cell['status'] == 'kept'].copy()
    log.info('  Candidate cells: %d | kept %d | dropped: pixels=%d, completeness=%d, too_short=%d', n_candidate, len(keep), int((per_cell['status'] == 'insufficient_pixels').sum()), int((per_cell['status'] == 'insufficient_completeness').sum()), int((per_cell['status'] == 'series_too_short').sum()))
    wide = agg.pivot_table(index='date', columns=['row_i', 'col_j'], values='cell_mean').reindex(all_dates)
    cell_rows: list[dict[str, Any]] = []
    series_rows: list[pd.DataFrame] = []
    n_degenerate = 0
    for _, cell in keep.iterrows():
        key = (int(cell['row_i']), int(cell['col_j']))
        s = wide[key]
        frac_interp = float(s.isna().mean())
        if cfg.grid.interpolate_gaps:
            s = s.interpolate(method='linear', limit_direction='both')
        if s.isna().any():
            n_degenerate += 1
            per_cell.loc[(per_cell['row_i'] == key[0]) & (per_cell['col_j'] == key[1]), 'status'] = 'unfillable_gaps'
            continue
        ts = s.to_numpy(dtype=np.float64)
        ts_anom = _deseasonalize(ts, all_dates)
        if not np.isfinite(ts_anom).all() or np.std(ts_anom) == 0.0:
            n_degenerate += 1
            per_cell.loc[(per_cell['row_i'] == key[0]) & (per_cell['col_j'] == key[1]), 'status'] = 'degenerate_series'
            continue
        mp_array = stumpy.stump(ts_anom, m)
        mp_vals = mp_array[:, 0].astype(np.float64)
        mp_vals[~np.isfinite(mp_vals)] = np.nan
        mpi = _compute_mpi(mp_vals, mpi_window)
        mp_dates = all_dates[:len(mp_vals)]
        mk = _trend_tests(mpi)
        cell_rows.append({'region': region, 'row_i': key[0], 'col_j': key[1], 'lat_c': float(cell['lat_c']), 'lon_c': float(cell['lon_c']), 'n_months': int(cell['n_months']), 'median_pix': float(cell['median_pix']), 'completeness': float(cell['completeness']), 'frac_interpolated': frac_interp, 'kndvi_mean': float(np.nanmean(ts)), 'anom_std': float(np.nanstd(ts_anom)), 'mp_mean': float(np.nanmean(mp_vals)), 'mp_std': float(np.nanstd(mp_vals)), 'mpi_mean': float(np.nanmean(mpi)), 'mpi_median': float(np.nanmedian(mpi)), 'mpi_mk_trend': mk['trend'], 'mpi_mk_p': mk['p'], 'mpi_mk_tau': mk['Tau'], 'mpi_mk_slope': mk['slope'], 'mpi_significant': bool(mk['p'] < alpha) if np.isfinite(mk['p']) else False, 'mpi_mk_p_hr': mk['p_hr']})
        series_rows.append(pd.DataFrame({'region': region, 'row_i': key[0], 'col_j': key[1], 'date': mp_dates, 'mp_value': mp_vals, 'mpi_rolling_min': mpi}))
    if n_degenerate:
        log.info('  Dropped %d further cells (unfillable gaps / degenerate series).', n_degenerate)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / f'grid_exclusions_{region}.csv', index=False)
    if cell_rows:
        cells_df = pd.DataFrame(cell_rows)
        adj, rej = _bh(cells_df['mpi_mk_p_hr'].to_numpy(dtype=float), float(cfg.grid.fdr_q))
        cells_df['mpi_mk_p_hr_bh'] = adj
        cells_df['mpi_significant_hr_bh'] = rej
        series_df = pd.concat(series_rows, ignore_index=True)
        cells_df.to_parquet(out_dir / f'grid_mpi_cells_{region}.parquet', index=False)
        series_df.to_parquet(out_dir / f'grid_mpi_series_{region}.parquet', index=False)
        log.info('  Saved %d cells -> %s', len(cells_df), out_dir / f'grid_mpi_cells_{region}.parquet')
    else:
        log.warning('  No usable cells for %s — exclusion ledger written anyway.', region)
    summary = {'region': region, 'cell_size_deg': cell_size, 'n_candidate_cells': n_candidate, 'n_kept_cells': len(cell_rows), 'n_drop_pixels': int((per_cell['status'] == 'insufficient_pixels').sum()), 'n_drop_completeness': int((per_cell['status'] == 'insufficient_completeness').sum()), 'n_drop_too_short': int((per_cell['status'] == 'series_too_short').sum()), 'n_drop_gaps': int((per_cell['status'] == 'unfillable_gaps').sum()), 'n_drop_degenerate': int((per_cell['status'] == 'degenerate_series').sum()), 'n_months_region': n_months_region, 'median_pix_per_cell': float(cells_df['median_pix'].median()) if cell_rows else np.nan, 'median_completeness_pixok': float(per_cell.loc[per_cell['median_pix'] >= min_pix, 'completeness'].median()), 'p95_completeness_pixok': float(per_cell.loc[per_cell['median_pix'] >= min_pix, 'completeness'].quantile(0.95)), 'min_pixels_per_cell': min_pix, 'min_completeness': min_comp, 'completeness_relaxed': region in overrides, 'median_frac_interpolated': float(pd.DataFrame(cell_rows)['frac_interpolated'].median()) if cell_rows else np.nan}
    return summary

def regional_mpi_from_e2(e2_dir: Path, region: str, window: int, alpha: float) -> dict[str, Any]:
    path = e2_dir / f'matrix_profile_{region}.parquet'
    if not path.exists():
        return {'regional_mpi_mean': np.nan, 'regional_mk_trend': 'unavailable', 'regional_mk_p_hr': np.nan, 'regional_mk_slope': np.nan, 'regional_mk_p': np.nan, 'regional_mk_tau': np.nan, 'regional_mk_significant': False}
    mp = pd.read_parquet(path)['mp_value'].to_numpy(dtype=np.float64)
    mpi = _compute_mpi(mp, window)
    mk = _trend_tests(mpi)
    return {'regional_mk_p_hr': mk['p_hr'], 'regional_mpi_mean': float(np.nanmean(mpi)), 'regional_mp_mean': float(np.nanmean(mp)), 'regional_mk_trend': mk['trend'], 'regional_mk_slope': mk['slope'], 'regional_mk_p': mk['p'], 'regional_mk_tau': mk['Tau'], 'regional_mk_significant': bool(mk['p'] < alpha) if np.isfinite(mk['p']) else False}

def compare_to_regional(region: str, cells_df: pd.DataFrame, e2_dir: Path, e8_dir: Path, window: int, alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {'region': region, 'n_cells': len(cells_df)}
    reg = regional_mpi_from_e2(e2_dir, region, window, alpha)
    reg_mpi_mean = reg['regional_mpi_mean']
    reg_slope = reg['regional_mk_slope']
    reg_trend = reg['regional_mk_trend']
    reg_sig = reg['regional_mk_significant']
    mpi_path = e8_dir / f'mpi_{region}.csv'
    out['e8_stored_mpi_mean'] = float(np.nanmean(pd.read_csv(mpi_path)['mpi_rolling_min'])) if mpi_path.exists() else np.nan
    cell_mpi = cells_df['mpi_mean'].to_numpy(dtype=np.float64)
    cell_slope = cells_df['mpi_mk_slope'].to_numpy(dtype=np.float64)
    out['regional_mpi_mean'] = reg_mpi_mean
    out['cell_mpi_mean'] = float(np.nanmean(cell_mpi))
    out['cell_mpi_median'] = float(np.nanmedian(cell_mpi))
    out['cell_mpi_std'] = float(np.nanstd(cell_mpi, ddof=1))
    out['cell_mpi_cv'] = float(np.nanstd(cell_mpi, ddof=1) / np.nanmean(cell_mpi))
    out['cell_mpi_min'] = float(np.nanmin(cell_mpi))
    out['cell_mpi_max'] = float(np.nanmax(cell_mpi))
    out['cell_mpi_p05'] = float(np.nanpercentile(cell_mpi, 5))
    out['cell_mpi_p95'] = float(np.nanpercentile(cell_mpi, 95))
    out['cell_mpi_iqr'] = float(np.nanpercentile(cell_mpi, 75) - np.nanpercentile(cell_mpi, 25))
    out['range_over_regional'] = float((np.nanmax(cell_mpi) - np.nanmin(cell_mpi)) / reg_mpi_mean) if np.isfinite(reg_mpi_mean) else np.nan
    if np.isfinite(reg_mpi_mean):
        out['regional_percentile_in_cells'] = float(100.0 * np.nanmean(cell_mpi < reg_mpi_mean))
        out['frac_cells_within_10pct_of_regional'] = float(np.nanmean(np.abs(cell_mpi - reg_mpi_mean) / reg_mpi_mean <= 0.1))
        out['frac_cells_within_25pct_of_regional'] = float(np.nanmean(np.abs(cell_mpi - reg_mpi_mean) / reg_mpi_mean <= 0.25))
        out['median_abs_rel_error'] = float(np.nanmedian(np.abs(cell_mpi - reg_mpi_mean) / reg_mpi_mean))
    else:
        for k in ('regional_percentile_in_cells', 'frac_cells_within_10pct_of_regional', 'frac_cells_within_25pct_of_regional', 'median_abs_rel_error'):
            out[k] = np.nan
    out['regional_mk_trend'] = reg_trend
    out['regional_mk_slope'] = reg_slope
    out['regional_mk_p'] = reg.get('regional_mk_p', np.nan)
    out['regional_mk_tau'] = reg.get('regional_mk_tau', np.nan)
    out['regional_mk_significant'] = reg_sig
    out['regional_mk_p_hr'] = reg.get('regional_mk_p_hr', np.nan)
    reg_tau = reg.get('regional_mk_tau', np.nan)
    cell_tau = cells_df['mpi_mk_tau'].to_numpy(dtype=np.float64)
    if np.isfinite(reg_tau) and reg_tau != 0:
        ts_ = np.sign(cell_tau[np.isfinite(cell_tau)])
        out['frac_cells_same_tau_sign'] = float(np.mean(ts_ == np.sign(reg_tau)))
        out['frac_cells_opposite_tau_sign'] = float(np.mean(ts_ == -np.sign(reg_tau)))
    else:
        out['frac_cells_same_tau_sign'] = np.nan
        out['frac_cells_opposite_tau_sign'] = np.nan
    if np.isfinite(reg_slope) and reg_slope != 0.0:
        reg_sign = np.sign(reg_slope)
        cell_sign = np.sign(cell_slope)
        valid = np.isfinite(cell_slope)
        cs = cell_sign[valid]
        out['frac_cells_same_slope_sign'] = float(np.mean(cs == reg_sign))
        out['frac_cells_zero_slope'] = float(np.mean(cs == 0))
        out['frac_cells_opposite_slope_sign'] = float(np.mean(cs == -reg_sign))
        nz = cs != 0
        out['frac_nonzero_cells_opposite_sign'] = float(np.mean(cs[nz] == -reg_sign) if nz.any() else np.nan)
    else:
        for k in ('frac_cells_same_slope_sign', 'frac_cells_zero_slope', 'frac_cells_opposite_slope_sign', 'frac_nonzero_cells_opposite_sign'):
            out[k] = np.nan
    trends = cells_df['mpi_mk_trend'].astype(str)
    out['frac_cells_increasing'] = float(trends.str.contains('increasing').mean())
    out['frac_cells_decreasing'] = float(trends.str.contains('decreasing').mean())
    out['frac_cells_no_trend'] = float((trends == 'no trend').mean())
    sig = cells_df['mpi_significant'].to_numpy(dtype=bool)
    out['frac_cells_significant'] = float(np.mean(sig))
    if reg_trend in ('increasing', 'decreasing'):
        agree = trends.str.contains(reg_trend) & sig
        opp = trends.str.contains('increasing' if reg_trend == 'decreasing' else 'decreasing') & sig
        out['frac_sig_cells_agreeing_with_regional'] = float(agree.mean())
        out['frac_sig_cells_opposing_regional'] = float(opp.mean())
    else:
        out['frac_sig_cells_agreeing_with_regional'] = np.nan
        out['frac_sig_cells_opposing_regional'] = np.nan
    return out

def regional_family_corrected(e2_dir: Path, regions: list[str], window: int, q: float) -> pd.DataFrame:
    rows = []
    for region in regions:
        reg = regional_mpi_from_e2(e2_dir, region, window, alpha=0.05)
        rows.append({'region': region, 'regional_tau': reg['regional_mk_tau'], 'regional_p_raw': reg['regional_mk_p'], 'regional_p_hr': reg['regional_mk_p_hr'], 'regional_mpi_mean': reg['regional_mpi_mean']})
    df = pd.DataFrame(rows)
    adj, rej = _bh(df['regional_p_hr'].to_numpy(dtype=float), q)
    df['regional_p_hr_bh'] = adj
    df['regional_significant_hr_bh'] = rej
    df['bh_family_size'] = int(np.isfinite(df['regional_p_hr']).sum())
    return df

def corrected_region_summary(region: str, cells_df: pd.DataFrame, ledger: pd.DataFrame, reg_row: pd.Series, cfg: DictConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    min_pix = int(cfg.grid.min_pixels_per_cell)
    m = int(cfg.matrix_profile.subsequence_length)
    uni = float(cfg.grid.uniform_completeness_check)
    overrides = OmegaConf.to_container(cfg.grid.get('min_completeness_overrides', {}) or {})
    used = float(overrides.get(region, cfg.grid.min_completeness))
    n = len(cells_df)
    tau = cells_df['mpi_mk_tau'].to_numpy(dtype=float)
    sig = cells_df['mpi_significant_hr_bh'].to_numpy(dtype=bool)
    sig_raw = cells_df['mpi_significant'].to_numpy(dtype=bool)
    level = cells_df['mpi_mean'].to_numpy(dtype=float)
    reg_tau = float(reg_row['regional_tau'])
    reg_lvl = float(reg_row['regional_mpi_mean'])
    sgn = np.sign(reg_tau)
    opp = np.sign(tau) == -sgn
    same = np.sign(tau) == sgn
    if uni < used:
        raise ValueError(f'uniform_completeness_check {uni} < threshold used {used} for {region}')
    pass_uni = (ledger['median_pix'] >= min_pix) & (ledger['completeness'] >= uni) & (ledger['n_months'] >= 2 * m)
    n_pass_uni_screen = int(pass_uni.sum())
    n_retained_uni = int((pass_uni & (ledger['status'] == 'kept')).sum())
    n_retained_uni_cells = int((cells_df['completeness'] >= uni).sum())
    if n_retained_uni != n_retained_uni_cells:
        raise AssertionError(f'{region}: uniform-{uni} count mismatch ledger={n_retained_uni} cells={n_retained_uni_cells}')
    summ = {'region': region, 'n_candidate_cells': len(ledger), 'n_cells': n, 'completeness_threshold_used': used, 'n_retained_used_threshold': n, 'uniform_threshold_checked': uni, 'n_pass_screen_uniform': n_pass_uni_screen, 'n_retained_uniform': n_retained_uni, 'median_frac_interpolated': float(np.median(cells_df['frac_interpolated'])), 'max_frac_interpolated': float(np.max(cells_df['frac_interpolated'])), 'regional_tau': reg_tau, 'regional_p_raw': float(reg_row['regional_p_raw']), 'regional_p_hr': float(reg_row['regional_p_hr']), 'regional_p_hr_bh': float(reg_row['regional_p_hr_bh']), 'regional_significant_hr_bh': bool(reg_row['regional_significant_hr_bh']), 'n_cells_opposite_tau': int(opp.sum()), 'frac_cells_opposite_tau': float(opp.mean()), 'n_cells_zero_tau': int((tau == 0).sum()), 'n_cells_sig_raw_uncorrected': int(sig_raw.sum()), 'frac_cells_sig_raw_uncorrected': float(sig_raw.mean()), 'n_cells_sig_hr_bh': int(sig.sum()), 'frac_cells_sig_hr_bh': float(sig.mean()), 'n_sig_cells_opposite_tau': int((sig & opp).sum()), 'n_sig_cells_same_tau': int((sig & same).sum()), 'frac_sig_cells_opposite_tau': float((sig & opp).sum() / sig.sum()) if sig.any() else np.nan, 'frac_all_cells_sig_and_opposite': float((sig & opp).mean()), 'level_cell_mean': float(np.mean(level)), 'level_cv': float(np.std(level, ddof=1) / np.mean(level)), 'regional_mpi_mean': reg_lvl, 'frac_cells_within_10pct_of_regional': float(np.mean(np.abs(level - reg_lvl) / reg_lvl <= 0.1)), 'regional_percentile_in_cells': float(100.0 * np.mean(level < reg_lvl))}
    rows_i = cells_df['row_i'].to_numpy()
    cols_j = cells_df['col_j'].to_numpy()
    n_perm = int(cfg.grid.moran_n_perm)
    seed = int(cfg.analysis.seed)
    moran_rows = []
    for contig in [str(x) for x in cfg.grid.moran_contiguity]:
        for field, vals in (('trend_tau', tau), ('level_mpi_mean', level)):
            res = morans_i(vals, rows_i, cols_j, contig, n_perm, seed)
            moran_rows.append({'region': region, 'field': field, 'contiguity': contig, 'weights': 'binary', 'n_perm': n_perm, 'seed': seed, **res})
    return (summ, moran_rows)

@hydra.main(version_base='1.3', config_path='../conf', config_name='resilience_forest')
def main(cfg: DictConfig) -> None:
    log.info('stumpy %s | numpy %s | pandas %s', stumpy.__version__, np.__version__, pd.__version__)
    log.info('Grid config:\n%s', OmegaConf.to_yaml(cfg.grid, resolve=True))
    out_dir = _resolve(cfg.output.dir) / 'E17'
    subdir = str(cfg.grid.get('out_subdir', '') or '')
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    e2_dir = _resolve(cfg.output.dir) / 'E2'
    e8_dir = _resolve(cfg.output.dir) / 'E8'
    alpha = float(cfg.analysis.alpha)
    mpi_window = int(cfg.grid.mpi_window)
    compare_only = bool(cfg.grid.get('compare_only', False))
    if compare_only:
        log.info('compare_only=true — reusing saved per-cell results in %s', out_dir)
    regions: list[str] = list(cfg.grid.regions)
    if cfg.run_mvp_only:
        regions = ['congo']
        log.info('[MVP] Processing only: congo')
    region_summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    all_cells: list[pd.DataFrame] = []
    prev_counts = pd.read_csv(out_dir / 'grid_cell_counts_E17.csv') if compare_only and (out_dir / 'grid_cell_counts_E17.csv').exists() else None
    for region in regions:
        if compare_only:
            if prev_counts is None or not (prev_counts['region'] == region).any():
                log.warning('Skipping %s — no stored cell counts.', region)
                continue
            summary = prev_counts[prev_counts['region'] == region].iloc[0].to_dict()
        else:
            try:
                summary = process_region(region, cfg, out_dir)
            except FileNotFoundError as exc:
                log.warning('Skipping %s — %s', region, exc)
                continue
            if summary is None:
                continue
        region_summaries.append(summary)
        cells_path = out_dir / f'grid_mpi_cells_{region}.parquet'
        if not cells_path.exists():
            log.warning('  %s: 0 cells passed screening (median completeness of pixel-sufficient cells = %.2f vs required %.2f) — no comparison.', region, summary['median_completeness_pixok'], summary['min_completeness'])
            continue
        cells_df = pd.read_parquet(cells_path)
        all_cells.append(cells_df)
        comp = compare_to_regional(region, cells_df, e2_dir, e8_dir, mpi_window, alpha)
        comp.update({k: summary[k] for k in ('n_candidate_cells', 'n_kept_cells', 'min_completeness', 'completeness_relaxed', 'median_frac_interpolated')})
        comparisons.append(comp)
        log.info('  Grid vs regional: regional MPI=%.4f | cell mean=%.4f | CV=%.3f | opposite tau sign=%.1f%% | sig. cells opposing regional=%.1f%%', comp['regional_mpi_mean'], comp['cell_mpi_mean'], comp['cell_mpi_cv'], 100 * (comp['frac_cells_opposite_tau_sign'] or float('nan')), 100 * comp['frac_sig_cells_opposing_regional'])
    if region_summaries:
        pd.DataFrame(region_summaries).to_csv(out_dir / 'grid_cell_counts_E17.csv', index=False)
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        comp_df.to_csv(out_dir / 'grid_vs_regional_E17.csv', index=False)
        log.info('Saved: %s', out_dir / 'grid_vs_regional_E17.csv')
    if all_cells:
        pd.concat(all_cells, ignore_index=True).to_parquet(out_dir / 'grid_mpi_cells_all.parquet', index=False)
    q = float(cfg.grid.fdr_q)
    family = list(cfg.regions.continents) + list(cfg.regions.subregions)
    reg_df = regional_family_corrected(e2_dir, family, mpi_window, q)
    reg_df.to_csv(out_dir / 'regional_trends_corrected_E17.csv', index=False)
    log.info('Regional MPI trends (Hamed-Rao + BH over %d regions):\n%s', len(reg_df), reg_df.round(4).to_string(index=False))
    e18_path = _resolve(cfg.output.dir) / 'E18' / 'mpi_trend_corrected_E18.csv'
    if e18_path.exists():
        e18 = pd.read_csv(e18_path).set_index('region')
        chk = reg_df.set_index('region').join(e18[['p_hamedrao', 'p_bh_hamedrao']], how='inner')
        log.info('Cross-check vs E18: max |dp_HR|=%.2e, max |dp_BH|=%.2e over %d regions', (chk['regional_p_hr'] - chk['p_hamedrao']).abs().max(), (chk['regional_p_hr_bh'] - chk['p_bh_hamedrao']).abs().max(), len(chk))
    corr_rows: list[dict[str, Any]] = []
    moran_rows: list[dict[str, Any]] = []
    trend_tabs: list[pd.DataFrame] = []
    for cells_df in all_cells:
        region = str(cells_df['region'].iloc[0])
        if 'mpi_mk_p_hr' not in cells_df.columns:
            raise RuntimeError(f'{region}: cells parquet predates the Hamed-Rao revision; re-run without grid.compare_only=true.')
        ledger = pd.read_csv(out_dir / f'grid_exclusions_{region}.csv')
        reg_row = reg_df.set_index('region').loc[region]
        summ, mrows = corrected_region_summary(region, cells_df, ledger, reg_row, cfg)
        corr_rows.append(summ)
        moran_rows.extend(mrows)
        trend_tabs.append(cells_df[['region', 'row_i', 'col_j', 'lat_c', 'lon_c', 'completeness', 'frac_interpolated', 'mpi_mean', 'mpi_mk_tau', 'mpi_mk_p', 'mpi_mk_p_hr', 'mpi_mk_p_hr_bh', 'mpi_significant', 'mpi_significant_hr_bh']].rename(columns={'mpi_mk_p': 'mpi_mk_p_raw', 'mpi_significant': 'mpi_significant_raw_uncorrected'}))
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(out_dir / 'summary_corrected_E17.csv', index=False)
        pd.DataFrame(moran_rows).to_csv(out_dir / 'moran_E17.csv', index=False)
        pd.concat(trend_tabs, ignore_index=True).to_csv(out_dir / 'cell_trends_corrected_E17.csv', index=False)
        log.info('Corrected summary:\n%s', pd.DataFrame(corr_rows).round(4).T.to_string())
        log.info("Moran's I:\n%s", pd.DataFrame(moran_rows).round(4).to_string(index=False))
    log.info('E17 complete.')

def _write_config() -> None:
    target = BASE_DIR / 'conf' / 'resilience_forest.yaml'
    print(f'Config is managed by Hydra at {target}.')
    print('Override with e.g.: grid.cell_size_deg=1.0 grid.min_pixels_per_cell=50')
if __name__ == '__main__':
    if '--write-config' in sys.argv:
        sys.argv.remove('--write-config')
        _write_config()
        sys.exit(0)
    main()
