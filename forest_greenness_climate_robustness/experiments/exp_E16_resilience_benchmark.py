from __future__ import annotations
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, t as student_t, binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_E10_fdr_correction import _bh_fdr
log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
REGION_ORDER = ['africa', 'asia', 'europe', 'north_america', 'south_america', 'australia_oceania', 'amazon', 'congo', 'boreal_siberia', 'southeast_asia', 'canadian_boreal', 'scandinavian_boreal', 'papua_new_guinea', 'russian_far_east']
INDICATORS = ['tac', 'lambda_ar1', 'variance']
ALL_METRICS = ['mpi'] + INDICATORS

@dataclass
class E16Config:
    window: int = 24
    mpi_min_periods: int = 12
    detrend_sigma: float = 60.0
    lag_max: int = 24
    primary_variant: str = 'deseason'
    min_obs: int = 30
    mpi_source: str = 'e2_recompute'

def _load_kndvi(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f'kNDVI parquet not found: {parquet_path}. Run E1 first.')
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['date', 'pixel_id']).reset_index(drop=True)

def _mean_ts(df: pd.DataFrame) -> np.ndarray:
    return df.groupby('date')['kndvi'].mean().sort_index().to_numpy(dtype=np.float64)

def _date_labels(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df['date'].unique()))

def _deseasonalize(ts: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(ts, index=dates, dtype=np.float64)
    climatology = s.groupby(s.index.month).transform('mean')
    return (s - climatology).to_numpy(dtype=np.float64)

def _gaussian_detrend(x: np.ndarray, sigma: float) -> np.ndarray:
    trend = gaussian_filter1d(x, sigma=float(sigma), mode='nearest')
    return x - trend

def _window_tac(w: np.ndarray) -> float:
    w = w[~np.isnan(w)]
    if len(w) < 4:
        return np.nan
    wc = w - w.mean()
    denom = np.dot(wc, wc)
    if denom <= 0:
        return np.nan
    return float(np.dot(wc[:-1], wc[1:]) / denom)

def _window_phi(w: np.ndarray) -> float:
    w = w[~np.isnan(w)]
    if len(w) < 4:
        return np.nan
    wc = w - w.mean()
    denom = np.dot(wc[:-1], wc[:-1])
    if denom <= 0:
        return np.nan
    return float(np.dot(wc[:-1], wc[1:]) / denom)

def _rolling_csd(x: np.ndarray, dates: pd.DatetimeIndex, window: int) -> pd.DataFrame:
    s = pd.Series(x, index=dates, dtype=np.float64)
    roll = s.rolling(window=window, center=True, min_periods=window)
    tac = roll.apply(_window_tac, raw=True)
    phi = roll.apply(_window_phi, raw=True)
    var = roll.var(ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.where(phi.to_numpy() > 0, np.log(phi.to_numpy()), np.nan)
    return pd.DataFrame({'date': dates, 'tac': tac.to_numpy(), 'phi_ar1': phi.to_numpy(), 'lambda_ar1': lam, 'variance': var.to_numpy()})

def _mk_tests(series: np.ndarray) -> dict:
    import pymannkendall as mk
    arr = series[~np.isnan(series)]
    if len(arr) < 10 or np.allclose(arr, arr[0]):
        return {'n_obs': int(len(arr)), 'tau': np.nan, 'p_hr': np.nan, 'p_uncorrected': np.nan, 'slope': np.nan, 'trend': 'insufficient_data'}
    hr = mk.hamed_rao_modification_test(arr)
    orig = mk.original_test(arr)
    return {'n_obs': int(len(arr)), 'tau': float(hr.Tau), 'p_hr': float(hr.p), 'p_uncorrected': float(orig.p), 'slope': float(hr.slope), 'trend': str(hr.trend)}

def _lag1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return 0.0
    xc = x - x.mean()
    d = np.dot(xc, xc)
    if d <= 0:
        return 0.0
    return float(np.clip(np.dot(xc[:-1], xc[1:]) / d, -0.999, 0.999))

def _spearman_eff(x: np.ndarray, y: np.ndarray) -> dict:
    mask = ~np.isnan(x) & ~np.isnan(y)
    xs, ys = (x[mask], y[mask])
    n = len(xs)
    if n < 10:
        return {'n': n, 'rho': np.nan, 'p_naive': np.nan, 'n_eff': np.nan, 'p_eff': np.nan}
    rho, p_naive = spearmanr(xs, ys)
    r1x, r1y = (_lag1(xs), _lag1(ys))
    n_eff = n * (1.0 - r1x * r1y) / (1.0 + r1x * r1y)
    n_eff = float(np.clip(n_eff, 3.0, float(n)))
    if abs(rho) >= 1.0 or n_eff <= 2:
        p_eff = 0.0 if abs(rho) >= 1.0 else np.nan
    else:
        tstat = rho * np.sqrt((n_eff - 2.0) / (1.0 - rho ** 2))
        p_eff = float(2.0 * student_t.sf(abs(tstat), df=n_eff - 2.0))
    return {'n': int(n), 'rho': float(rho), 'p_naive': float(p_naive), 'n_eff': round(n_eff, 2), 'p_eff': float(p_eff)}

def _lag_scan(mpi: np.ndarray, ind: np.ndarray, lag_max: int) -> tuple[int, float]:
    best_lag, best_rho = (0, np.nan)
    n = len(mpi)
    for k in range(-lag_max, lag_max + 1):
        if k >= 0:
            a, b = (mpi[:n - k], ind[k:])
        else:
            a, b = (mpi[-k:], ind[:n + k])
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 24:
            continue
        rho, _ = spearmanr(a[mask], b[mask])
        if np.isnan(rho):
            continue
        if np.isnan(best_rho) or abs(rho) > abs(best_rho):
            best_lag, best_rho = (k, float(rho))
    return (best_lag, best_rho)

def _load_mpi(e2_dir: Path, e8_dir: Path, region: str, window: int, min_periods: int, source: str) -> pd.DataFrame:
    if source == 'e8_stored':
        path = e8_dir / f'mpi_{region}.csv'
        if not path.exists():
            raise FileNotFoundError(f'E8 MPI file not found: {path}. Run exp_E8 first.')
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'mpi_rolling_min']].rename(columns={'mpi_rolling_min': 'mpi'})
    path = e2_dir / f'matrix_profile_{region}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'E2 matrix profile not found: {path}. Run exp_E2 first.')
    mp = pd.read_parquet(path)
    mp['date'] = pd.to_datetime(mp['date'])
    mp = mp.sort_values('date').reset_index(drop=True)
    mpi = mp['mp_value'].rolling(window=window, min_periods=min_periods).min()
    return pd.DataFrame({'date': mp['date'], 'mpi': mpi.to_numpy()})

def process_region(region: str, cfg: DictConfig, e16: DictConfig, e2_dir: Path, e8_dir: Path, out_dir: Path) -> pd.DataFrame | None:
    kndvi_dir = BASE_DIR / cfg.data.kndvi_dir
    parquet_path = kndvi_dir / f'kndvi_{region}_2000_2025.parquet'
    df = _load_kndvi(parquet_path)
    dates = _date_labels(df)
    ts = _mean_ts(df)
    ts_anom = _deseasonalize(ts, dates)
    ts_detr = _gaussian_detrend(ts_anom, float(e16.detrend_sigma))
    mpi_df = _load_mpi(e2_dir, e8_dir, region, window=int(e16.window), min_periods=int(e16.mpi_min_periods), source=str(e16.mpi_source))
    frames = []
    for variant, series in (('deseason', ts_anom), ('deseason_detrend', ts_detr)):
        csd = _rolling_csd(series, dates, int(e16.window))
        csd['variant'] = variant
        merged = csd.merge(mpi_df, on='date', how='inner')
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    out = out[['date', 'variant', 'mpi', 'tac', 'phi_ar1', 'lambda_ar1', 'variance']]
    out.to_csv(out_dir / f'indicators_{region}.csv', index=False)
    log.info('  %s: n=%d months, merged rows/variant=%d', region, len(dates), len(out) // 2)
    return out

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    e16 = OmegaConf.merge(OmegaConf.structured(E16Config()), cfg.get('e16', {}))
    e2_dir = BASE_DIR / cfg.output.dir / 'E2'
    e8_dir = BASE_DIR / cfg.output.dir / 'E8'
    out_dir = BASE_DIR / cfg.output.dir / 'E16'
    out_dir.mkdir(parents=True, exist_ok=True)
    q = float(cfg.analysis.alpha)
    primary = str(e16.primary_variant)
    lag_max = int(e16.lag_max)
    regions = ['africa', 'amazon'] if cfg.run_mvp_only else REGION_ORDER
    log.info('E16: window=%d (matched to MPI), detrend_sigma=%.0f mo, lag_max=%d mo, primary variant=%s, FDR q=%.3f, mpi_source=%s', int(e16.window), float(e16.detrend_sigma), lag_max, primary, q, str(e16.mpi_source))
    stale_rows: list[dict] = []
    for region in REGION_ORDER:
        p2 = e2_dir / f'matrix_profile_{region}.parquet'
        p8 = e8_dir / f'mpi_{region}.csv'
        if not (p2.exists() and p8.exists()):
            continue
        a = _load_mpi(e2_dir, e8_dir, region, int(e16.window), int(e16.mpi_min_periods), 'e2_recompute')
        b = _load_mpi(e2_dir, e8_dir, region, int(e16.window), int(e16.mpi_min_periods), 'e8_stored')
        m = a.merge(b, on='date', suffixes=('_e2', '_e8'))
        d = np.abs(m['mpi_e2'].to_numpy() - m['mpi_e8'].to_numpy())
        stale_rows.append({'region': region, 'n_common_dates': len(m), 'max_abs_diff': round(float(np.nanmax(d)), 6), 'median_abs_diff': round(float(np.nanmedian(d)), 6), 'consistent_1e-6': bool(np.nanmax(d) < 1e-06)})
    if stale_rows:
        stale_df = pd.DataFrame(stale_rows)
        stale_df.to_csv(out_dir / 'e8_staleness_audit_E16.csv', index=False)
        n_bad = int((~stale_df['consistent_1e-6']).sum())
        if n_bad:
            log.warning('STALE E8: %d/%d regions where outputs/E8/mpi_*.csv does NOT match the current outputs/E2 matrix profile (max|diff| up to %.4f). E16 uses mpi_source=%s. See e8_staleness_audit_E16.csv.', n_bad, len(stale_df), float(stale_df['max_abs_diff'].max()), str(e16.mpi_source))
    trend_rows: list[dict] = []
    corr_rows: list[dict] = []
    for region in regions:
        try:
            tbl = process_region(region, cfg, e16, e2_dir, e8_dir, out_dir)
        except FileNotFoundError as exc:
            log.warning('Skipping %s — %s', region, exc)
            continue
        if tbl is None or tbl.empty:
            continue
        for variant, sub in tbl.groupby('variant'):
            sub = sub.sort_values('date').reset_index(drop=True)
            for metric in ALL_METRICS:
                res = _mk_tests(sub[metric].to_numpy(dtype=float))
                trend_rows.append({'region': region, 'variant': variant, 'metric': metric, **res})
            mpi_arr = sub['mpi'].to_numpy(dtype=float)
            for ind in INDICATORS:
                ind_arr = sub[ind].to_numpy(dtype=float)
                sp = _spearman_eff(mpi_arr, ind_arr)
                best_lag, rho_lag = _lag_scan(mpi_arr, ind_arr, lag_max)
                corr_rows.append({'region': region, 'variant': variant, 'indicator': ind, 'n': sp['n'], 'spearman_rho': round(sp['rho'], 4) if not np.isnan(sp['rho']) else np.nan, 'p_naive': sp['p_naive'], 'n_eff': sp['n_eff'], 'p_eff': sp['p_eff'], 'best_lag_months': best_lag, 'rho_at_best_lag': round(rho_lag, 4) if not np.isnan(rho_lag) else np.nan})
    trend_df = pd.DataFrame(trend_rows)
    corr_df = pd.DataFrame(corr_rows)
    trend_df['test_family'] = 'MK_trend'
    corr_df['test_family'] = 'MPI_vs_indicator_Spearman'
    fdr_pool = pd.concat([trend_df.loc[trend_df['variant'] == primary, ['test_family']].assign(p=trend_df.loc[trend_df['variant'] == primary, 'p_hr'], key=trend_df.loc[trend_df['variant'] == primary].index, src='trend'), corr_df.loc[corr_df['variant'] == primary, ['test_family']].assign(p=corr_df.loc[corr_df['variant'] == primary, 'p_eff'], key=corr_df.loc[corr_df['variant'] == primary].index, src='corr')], ignore_index=True)
    valid = fdr_pool['p'].notna()
    adj, rej = _bh_fdr(fdr_pool.loc[valid, 'p'].to_numpy(dtype=float), q)
    fdr_pool['p_bh'] = np.nan
    fdr_pool['fdr_significant'] = False
    fdr_pool.loc[valid, 'p_bh'] = adj
    fdr_pool.loc[valid, 'fdr_significant'] = rej
    log.info('BH-FDR over the full E16 family: %d tests (variant=%s), %d survive at q=%.2f.', int(valid.sum()), primary, int(rej.sum()), q)
    for src, target in (('trend', trend_df), ('corr', corr_df)):
        part = fdr_pool[fdr_pool['src'] == src]
        target['p_bh'] = np.nan
        target['fdr_significant'] = False
        target.loc[part['key'].to_numpy(), 'p_bh'] = part['p_bh'].to_numpy()
        target.loc[part['key'].to_numpy(), 'fdr_significant'] = part['fdr_significant'].to_numpy()
    trend_df.to_csv(out_dir / 'trend_tests_E16.csv', index=False)
    corr_df.to_csv(out_dir / 'mpi_correlation_E16.csv', index=False)
    log.info('Saved: %s, %s', out_dir / 'trend_tests_E16.csv', out_dir / 'mpi_correlation_E16.csv')
    tp = trend_df[trend_df['variant'] == primary]
    cp = corr_df[corr_df['variant'] == primary]
    mpi_trend = tp[tp['metric'] == 'mpi'].set_index('region')
    agree_rows: list[dict] = []
    for _, r in tp[tp['metric'] != 'mpi'].iterrows():
        region, ind = (r['region'], r['metric'])
        if region not in mpi_trend.index:
            continue
        m = mpi_trend.loc[region]
        c = cp[(cp['region'] == region) & (cp['indicator'] == ind)]
        c = c.iloc[0] if not c.empty else None
        tau_i, tau_m = (r['tau'], m['tau'])
        if np.isnan(tau_i) or np.isnan(tau_m):
            agrees = None
        else:
            agrees = bool(np.sign(tau_i) == np.sign(tau_m))
        agree_rows.append({'region': region, 'indicator': ind, 'mpi_tau': round(float(tau_m), 4) if not np.isnan(tau_m) else np.nan, 'mpi_trend_sign': '+' if tau_m > 0 else '-' if tau_m < 0 else '0', 'mpi_p_hr': round(float(m['p_hr']), 6), 'mpi_p_bh': round(float(m['p_bh']), 6) if not np.isnan(m['p_bh']) else np.nan, 'mpi_fdr_sig': bool(m['fdr_significant']), 'ind_tau': round(float(tau_i), 4) if not np.isnan(tau_i) else np.nan, 'ind_trend_sign': '+' if tau_i > 0 else '-' if tau_i < 0 else '0', 'ind_p_hr': round(float(r['p_hr']), 6), 'ind_p_bh': round(float(r['p_bh']), 6) if not np.isnan(r['p_bh']) else np.nan, 'ind_fdr_sig': bool(r['fdr_significant']), 'direction_agrees_with_mpi': agrees, 'spearman_rho': c['spearman_rho'] if c is not None else np.nan, 'spearman_p_eff': round(float(c['p_eff']), 6) if c is not None else np.nan, 'spearman_p_bh': round(float(c['p_bh']), 6) if c is not None and (not np.isnan(c['p_bh'])) else np.nan, 'spearman_fdr_sig': bool(c['fdr_significant']) if c is not None else False, 'best_lag_months': int(c['best_lag_months']) if c is not None else np.nan, 'rho_at_best_lag': c['rho_at_best_lag'] if c is not None else np.nan})
    agree_df = pd.DataFrame(agree_rows)
    agree_df['region'] = pd.Categorical(agree_df['region'], categories=REGION_ORDER, ordered=True)
    agree_df = agree_df.sort_values(['region', 'indicator']).reset_index(drop=True)
    agree_df.to_csv(out_dir / 'agreement_table_E16.csv', index=False)
    log.info('Saved agreement table: %s (%d rows)', out_dir / 'agreement_table_E16.csv', len(agree_df))
    summary_rows: list[dict] = []
    for ind in INDICATORS:
        sub = agree_df[agree_df['indicator'] == ind]
        ok = sub['direction_agrees_with_mpi'].dropna()
        n_reg, n_agree = (len(ok), int(ok.sum()))
        bt = binomtest(n_agree, n_reg, 0.5, alternative='two-sided') if n_reg > 0 else None
        rhos = sub['spearman_rho'].dropna().to_numpy(dtype=float)
        lags = sub['best_lag_months'].dropna().to_numpy(dtype=float)
        summary_rows.append({'indicator': ind, 'variant': primary, 'n_regions': n_reg, 'n_direction_agree': n_agree, 'frac_direction_agree': round(n_agree / n_reg, 4) if n_reg else np.nan, 'sign_test_p': round(float(bt.pvalue), 6) if bt is not None else np.nan, 'median_spearman_rho': round(float(np.median(rhos)), 4) if len(rhos) else np.nan, 'min_spearman_rho': round(float(np.min(rhos)), 4) if len(rhos) else np.nan, 'max_spearman_rho': round(float(np.max(rhos)), 4) if len(rhos) else np.nan, 'n_rho_fdr_sig': int(sub['spearman_fdr_sig'].sum()), 'n_abs_rho_gt_0.7': int(np.sum(np.abs(rhos) > 0.7)), 'median_best_lag_months': float(np.median(lags)) if len(lags) else np.nan, 'n_mpi_leads': int(np.sum(lags > 0)), 'n_indicator_leads': int(np.sum(lags < 0))})
    mpi_tp = tp[tp['metric'] == 'mpi']
    summary_rows.append({'indicator': 'mpi (reference)', 'variant': primary, 'n_regions': len(mpi_tp), 'n_direction_agree': np.nan, 'frac_direction_agree': np.nan, 'sign_test_p': np.nan, 'median_spearman_rho': np.nan, 'min_spearman_rho': np.nan, 'max_spearman_rho': np.nan, 'n_rho_fdr_sig': np.nan, 'n_abs_rho_gt_0.7': np.nan, 'median_best_lag_months': np.nan, 'n_mpi_leads': np.nan, 'n_indicator_leads': np.nan})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'summary_E16.csv', index=False)
    print('\n' + '=' * 78)
    print('E16 — MPI vs established critical-slowing-down indicators')
    print(f'Primary preprocessing variant : {primary}')
    print(f'Rolling window                : {int(e16.window)} months (centred, matched to MPI)')
    print(f'Trend test                    : Hamed-Rao modified Mann-Kendall')
    print(f'Multiple testing              : Benjamini-Hochberg, q = {q}')
    print('=' * 78)
    print(summary_df.to_string(index=False))
    print('\nTrend direction counts (primary variant):')
    for metric in ALL_METRICS:
        s = tp[tp['metric'] == metric]
        print(f"  {metric:<12s}  tau>0: {int((s['tau'] > 0).sum()):2d}/{len(s):2d}   FDR-significant: {int(s['fdr_significant'].sum()):2d}")
    print('=' * 78 + '\n')
    log.info('E16 complete.')
if __name__ == '__main__':
    main()
