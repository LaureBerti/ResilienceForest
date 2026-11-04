from __future__ import annotations
import logging
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
log = logging.getLogger(__name__)

def _load_kndvi_mean(raw_dir: Path, region: str) -> pd.Series:
    path = raw_dir / 'kndvi' / f'kndvi_{region}_2000_2025.parquet'
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date')['kndvi'].mean().sort_index()

def _anomaly(series: pd.Series, method: str) -> pd.Series:
    if method == 'climatology':
        return series - series.groupby(series.index.month).transform('mean')
    if method == 'rollmean12':
        return series - series.rolling(12, center=True, min_periods=6).mean()
    if method == 'none':
        return series
    raise ValueError(method)

def _enso_years(climate_csv: Path, thr: float, min_run: int) -> set[int]:
    c = pd.read_csv(climate_csv, parse_dates=['date']).set_index('date')['ENSO'].dropna()
    flag = (c.abs() >= thr).astype(int).to_numpy()
    years: set[int] = set()
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j < len(flag) and flag[j]:
                j += 1
            if j - i >= min_run:
                years.update(c.index[i:j].year.tolist())
            i = j
        else:
            i += 1
    return years

def _bic_l2(signal: np.ndarray, bkps_end: list[int]) -> float:
    n = len(signal)
    edges = [0] + list(bkps_end)
    rss = sum((float(((signal[a:b] - signal[a:b].mean()) ** 2).sum()) for a, b in zip(edges[:-1], edges[1:])))
    k = len(bkps_end) - 1
    n_par = k + 1 + k + 1
    return n * np.log(rss / n) + n_par * np.log(n)

def _random_breaks(n: int, k: int, min_size: int, rng: np.random.Generator) -> list[int]:
    while True:
        b = np.sort(rng.choice(np.arange(min_size, n - min_size), size=k, replace=False))
        if np.all(np.diff(b) >= min_size):
            return b.tolist()

def _detect_pelt(signal_arr: np.ndarray, pen: float=10.0) -> list[int]:
    import ruptures as rpt
    algo = rpt.Pelt(model='rbf', min_size=3, jump=1)
    algo.fit(signal_arr.reshape(-1, 1))
    try:
        result = algo.predict(pen=pen)
    except Exception as exc:
        log.warning('PELT failed: %s', exc)
        return []
    return [bp - 1 for bp in result if bp < len(signal_arr)]

def _detect_binseg(signal_arr: np.ndarray, n_bkps: int=3) -> list[int]:
    import ruptures as rpt
    algo = rpt.Binseg(model='l2', min_size=3, jump=1)
    algo.fit(signal_arr.reshape(-1, 1))
    try:
        result = algo.predict(n_bkps=n_bkps)
    except Exception as exc:
        log.warning('BinSeg failed: %s', exc)
        return []
    return [bp - 1 for bp in result if bp < len(signal_arr)]

def _breakpoints_to_df(indices: list[int], dates: pd.DatetimeIndex, region: str, method: str, penalty: float | int | str, enso_years: set[int], signal: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx in indices:
        if 0 <= idx < len(dates):
            rows.append({'region': region, 'break_index': idx, 'break_date': dates[idx], 'break_year': dates[idx].year, 'method': method, 'penalty': str(penalty), 'in_enso_year': dates[idx].year in enso_years, 'mean_anom_12m_before': float(np.mean(signal[max(0, idx - 11):idx + 1])), 'mean_anom_12m_after': float(np.mean(signal[idx + 1:idx + 13])) if idx + 1 < len(signal) else float('nan')})
    return pd.DataFrame(rows)

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    raw_dir = base_dir / cfg.data.raw_dir
    out_dir = base_dir / cfg.output.dir / 'E7'
    out_dir.mkdir(parents=True, exist_ok=True)
    all_regions = list(cfg.regions.continents) + list(cfg.regions.subregions)
    if cfg.run_mvp_only:
        all_regions = ['africa', 'amazon']
    e7 = cfg.get('e7', {})
    n_bkps = int(cfg.analysis.n_changepoints)
    pelt_pen = float(e7.get('pelt_penalty', 10.0))
    min_size = int(e7.get('min_size', 3))
    n_perm = int(e7.get('n_perm', 10000))
    thr = float(e7.get('oni_threshold', 0.5))
    min_run = int(e7.get('min_run', 5))
    method = str(cfg.analysis.get('kndvi_anomaly', 'climatology'))
    rng = np.random.default_rng(int(cfg.analysis.seed))
    enso_years = _enso_years(raw_dir / 'climate' / 'climate_indicators_2000_2025.csv', thr, min_run)
    log.info('E7: anomaly=%s | ENSO years (|ONI|>=%.1f, >=%d months): %s', method, thr, min_run, sorted(enso_years))
    import ruptures as rpt
    frames, summ, bic_rows = ([], [], [])
    pooled_obs, pooled_null = (0, np.zeros(n_perm, dtype=int))
    for region in all_regions:
        if not (raw_dir / 'kndvi' / f'kndvi_{region}_2000_2025.parquet').exists():
            log.warning('kNDVI file missing for region=%s, skipping.', region)
            continue
        series = _anomaly(_load_kndvi_mean(raw_dir, region), method)
        sig = series.to_numpy(dtype=np.float64)
        dates = series.index
        n = len(sig)
        year_of = np.array(dates.year)
        eligible = np.arange(min_size, n - min_size)
        chance = float(np.isin(year_of[eligible], list(enso_years)).mean())
        pelt_idx = _detect_pelt(sig, pen=pelt_pen)
        bs_idx = _detect_binseg(sig, n_bkps=n_bkps)
        df = pd.concat([_breakpoints_to_df(pelt_idx, dates, region, 'PELT', pelt_pen, enso_years, sig), _breakpoints_to_df(bs_idx, dates, region, 'BinSeg', n_bkps, enso_years, sig)], ignore_index=True)
        df.to_csv(out_dir / f'changepoints_{region}.csv', index=False)
        frames.append(df)
        algo = rpt.Binseg(model='l2', min_size=min_size, jump=1).fit(sig.reshape(-1, 1))
        bics = {k: _bic_l2(sig, algo.predict(n_bkps=k)) for k in (1, 2, 3, 4)}
        k_best = min(bics, key=bics.get)
        bic_rows.append({'region': region, **{f'bic_k{k}': round(v, 3) for k, v in bics.items()}, 'k_bic': k_best})
        obs = int(np.isin(year_of[bs_idx], list(enso_years)).sum()) if bs_idx else 0
        null = np.array([int(np.isin(year_of[_random_breaks(n, n_bkps, min_size, rng)], list(enso_years)).sum()) for _ in range(n_perm)])
        pooled_obs += obs
        pooled_null += null
        d = df[df.method == 'BinSeg']
        summ.append({'region': region, 'anomaly': method, 'n_binseg': len(bs_idx), 'n_pelt': len(pelt_idx), 'binseg_in_enso': obs, 'chance_rate': round(chance, 4), 'expected_in_enso': round(chance * len(bs_idx), 3), 'perm_p_ge': round(float((null >= obs).mean()), 4), 'break_dates': ';'.join(d.break_date.dt.strftime('%Y-%m')), 'directions': ';'.join(np.where(d.mean_anom_12m_after > d.mean_anom_12m_before, 'up', 'down')), 'k_bic': k_best})
        log.info('  %s: BinSeg %s in-ENSO %d/%d (chance %.2f, p=%.3f); PELT %d; BIC k=%d', region, summ[-1]['break_dates'], obs, len(bs_idx), chance, summ[-1]['perm_p_ge'], len(pelt_idx), k_best)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(out_dir / 'breakpoint_summary_E7.csv', index=False)
    pd.DataFrame(bic_rows).to_csv(out_dir / 'bic_selection_E7.csv', index=False)
    pooled = {'anomaly': method, 'oni_threshold': thr, 'min_run': min_run, 'enso_years': ' '.join(map(str, sorted(enso_years))), 'n_breaks': int(sdf.n_binseg.sum()), 'observed_in_enso': pooled_obs, 'expected_in_enso': round(float(pooled_null.mean()), 3), 'perm_p_ge': round(float((pooled_null >= pooled_obs).mean()), 4), 'n_regions_k_bic_eq_3': int((pd.DataFrame(bic_rows).k_bic == n_bkps).sum()), 'n_regions_pelt_any': int((sdf.n_pelt > 0).sum()), 'n_perm': n_perm}
    pd.DataFrame([pooled]).to_csv(out_dir / 'pooled_null_E7.csv', index=False)
    log.info('E7 pooled: %s', pooled)
if __name__ == '__main__':
    main()
