from __future__ import annotations
import logging
import time
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
log = logging.getLogger('E20')
BASE_DIR = Path(__file__).resolve().parent.parent

def _anomaly(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(values, index=dates, dtype=np.float64)
    return (s - s.groupby(s.index.month).transform('mean')).to_numpy(np.float64)

def _mpi(mp: np.ndarray, window: int, min_periods: int | None=None) -> np.ndarray:
    mp_ = window // 2 if min_periods is None else min_periods
    return pd.Series(mp).rolling(window=window, min_periods=mp_).min().to_numpy(np.float64)

def _chain(series: np.ndarray, dates: pd.DatetimeIndex, m: int, stumpy, min_periods: int | None=None) -> dict:
    a = _anomaly(series, dates)
    mp = stumpy.stump(a, m)[:, 0].astype(np.float64)
    mpi = _mpi(mp, m, min_periods)
    return {'anom': a, 'mp': mp, 'mpi': mpi, 'best_motif': float(np.min(mp)), 'mean_mp': float(np.mean(mp))}

def _load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, int]:
    df = pd.read_parquet(path, columns=['pixel_id', 'date', 'kndvi'])
    df['date'] = pd.to_datetime(df['date'])
    if df.duplicated(['pixel_id', 'date']).any():
        raise ValueError(f'{path.name}: duplicated (pixel_id, date) rows; pivot would be ambiguous')
    pix_codes, pix_uniques = pd.factorize(df['pixel_id'], sort=True)
    dates = pd.DatetimeIndex(sorted(df['date'].unique()))
    date_codes = dates.get_indexer(df['date'])
    X = np.full((len(pix_uniques), len(dates)), np.nan, dtype=np.float64)
    X[pix_codes, date_codes] = df['kndvi'].to_numpy(np.float64)
    del df
    V = np.isfinite(X).astype(np.float64)
    X0 = np.where(np.isfinite(X), X, 0.0)
    return (X0, V, dates, len(pix_uniques))

def _weighted_mean(W: np.ndarray, X0: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    den = W @ V
    num = W @ X0
    with np.errstate(invalid='ignore', divide='ignore'):
        return (num / den, den)

def _draw(rng: np.random.Generator, kind: str, size: int, n_pix: int) -> np.ndarray:
    if kind == 'poisson':
        return rng.poisson(1.0, size=(size, n_pix)).astype(np.float64)
    if kind == 'multinomial':
        return rng.multinomial(n_pix, np.full(n_pix, 1.0 / n_pix), size=size).astype(np.float64)
    raise ValueError(f'unknown e20.weights: {kind!r}')

def _reference(region: str, cfg: DictConfig) -> dict:
    from exp_E2_matrix_profile import _date_labels, _deseasonalize, _load_kndvi, _mean_ts
    kndvi_dir = BASE_DIR / cfg.data.kndvi_dir
    df = _load_kndvi(kndvi_dir / f'kndvi_{region}_2000_2025.parquet', region)
    dates = _date_labels(df)
    ts = _mean_ts(df)
    del df
    out = BASE_DIR / cfg.output.dir
    mp_e2 = pd.read_parquet(out / 'E2' / f'matrix_profile_{region}.parquet')
    e8 = pd.read_csv(out / 'E8' / f'mpi_{region}.csv', parse_dates=['date'])
    motifs = pd.read_csv(out / 'E2' / f'motifs_{region}.csv')
    return {'dates': dates, 'ts': ts, 'anom': _deseasonalize(ts, dates), 'mp': mp_e2.sort_values('date')['mp_value'].to_numpy(np.float64), 'mpi_dates': pd.DatetimeIndex(e8['date']), 'mpi': e8['mpi_rolling_min'].to_numpy(np.float64), 'best_motif': float(motifs.loc[motifs.motif_rank == 1, 'distance'].iloc[0])}

def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    a, b = (np.asarray(a, np.float64), np.asarray(b, np.float64))
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float('inf')
    ok = ~np.isnan(a)
    return float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0

def _check(series: np.ndarray, dates: pd.DatetimeIndex, res: dict, ref: dict, n_mp: int, tol: float) -> dict:
    d = {'d_dates': 0.0 if dates.equals(ref['dates']) else float('inf'), 'd_series': _maxdiff(series, ref['ts']), 'd_anomaly': _maxdiff(res['anom'], ref['anom']), 'd_mp': _maxdiff(res['mp'], ref['mp']), 'd_mpi_dates': 0.0 if dates[:n_mp].equals(ref['mpi_dates']) else float('inf'), 'd_mpi': _maxdiff(res['mpi'], ref['mpi']), 'd_best_motif': abs(res['best_motif'] - ref['best_motif'])}
    d['max_dev'] = max(d.values())
    d['pass'] = bool(d['max_dev'] <= tol)
    return d

def _band(point: np.ndarray, boot: np.ndarray, dates, q: tuple[float, float]) -> pd.DataFrame:
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lo, hi = np.nanpercentile(boot, q, axis=0)
            mean = np.nanmean(boot, axis=0)
            sd = np.nanstd(boot, axis=0, ddof=1)
    return pd.DataFrame({'date': dates, 'point': point, 'q_lo': lo, 'q_hi': hi, 'boot_mean': mean, 'boot_sd': sd})

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    import stumpy
    from exp_E13_hamed_rao_mk import hamed_rao_mk
    e = cfg.e20
    m = int(cfg.matrix_profile.subsequence_length)
    B, chunk, tol = (int(e.n_boot), int(e.chunk), float(e.acceptance_tol))
    alpha = float(e.alpha)
    qlo, qhi = (float(x) for x in e.ci)
    regions = list(e.regions) or list(cfg.regions.continents) + list(cfg.regions.subregions)
    rng = np.random.default_rng(int(cfg.analysis.seed))
    out_dir = BASE_DIR / cfg.output.dir / 'E20'
    out_dir.mkdir(parents=True, exist_ok=True)
    kndvi_dir = BASE_DIR / cfg.data.kndvi_dir
    log.info('E20: %d regions, B=%d (%s weights), m=%d, seed=%d', len(regions), B, e.weights, m, int(cfg.analysis.seed))
    t_all = time.perf_counter()
    rows, acc_rows = ([], [])
    for region in regions:
        t0 = time.perf_counter()
        X0, V, dates, n_pix = _load_matrix(kndvi_dir / f'kndvi_{region}_2000_2025.parquet')
        n_valid_month = V.sum(axis=0)
        log.info('── %s: %d pixels x %d months (%.2f GB); valid pixels per month min=%d median=%d', region, n_pix, len(dates), 2 * X0.nbytes / 1000000000.0, int(n_valid_month.min()), int(np.median(n_valid_month)))
        ones = np.ones((1, n_pix))
        series_pt = _weighted_mean(ones, X0, V)[0][0]
        pt = _chain(series_pt, dates, m, stumpy)
        n_mp = len(pt['mp'])
        ref = _reference(region, cfg)
        acc = _check(series_pt, dates, pt, ref, n_mp, tol)
        bad_mp = _chain(series_pt, dates, m, stumpy, min_periods=m)
        neg_a = _check(series_pt, dates, bad_mp, ref, n_mp, tol)
        w_drop = ones.copy()
        w_drop[0, int(np.argmax(V.sum(axis=1)))] = 0.0
        s_drop = _weighted_mean(w_drop, X0, V)[0][0]
        neg_b = _check(s_drop, dates, _chain(s_drop, dates, m, stumpy), ref, n_mp, tol)
        acc_rows.append({'region': region, **acc, 'neg_minperiods24_max_dev': neg_a['max_dev'], 'neg_minperiods24_rejected': not neg_a['pass'], 'neg_drop1pixel_max_dev': neg_b['max_dev'], 'neg_drop1pixel_rejected': not neg_b['pass']})
        log.info('  ACCEPTANCE %s  max dev %.2e (series %.1e, anomaly %.1e, MP %.1e, MPI %.1e, motif %.1e) | negative controls rejected: min_periods=24 %s (dev %.2e), drop 1 pixel %s (dev %.2e)', 'PASS' if acc['pass'] else 'FAIL', acc['max_dev'], acc['d_series'], acc['d_anomaly'], acc['d_mp'], acc['d_mpi'], acc['d_best_motif'], not neg_a['pass'], neg_a['max_dev'], not neg_b['pass'], neg_b['max_dev'])
        assert acc['pass'], f'E20 acceptance test FAILED for {region}: {acc}. The bootstrap harness does not reproduce E2/E8; do not adjust E2/E8 to match.'
        assert not neg_a['pass'] and (not neg_b['pass']), f'E20 negative control ACCEPTED a perturbed chain for {region}; test is toothless.'
        mpi_pt = pt['mpi']
        tau_pt, p_pt, _ = hamed_rao_mk(mpi_pt[~np.isnan(mpi_pt)])
        T = len(dates)
        anom_b = np.empty((B, T))
        mp_b = np.empty((B, n_mp))
        mpi_b = np.empty((B, n_mp))
        tau_b, p_b = (np.empty(B), np.empty(B))
        bm_b, mmp_b = (np.empty(B), np.empty(B))
        n_redraw = 0
        for start in range(0, B, chunk):
            size = min(chunk, B - start)
            W = _draw(rng, e.weights, size, n_pix)
            means, den = _weighted_mean(W, X0, V)
            bad = np.where((den <= 0).any(axis=1))[0]
            while len(bad):
                n_redraw += len(bad)
                W[bad] = _draw(rng, e.weights, len(bad), n_pix)
                means[bad], den[bad] = _weighted_mean(W[bad], X0, V)
                bad = np.where((den <= 0).any(axis=1))[0]
            for k in range(size):
                b = start + k
                r = _chain(means[k], dates, m, stumpy)
                anom_b[b], mp_b[b], mpi_b[b] = (r['anom'], r['mp'], r['mpi'])
                bm_b[b], mmp_b[b] = (r['best_motif'], r['mean_mp'])
                tau_b[b], p_b[b], _ = hamed_rao_mk(r['mpi'][~np.isnan(r['mpi'])])
        del X0, V
        _band(pt['anom'], anom_b, dates, (qlo, qhi)).to_csv(out_dir / f'anomaly_band_{region}.csv', index=False)
        _band(mpi_pt, mpi_b, dates[:n_mp], (qlo, qhi)).to_csv(out_dir / f'mpi_band_{region}.csv', index=False)
        if bool(e.save_replicates):
            np.savez_compressed(out_dir / f'replicates_{region}.npz', dates=dates.strftime('%Y-%m-%d').to_numpy(), anomaly=anom_b, mp=mp_b, mpi=mpi_b, tau=tau_b, p_hamedrao=p_b, best_motif=bm_b, mean_mp=mmp_b)
        valid_mpi = ~np.isnan(mpi_pt)
        assert np.array_equal(np.isnan(mpi_b), np.broadcast_to(~valid_mpi, mpi_b.shape))
        runtime = time.perf_counter() - t0
        sign_pt = np.sign(tau_pt)
        rows.append({'region': region, 'n_pixels': n_pix, 'n_months': T, 'n_mpi': int((~np.isnan(mpi_pt)).sum()), 'B': B, 'weights': e.weights, 'tau_point': tau_pt, 'p_hamedrao_point': p_pt, 'tau_ci_lo': np.percentile(tau_b, qlo), 'tau_ci_hi': np.percentile(tau_b, qhi), 'frac_same_sign': float(np.mean(np.sign(tau_b) == sign_pt)), 'frac_p_lt_alpha': float(np.mean(p_b < alpha)), 'best_motif_point': pt['best_motif'], 'best_motif_ci_lo': np.percentile(bm_b, qlo), 'best_motif_ci_hi': np.percentile(bm_b, qhi), 'mean_mp_point': pt['mean_mp'], 'mean_mp_ci_lo': np.percentile(mmp_b, qlo), 'mean_mp_ci_hi': np.percentile(mmp_b, qhi), 'mpi_band_median_width': float(np.median(np.percentile(mpi_b[:, valid_mpi], qhi, axis=0) - np.percentile(mpi_b[:, valid_mpi], qlo, axis=0))), 'n_redrawn_replicates': n_redraw, 'draw_acceptance_rate': B / (B + n_redraw), 'min_valid_pixels_per_month': int(n_valid_month.min()), 'n_months_lt30_pixels': int((n_valid_month < 30).sum()), 'runtime_s': round(runtime, 1)})
        log.info('  tau=%.3f [%.3f, %.3f]  same-sign %.2f  p<%.2f in %.2f  best motif %.3f [%.3f, %.3f]  (%.1f s)', tau_pt, rows[-1]['tau_ci_lo'], rows[-1]['tau_ci_hi'], rows[-1]['frac_same_sign'], alpha, rows[-1]['frac_p_lt_alpha'], pt['best_motif'], rows[-1]['best_motif_ci_lo'], rows[-1]['best_motif_ci_hi'], runtime)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'summary_E20.csv', index=False)
    pd.DataFrame(acc_rows).to_csv(out_dir / 'acceptance_E20.csv', index=False)
    total = time.perf_counter() - t_all
    log.info('Summary:\n%s', summary[['region', 'n_pixels', 'tau_point', 'tau_ci_lo', 'tau_ci_hi', 'frac_same_sign', 'frac_p_lt_alpha']].round(3).to_string(index=False))
    log.info('E20 complete: %d regions, B=%d, total runtime %.1f s (%.1f min)', len(regions), B, total, total / 60)
if __name__ == '__main__':
    main()
