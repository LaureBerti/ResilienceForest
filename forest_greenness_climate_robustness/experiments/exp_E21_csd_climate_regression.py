from __future__ import annotations
import logging
import sys
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_E6_rolling_correlation import _kndvi_anomaly
from exp_E16_resilience_benchmark import REGION_ORDER, _date_labels, _gaussian_detrend, _load_kndvi, _mean_ts, _rolling_csd
log = logging.getLogger(__name__)
MODES = ['AO', 'ENSO', 'NAO', 'PNA']
INDICATORS = ['tac', 'lambda_ar1', 'variance']
DEFAULTS = {'window': 24, 'detrend_sigma': 60.0, 'n_surr': 2000, 'seed': 42, 'anomalies': ['climatology', 'stl'], 'detrends': ['none', 'gaussian'], 'stats': ['mean', 'sd'], 'min_windows': 60, 'out_dir': 'E21', 'season_months': None, 'power_rhos': [0.3, 0.5, 0.7]}

def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    from statsmodels.stats.multitest import multipletests
    out[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return out

def _phase_surrogates(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    T, k = X.shape
    mu = X.mean(axis=0)
    F = np.fft.rfft(X - mu, axis=0)
    nf = F.shape[0]
    out = np.empty((n, T, k))
    for b in range(n):
        ph = rng.uniform(0, 2 * np.pi, nf)
        ph[0] = 0.0
        if T % 2 == 0:
            ph[-1] = 0.0
        out[b] = np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0) + mu
    return out

def _rolling_csd_season(x: np.ndarray, dates: pd.DatetimeIndex, window: int, months: list[int]) -> pd.DataFrame:
    T = len(x)
    ins = np.isin(dates.month, months) & ~np.isnan(x)
    consec = np.zeros(T, bool)
    gap = dates[1:].year * 12 + dates[1:].month - (dates[:-1].year * 12 + dates[:-1].month)
    consec[:-1] = ins[:-1] & ins[1:] & (gap == 1)
    tac = np.full(T, np.nan)
    phi = np.full(T, np.nan)
    var = np.full(T, np.nan)
    lo_off = window // 2
    for i in range(T):
        lo, hi = (i - lo_off, i - lo_off + window)
        if lo < 0 or hi > T:
            continue
        sel = ins[lo:hi]
        if sel.sum() < 6:
            continue
        xc = x[lo:hi] - x[lo:hi][sel].mean()
        var[i] = float(np.sum(xc[sel] ** 2) / (sel.sum() - 1))
        pr = consec[lo:hi - 1]
        if pr.sum() < 4:
            continue
        num = float(np.sum(xc[:-1][pr] * xc[1:][pr]))
        tac[i] = num / float(np.sum(xc[sel] ** 2))
        d = float(np.sum(xc[:-1][pr] ** 2))
        phi[i] = num / d if d > 0 else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.where(phi > 0, np.log(phi), np.nan)
    return pd.DataFrame({'date': dates, 'tac': tac, 'phi_ar1': phi, 'lambda_ar1': lam, 'variance': var})

def _rolling_stat(x: np.ndarray, window: int, stat: str) -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(window=window, center=True, min_periods=window)
    return (r.mean() if stat == 'mean' else r.std(ddof=1)).to_numpy()

def _rolling_stat_batch(Xs: np.ndarray, window: int, stat: str) -> np.ndarray:
    n, T, k = Xs.shape
    c = np.concatenate([np.zeros((n, 1, k)), np.cumsum(Xs, axis=1)], axis=1)
    c2 = np.concatenate([np.zeros((n, 1, k)), np.cumsum(Xs ** 2, axis=1)], axis=1)
    out = np.full((n, T, k), np.nan)
    lo_off = window // 2
    for i in range(T):
        lo, hi = (i - lo_off, i - lo_off + window)
        if lo < 0 or hi > T:
            continue
        s = c[:, hi] - c[:, lo]
        if stat == 'mean':
            out[:, i] = s / window
        else:
            s2 = c2[:, hi] - c2[:, lo]
            out[:, i] = np.sqrt(np.maximum((s2 - s * s / window) / (window - 1), 0.0))
    return out

def _zscore(a: np.ndarray, axis: int) -> np.ndarray:
    return (a - a.mean(axis=axis, keepdims=True)) / a.std(axis=axis, ddof=0, keepdims=True)

def _stats_on(y: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    yz = (y - y.mean()) / y.std()
    Pz = _zscore(P, axis=-2)
    W = len(y)
    r = np.einsum('...wk,w->...k', Pz, yz) / W
    G = np.einsum('...wk,...wl->...kl', Pz, Pz)
    h = np.einsum('...wk,w->...k', Pz, yz)
    beta = np.linalg.solve(G, h[..., None])[..., 0]
    r2 = np.einsum('...k,...k->...', beta, h) / W
    return (r, beta, r2)

def _region_series(region: str, cfg: DictConfig) -> tuple[pd.Series, pd.DatetimeIndex]:
    path = BASE_DIR / cfg.data.kndvi_dir / f'kndvi_{region}_2000_2025.parquet'
    df = _load_kndvi(path)
    dates = _date_labels(df)
    return (pd.Series(_mean_ts(df), index=dates), dates)

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    e21 = OmegaConf.merge(OmegaConf.create(DEFAULTS), cfg.get('e21', {}))
    out_dir = BASE_DIR / cfg.output.dir / str(e21.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w = int(e21.window)
    q = float(cfg.analysis.alpha)
    rng = np.random.default_rng(int(e21.seed))
    clim = pd.read_csv(BASE_DIR / cfg.data.climate_dir / 'climate_indicators_2000_2025.csv', parse_dates=['date']).set_index('date').sort_index()[MODES]
    clim = clim.loc['2000-01-01':'2024-12-01'].dropna()
    Xc = clim.to_numpy(dtype=float)
    surr = _phase_surrogates(Xc, int(e21.n_surr), rng)
    surr_df_index = clim.index
    log.info('Climate matrix %s, %d surrogates', Xc.shape, int(e21.n_surr))
    regions = ['africa', 'amazon'] if cfg.run_mvp_only else REGION_ORDER
    marg_rows, joint_rows, power_rows = ([], [], [])
    for region in regions:
        raw, dates = _region_series(region, cfg)
        pos = surr_df_index.get_indexer(dates)
        if (pos < 0).any():
            keep = pos >= 0
            raw, dates, pos = (raw[keep], dates[keep], pos[keep])
        Xr = Xc[pos]
        Sr = surr[:, pos, :]
        for anom in e21.anomalies:
            a = _kndvi_anomaly(raw, str(anom)).to_numpy(dtype=float)
            for detr in e21.detrends:
                x = _gaussian_detrend(a, float(e21.detrend_sigma)) if detr == 'gaussian' else a
                csd = _rolling_csd(x, dates, w) if e21.season_months is None else _rolling_csd_season(x, dates, w, list(e21.season_months))
                for stat in e21.stats:
                    P_obs = np.column_stack([_rolling_stat(Xr[:, j], w, str(stat)) for j in range(4)])
                    P_null = _rolling_stat_batch(Sr, w, str(stat))
                    for ind in INDICATORS:
                        y = csd[ind].to_numpy(dtype=float)
                        ok = ~np.isnan(y) & ~np.isnan(P_obs).any(axis=1)
                        if ok.sum() < int(e21.min_windows):
                            continue
                        r, beta, r2 = _stats_on(y[ok], P_obs[ok])
                        rn, bn, r2n = _stats_on(y[ok], P_null[:, ok, :])
                        n = rn.shape[0]
                        p_r = (1 + (np.abs(rn) >= np.abs(r)[None]).sum(0)) / (n + 1)
                        p_b = (1 + (np.abs(bn) >= np.abs(beta)[None]).sum(0)) / (n + 1)
                        p_r2 = (1 + (r2n >= r2).sum()) / (n + 1)
                        for j, mode in enumerate(MODES):
                            marg_rows.append(dict(anomaly=anom, detrend=detr, stat=stat, indicator=ind, region=region, mode=mode, n_windows=int(ok.sum()), r=r[j], p_surr=p_r[j], beta_joint=beta[j], p_beta_surr=p_b[j], r_null_q95=float(np.quantile(np.abs(rn[:, j]), 0.95))))
                        joint_rows.append(dict(anomaly=anom, detrend=detr, stat=stat, indicator=ind, region=region, n_windows=int(ok.sum()), r2=r2, p_surr=p_r2, r2_null_median=float(np.median(r2n)), r2_null_q95=float(np.quantile(r2n, 0.95))))
                    if anom == 'climatology' and detr == 'none' and (stat == 'mean'):
                        y0 = csd['tac'].to_numpy(dtype=float)
                        ok = ~np.isnan(y0) & ~np.isnan(P_obs).any(axis=1)
                        e = (y0[ok] - y0[ok].mean()) / y0[ok].std()
                        z = P_obs[ok, 1]
                        z = (z - z.mean()) / z.std()
                        e = e - e @ z / (z @ z) * z
                        e = e / e.std()
                        for rho in e21.power_rhos:
                            ys = float(rho) * z + np.sqrt(1 - float(rho) ** 2) * e
                            rs, _, _ = _stats_on(ys, P_obs[ok])
                            rns, _, _ = _stats_on(ys, P_null[:, ok, :])
                            pw = (1 + (np.abs(rns[:, 1]) >= abs(rs[1])).sum()) / (rns.shape[0] + 1)
                            power_rows.append(dict(region=region, rho=float(rho), r_obs=rs[1], p_surr=pw))
        log.info('  %s done', region)
    fam = ['anomaly', 'detrend', 'stat', 'indicator']
    M = pd.DataFrame(marg_rows)
    M['p_bh'] = M.groupby(fam)['p_surr'].transform(lambda s: _bh(s.to_numpy()))
    M['p_beta_bh'] = M.groupby(fam)['p_beta_surr'].transform(lambda s: _bh(s.to_numpy()))
    J = pd.DataFrame(joint_rows)
    J['p_bh'] = J.groupby(fam)['p_surr'].transform(lambda s: _bh(s.to_numpy()))
    M.to_csv(out_dir / 'marginal_E21.csv', index=False)
    J.to_csv(out_dir / 'joint_E21.csv', index=False)
    key = ['detrend', 'stat', 'indicator', 'region', 'mode']
    piv = M.pivot_table(index=key, columns='anomaly', values=['r', 'p_bh']).reset_index()
    piv.columns = ['_'.join([c for c in col if c]) for col in piv.columns]
    rob = piv[(piv['p_bh_climatology'] < q) & (piv['p_bh_stl'] < q) & (np.sign(piv['r_climatology']) == np.sign(piv['r_stl']))]
    rob.to_csv(out_dir / 'robust_E21.csv', index=False)
    s1 = M.groupby(fam).agg(n_tests=('p_surr', 'size'), n_p05=('p_surr', lambda s: int((s < q).sum())), n_bh=('p_bh', lambda s: int((s < q).sum())), median_abs_r=('r', lambda s: float(np.median(np.abs(s)))), max_abs_r=('r', lambda s: float(np.max(np.abs(s))))).reset_index()
    s2 = J.groupby(fam).agg(n_joint_bh=('p_bh', lambda s: int((s < q).sum())), median_r2=('r2', 'median'), median_r2_null=('r2_null_median', 'median')).reset_index()
    S = s1.merge(s2, on=fam)
    S.to_csv(out_dir / 'summary_E21.csv', index=False)
    print(S.to_string(index=False))
    PW = pd.DataFrame(power_rows)
    PW.to_csv(out_dir / 'power_E21.csv', index=False)
    print(PW.groupby('rho').agg(frac_p05=('p_surr', lambda s: float((s < q).mean())), median_p=('p_surr', 'median')).to_string())
    print(f'\nRobust (BH<{q} under climatology AND STL, same sign): {len(rob)}')
    print(rob.to_string(index=False))
if __name__ == '__main__':
    main()
