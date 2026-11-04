from __future__ import annotations
import logging
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
log = logging.getLogger(__name__)
DEFAULT_VARIANTS = ['climatology', 'rollmean12', 'stl', 'none']

def _anomaly(s: pd.Series, variant: str) -> pd.Series:
    if variant == 'climatology':
        return s - s.groupby(s.index.month).transform('mean')
    if variant == 'rollmean12':
        return (s - s.rolling(12, center=True, min_periods=6).mean()).ffill().bfill()
    if variant == 'stl':
        from statsmodels.tsa.seasonal import STL
        res = STL(s, period=12, robust=True).fit()
        return s - res.seasonal
    if variant == 'none':
        return s.copy()
    raise ValueError(f'unknown deseasonalisation variant: {variant!r}')

def _mpi(mp_values: np.ndarray, window: int) -> pd.Series:
    return pd.Series(mp_values).rolling(window=window, min_periods=window // 2).min()

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    import stumpy
    from exp_E10_fdr_correction import _bh_fdr
    from exp_E13_hamed_rao_mk import hamed_rao_mk
    base = Path(hydra.utils.get_original_cwd())
    kndvi_dir = base / cfg.data.kndvi_dir if 'kndvi_dir' in cfg.data else base / 'data' / 'raw' / 'kndvi'
    out_dir = base / cfg.output.dir / 'E19'
    out_dir.mkdir(parents=True, exist_ok=True)
    m = int(cfg.matrix_profile.subsequence_length)
    q = float(cfg.analysis.alpha)
    variants = list(cfg.get('sensitivity', {}).get('variants', DEFAULT_VARIANTS))
    regions = sorted((p.stem.replace('kndvi_', '').replace('_2000_2025', '') for p in kndvi_dir.glob('kndvi_*_2000_2025.parquet')))
    log.info('E19: %d regions x %d variants (m=%d, q=%.2f)', len(regions), len(variants), m, q)
    rows: list[dict] = []
    for region in regions:
        df = pd.read_parquet(kndvi_dir / f'kndvi_{region}_2000_2025.parquet', columns=['date', 'kndvi'])
        s = df.groupby('date')['kndvi'].mean().sort_index()
        for variant in variants:
            a = _anomaly(s, variant).to_numpy(dtype=np.float64)
            mp = stumpy.stump(a, m)[:, 0].astype(np.float64)
            mpi = _mpi(mp, m).dropna().to_numpy()
            tau, p_hr, _ = hamed_rao_mk(mpi)
            rows.append({'region': region, 'variant': variant, 'n': len(mpi), 'tau': tau, 'p_hamedrao': p_hr})
        log.info('  %s done', region)
    res = pd.DataFrame(rows)
    res['p_bh'] = np.nan
    res['significant'] = False
    for variant, grp in res.groupby('variant'):
        adj, rej = _bh_fdr(grp['p_hamedrao'].to_numpy(dtype=float), q)
        res.loc[grp.index, 'p_bh'] = adj
        res.loc[grp.index, 'significant'] = rej
    res['sign'] = np.where(res['tau'] >= 0, '+', '-')
    res.to_csv(out_dir / 'deseason_sensitivity_E19.csv', index=False)
    log.info('Saved: %s', out_dir / 'deseason_sensitivity_E19.csv')
    e18_path = base / cfg.output.dir / 'E18' / 'mpi_trend_corrected_E18.csv'
    if e18_path.exists():
        e18 = pd.read_csv(e18_path).set_index('region')
        clim = res[res.variant == 'climatology'].set_index('region')
        common = clim.index.intersection(e18.index)
        dtau = (clim.loc[common, 'tau'] - e18.loc[common, 'tau']).abs().max()
        dp = (clim.loc[common, 'p_hamedrao'] - e18.loc[common, 'p_hamedrao']).abs().max()
        status = 'PASS' if dtau < 0.0001 and dp < 0.0001 else 'FAIL'
        log.info('ACCEPTANCE TEST vs E18: %s (max |dtau|=%.2e, max |dp|=%.2e over %d regions)', status, dtau, dp, len(common))
        if status == 'FAIL':
            log.warning("Variant 'climatology' does not reproduce E18. This harness's MPI construction has drifted from exp_E8; do NOT adjust E18 to match.")
    else:
        log.warning('E18 output not found; acceptance test skipped.')
    summary = res.groupby('variant').agg(n_significant=('significant', 'sum'), n_positive_tau=('tau', lambda x: int((x > 0).sum())), mean_abs_tau=('tau', lambda x: float(np.abs(x).mean()))).reset_index()
    summary['survivors'] = [', '.join(sorted(res[(res.variant == v) & res.significant].region)) or '(none)' for v in summary.variant]
    summary.to_csv(out_dir / 'summary_E19.csv', index=False)
    wide = res.pivot(index='region', columns='variant', values='tau')
    wide.corr().to_csv(out_dir / 'tau_correlation_E19.csv')
    log.info('Per-variant survivors:\n%s', summary.to_string(index=False))
    log.info('Cross-variant tau correlation:\n%s', wide.corr().round(3).to_string())
    ref = variants[0]
    for v in variants[1:]:
        flips = int((np.sign(wide[ref]) != np.sign(wide[v])).sum())
        log.info('sign flips %s vs %s: %d of %d regions', ref, v, flips, len(wide))
    log.info('E19 complete.')
if __name__ == '__main__':
    main()
