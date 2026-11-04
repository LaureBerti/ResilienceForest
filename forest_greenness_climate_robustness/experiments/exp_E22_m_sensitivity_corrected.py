from __future__ import annotations
import logging
import sys
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import stumpy
from omegaconf import DictConfig, OmegaConf
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_E2_matrix_profile import _extract_motifs
from exp_E3_overlap import _climate_mp, _detect_overlaps, _find_climate_motif_starts
from exp_E10_fdr_correction import _bh_fdr
from exp_E11_overlap_null_model import _run_permutation
from exp_E13_hamed_rao_mk import hamed_rao_mk
from exp_E19_deseason_sensitivity import _anomaly, _mpi
log = logging.getLogger(__name__)
MODES = ['AO', 'ENSO', 'NAO', 'PNA']
DEFAULTS = {'m_values': [12, 18, 24, 36, 48], 'anomalies': ['climatology', 'stl'], 'fixed_mpi_window': 24, 'n_permutations': 10000}

def _mpi_fixed(mp: np.ndarray, window: int) -> pd.Series:
    return pd.Series(mp).rolling(window=window, min_periods=window // 2).min()

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    e22 = OmegaConf.merge(OmegaConf.create(DEFAULTS), cfg.get('e22', {}))
    out_dir = BASE_DIR / cfg.output.dir / 'E22'
    out_dir.mkdir(parents=True, exist_ok=True)
    q = float(cfg.analysis.alpha)
    n_motifs = int(cfg.matrix_profile.n_motifs)
    seed = int(cfg.analysis.seed)
    kndvi_dir = BASE_DIR / cfg.data.kndvi_dir
    clim_all = pd.read_csv(BASE_DIR / cfg.data.climate_dir / 'climate_indicators_2000_2025.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    regions = sorted((p.stem.replace('kndvi_', '').replace('_2000_2025', '') for p in kndvi_dir.glob('kndvi_*_2000_2025.parquet')))
    if cfg.run_mvp_only:
        regions = ['africa', 'amazon']
    mpi_rows, ov_rows = ([], [])
    for region in regions:
        df = pd.read_parquet(kndvi_dir / f'kndvi_{region}_2000_2025.parquet', columns=['date', 'kndvi'])
        df['date'] = pd.to_datetime(df['date'])
        s = df.groupby('date')['kndvi'].mean().sort_index()
        dates = pd.DatetimeIndex(s.index)
        clim = clim_all[(clim_all['date'] >= dates[0]) & (clim_all['date'] <= dates[-1])].reset_index(drop=True)
        for anom in e22.anomalies:
            a = _anomaly(s, str(anom)).to_numpy(dtype=np.float64)
            for m in e22.m_values:
                m = int(m)
                mp_arr = stumpy.stump(a, m)
                mp = mp_arr[:, 0].astype(np.float64)
                for win_kind, series in (('fixed', _mpi_fixed(mp, int(e22.fixed_mpi_window))), ('m', _mpi(mp, m))):
                    v = series.dropna().to_numpy()
                    tau, p_hr, _ = hamed_rao_mk(v)
                    mpi_rows.append(dict(region=region, m=m, anomaly=anom, mpi_window=win_kind, n=len(v), tau=tau, p_hamedrao=p_hr))
                kstarts = [r['start_idx'] for r in _extract_motifs(mp_arr, a, m, n_motifs)]
                for mode in MODES:
                    cts = clim[mode].to_numpy(dtype=np.float64)
                    if np.any(np.isnan(cts)):
                        cts = pd.Series(cts).interpolate(method='linear').to_numpy(dtype=np.float64)
                    cmp_vals, _ = _climate_mp(cts, m)
                    cstarts = [int(x) for x in _find_climate_motif_starts(cmp_vals, n_motifs=n_motifs, m=m)]
                    recs = _detect_overlaps(kstarts, cstarts, m, m, dates, pd.DatetimeIndex(clim['date']))
                    obs = len(recs)
                    null, _ = _run_permutation(len(a) - m + 1, len(cmp_vals), len(kstarts), len(cstarts), m, m // 4, int(e22.n_permutations), seed)
                    ov_rows.append(dict(region=region, mode=mode, m=m, anomaly=anom, k_kndvi=len(kstarts), k_clim=len(cstarts), observed=obs, null_mean=float(null.mean()), p_perm=float(np.mean(null >= obs))))
        log.info('  %s done', region)
    M = pd.DataFrame(mpi_rows)
    M['p_bh'] = np.nan
    for _, g in M.groupby(['m', 'anomaly', 'mpi_window']):
        adj, _ = _bh_fdr(g['p_hamedrao'].to_numpy(dtype=float), q)
        M.loc[g.index, 'p_bh'] = adj
    M['significant'] = M['p_bh'] < q
    M.to_csv(out_dir / 'mpi_trends_E22.csv', index=False)
    O = pd.DataFrame(ov_rows)
    O['p_bh'] = np.nan
    for _, g in O.groupby(['m', 'anomaly']):
        adj, _ = _bh_fdr(g['p_perm'].to_numpy(dtype=float), q)
        O.loc[g.index, 'p_bh'] = adj
    O.to_csv(out_dir / 'overlap_E22.csv', index=False)
    rows = []
    for (m, win), g in M.groupby(['m', 'mpi_window']):
        w = g.pivot(index='region', columns='anomaly', values='tau')
        sig = g.pivot(index='region', columns='anomaly', values='significant')
        both = sig['climatology'] & sig['stl'] & (np.sign(w['climatology']) == np.sign(w['stl']))
        og = O[O.m == m]
        rows.append(dict(m=m, mpi_window=win, mpi_sig_clim=', '.join(sorted(sig.index[sig['climatology']])) or '(none)', mpi_sig_stl=', '.join(sorted(sig.index[sig['stl']])) or '(none)', n_mpi_sig_clim=int(sig['climatology'].sum()), n_mpi_sig_stl=int(sig['stl'].sum()), n_mpi_robust=int(both.sum()), mpi_robust=', '.join(sorted(both.index[both])) or '(none)', tau_corr_clim_stl=float(w['climatology'].corr(w['stl'])), n_overlap_p05_clim=int(((og.anomaly == 'climatology') & (og.p_perm < q)).sum()), n_overlap_p05_stl=int(((og.anomaly == 'stl') & (og.p_perm < q)).sum()), n_overlap_bh_clim=int(((og.anomaly == 'climatology') & (og.p_bh < q)).sum()), n_overlap_bh_stl=int(((og.anomaly == 'stl') & (og.p_bh < q)).sum()), min_p_overlap=float(og.p_perm.min())))
    S = pd.DataFrame(rows)
    for anom in e22.anomalies:
        w = M[(M.anomaly == anom) & (M.mpi_window == 'fixed')].pivot(index='region', columns='m', values='tau')
        w.corr().to_csv(out_dir / f'tau_corr_across_m_{anom}_E22.csv')
        log.info('tau correlation across m (%s, fixed window):\n%s', anom, w.corr().round(2).to_string())
    S.to_csv(out_dir / 'summary_E22.csv', index=False)
    print(S.to_string(index=False))
    acc = []
    e19p = BASE_DIR / cfg.output.dir / 'E19' / 'deseason_sensitivity_E19.csv'
    if e19p.exists() and 24 in [int(x) for x in e22.m_values]:
        e19 = pd.read_csv(e19p)
        for anom in e22.anomalies:
            a19 = e19[e19.variant == anom].set_index('region')
            for win in ('fixed', 'm'):
                a22 = M[(M.m == 24) & (M.anomaly == anom) & (M.mpi_window == win)].set_index('region')
                c = a22.index.intersection(a19.index)
                acc.append(dict(check=f'MPI m=24 {anom} {win} vs E19', max_abs_dtau=float((a22.loc[c, 'tau'] - a19.loc[c, 'tau']).abs().max()), max_abs_dp=float((a22.loc[c, 'p_hamedrao'] - a19.loc[c, 'p_hamedrao']).abs().max())))
    e11p = BASE_DIR / cfg.output.dir / 'E11' / 'null_model_summary_E11.csv'
    if e11p.exists() and 24 in [int(x) for x in e22.m_values]:
        e11 = pd.read_csv(e11p).rename(columns={'indicator': 'mode'}).set_index(['region', 'mode'])
        o = O[(O.m == 24) & (O.anomaly == 'climatology')].set_index(['region', 'mode'])
        c = o.index.intersection(e11.index)
        acc.append(dict(check='overlap m=24 climatology vs E11', n_mismatch_observed=int((o.loc[c, 'observed'] != e11.loc[c, 'observed_count']).sum()), n_mismatch_k=int(((o.loc[c, 'k_kndvi'] != e11.loc[c, 'k_kndvi_motifs']) | (o.loc[c, 'k_clim'] != e11.loc[c, 'k_climate_motifs'])).sum()), max_abs_dp=float((o.loc[c, 'p_perm'] - e11.loc[c, 'p_perm']).abs().max()), n_pairs=len(c)))
    A = pd.DataFrame(acc)
    A.to_csv(out_dir / 'acceptance_E22.csv', index=False)
    print(A.to_string(index=False))
if __name__ == '__main__':
    main()
