from __future__ import annotations
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_E10_fdr_correction import _bh_fdr
from exp_E13_hamed_rao_mk import hamed_rao_mk
log = logging.getLogger(__name__)

@dataclass
class E18Config:
    q: float = 0.05
    alpha: float = 0.05
    e8_dir: str = 'outputs/E8'
    out_dir: str = 'outputs/E18'

def _mk_variance(s: float, z: float) -> float:
    if z == 0 or not np.isfinite(z):
        return float('nan')
    numerator = s - np.sign(s)
    return float((numerator / z) ** 2)

def _effective_n(series: np.ndarray) -> tuple[float, float]:
    import pymannkendall as mk
    n = len(series)
    orig = mk.original_test(series)
    hr = mk.hamed_rao_modification_test(series)
    var_orig = _mk_variance(float(orig.s), float(orig.z))
    var_hr = _mk_variance(float(hr.s), float(hr.z))
    if not np.isfinite(var_orig) or var_orig <= 0 or (not np.isfinite(var_hr)):
        return (float('nan'), float('nan'))
    inflation = var_hr / var_orig
    n_eff = n / inflation if inflation > 0 else float('nan')
    return (float(inflation), float(n_eff))

def _correct_family(series_by_region: dict[str, np.ndarray], q: float, alpha: float, family_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for region, series in series_by_region.items():
        n = len(series)
        tau, p_hr, p_raw = hamed_rao_mk(series)
        inflation, n_eff = _effective_n(series)
        rows.append({'family': family_label, 'region': region, 'tau': round(float(tau), 4), 'n': int(n), 'n_eff': round(n_eff, 1) if np.isfinite(n_eff) else np.nan, 'var_inflation': round(inflation, 3) if np.isfinite(inflation) else np.nan, 'p_raw': float(p_raw), 'p_hamedrao': float(p_hr), 'sig_raw': bool(p_raw < alpha), 'sig_hamedrao': bool(p_hr < alpha)})
    df = pd.DataFrame(rows)
    adj_raw, rej_raw = _bh_fdr(df['p_raw'].to_numpy(dtype=float), q)
    adj_hr, rej_hr = _bh_fdr(df['p_hamedrao'].to_numpy(dtype=float), q)
    df['p_bh_raw'] = adj_raw
    df['sig_bh_raw'] = rej_raw
    df['p_bh_hamedrao'] = adj_hr
    df['sig_bh_hamedrao'] = rej_hr
    df['survives_both'] = df['sig_bh_hamedrao']
    return df

def _load_series(e8_dir: Path, region: str, column: str) -> np.ndarray | None:
    path = e8_dir / f'mpi_{region}.csv'
    if not path.exists():
        log.warning('Missing per-region E8 series: %s — skipping %s.', path, region)
        return None
    df = pd.read_csv(path)
    if column not in df.columns:
        log.warning('Column %s absent from %s — skipping %s.', column, path, region)
        return None
    arr = df[column].to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        log.warning('Too few observations (%d) for %s / %s — skipping.', len(arr), region, column)
        return None
    return arr

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e18_cfg = OmegaConf.merge(OmegaConf.structured(E18Config()), cfg.get('e18', {}))
    q = float(e18_cfg.q)
    alpha = float(e18_cfg.alpha)
    e8_dir = base_dir / str(e18_cfg.e8_dir)
    out_dir = base_dir / str(e18_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trend_path = e8_dir / 'mpi_trend_E8.csv'
    if not trend_path.exists():
        raise FileNotFoundError(f'E8 trend summary not found: {trend_path}. Run exp_E8 first.')
    e8 = pd.read_csv(trend_path)
    regions = [str(r) for r in e8['region']]
    log.info('Correcting %d region trend tests read from %s', len(regions), trend_path)
    families = {'MPI_rolling_min': 'mpi_rolling_min', 'mean_MP_rolling_mean': 'mp_rolling_mean'}
    summary_rows: list[dict] = []
    out_names = {'MPI_rolling_min': 'mpi_trend_corrected_E18.csv', 'mean_MP_rolling_mean': 'mean_mp_trend_corrected_E18.csv'}
    for family_label, column in families.items():
        series_by_region: dict[str, np.ndarray] = {}
        for region in regions:
            arr = _load_series(e8_dir, region, column)
            if arr is not None:
                series_by_region[region] = arr
        if not series_by_region:
            log.warning('No usable series for family %s — skipping.', family_label)
            continue
        df = _correct_family(series_by_region, q=q, alpha=alpha, family_label=family_label)
        out_path = out_dir / out_names[family_label]
        df.to_csv(out_path, index=False)
        log.info('Saved %s (%d rows)', out_path, len(df))
        summary_rows.append({'family': family_label, 'n_tests': len(df), 'n_sig_uncorrected': int(df['sig_raw'].sum()), 'n_sig_hamedrao_only': int(df['sig_hamedrao'].sum()), 'n_sig_bh_on_raw_only': int(df['sig_bh_raw'].sum()), 'n_sig_both': int(df['sig_bh_hamedrao'].sum()), 'alpha': alpha, 'fdr_q': q, 'survivors': ';'.join(df.loc[df['survives_both'], 'region'].tolist()) or 'none'})
        print('\n' + '=' * 78)
        print(f'E18 — family: {family_label}  (n_tests={len(df)}, alpha={alpha}, q={q})')
        print('=' * 78)
        cols = ['region', 'tau', 'n', 'n_eff', 'p_raw', 'p_hamedrao', 'p_bh_raw', 'p_bh_hamedrao', 'survives_both']
        print(df[cols].to_string(index=False))
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = out_dir / 'correction_summary_E18.csv'
        summary.to_csv(summary_path, index=False)
        print('\n' + summary.to_string(index=False))
        log.info('Saved summary: %s', summary_path)
    log.info('E18 complete.')
if __name__ == '__main__':
    main()
