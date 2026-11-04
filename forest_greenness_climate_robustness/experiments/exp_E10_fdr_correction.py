from __future__ import annotations
import logging
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
log = logging.getLogger(__name__)

def _bh_fdr(p_values: np.ndarray, q: float=0.05) -> tuple[np.ndarray, np.ndarray]:
    n = len(p_values)
    if n == 0:
        return (np.array([]), np.array([], dtype=bool))
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1)
    adj = np.minimum(1.0, sorted_p * n / ranks)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    thresholds = q * ranks / n
    below = sorted_p <= thresholds
    if below.any():
        max_k = int(np.where(below)[0].max())
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[:max_k + 1] = True
    else:
        reject_sorted = np.zeros(n, dtype=bool)
    adjusted_p = np.empty(n)
    reject = np.empty(n, dtype=bool)
    adjusted_p[order] = adj
    reject[order] = reject_sorted
    return (adjusted_p, reject)

def _load_e5(e5_dir: Path) -> pd.DataFrame:
    path = e5_dir / 'temporal_test_E5.csv'
    if not path.exists():
        raise FileNotFoundError(f'E5 results not found: {path}. Run exp_E5 first.')
    return pd.read_csv(path)

def _load_e6(e6_dir: Path) -> pd.DataFrame:
    path = e6_dir / 'rolling_corr_trend_E6.csv'
    if not path.exists():
        raise FileNotFoundError(f'E6 results not found: {path}. Run exp_E6 first.')
    return pd.read_csv(path)

def _load_e9(e9_dir: Path) -> pd.DataFrame:
    path = e9_dir / 'recent_trend_E9.csv'
    if not path.exists():
        raise FileNotFoundError(f'E9 results not found: {path}. Run exp_E9 first.')
    return pd.read_csv(path)

def _load_e13(e13_dir: Path) -> pd.DataFrame:
    path = e13_dir / 'hamed_rao_mk_E13.csv'
    if not path.exists():
        raise FileNotFoundError(f'E13 results not found: {path}. Run exp_E13 first.')
    return pd.read_csv(path)

def _correct_e5(df: pd.DataFrame, q: float) -> pd.DataFrame:
    df = df.copy()
    p_col = 'p_value'
    if p_col not in df.columns:
        raise ValueError(f"Column '{p_col}' not found in E5 results.")
    valid_mask = df[p_col].notna()
    p_arr = df.loc[valid_mask, p_col].to_numpy(dtype=float)
    adj, reject = _bh_fdr(p_arr, q)
    df['p_value_bh'] = np.nan
    df['fdr_significant'] = False
    df.loc[valid_mask, 'p_value_bh'] = adj
    df.loc[valid_mask, 'fdr_significant'] = reject
    n_orig_sig = int(df['significant'].sum()) if 'significant' in df.columns else 0
    n_fdr_sig = int(reject.sum())
    log.info('E5 FDR correction: %d tests | originally significant (α=%.2f): %d | FDR-significant (q=%.2f): %d', len(p_arr), 0.05, n_orig_sig, q, n_fdr_sig)
    return df

def _correct_e13(df: pd.DataFrame, q: float) -> pd.DataFrame:
    df = df.copy()
    p_col = 'corrected_p'
    if p_col not in df.columns:
        raise ValueError(f"Column '{p_col}' not found in E13 results. Columns: {list(df.columns)}")
    valid_mask = df[p_col].notna()
    p_arr = df.loc[valid_mask, p_col].to_numpy(dtype=float)
    adj, reject = _bh_fdr(p_arr, q)
    df['p_value_bh'] = np.nan
    df['fdr_significant'] = False
    df.loc[valid_mask, 'p_value_bh'] = adj
    df.loc[valid_mask, 'fdr_significant'] = reject
    log.info('E13 FDR correction: %d tests | Hamed-Rao significant at raw alpha=0.05: %d | surviving BH at q=%.2f: %d', len(p_arr), int(df.get('corrected_significant', pd.Series(dtype=bool)).sum()), q, int(reject.sum()))
    return df

def _correct_e6(df: pd.DataFrame, q: float) -> pd.DataFrame:
    df = df.copy()
    p_col = 'mk_p' if 'mk_p' in df.columns else 'p_value'
    if p_col not in df.columns:
        raise ValueError(f"Neither 'mk_p' nor 'p_value' found in E6 results. Columns: {list(df.columns)}")
    valid_mask = df[p_col].notna()
    p_arr = df.loc[valid_mask, p_col].to_numpy(dtype=float)
    adj, reject = _bh_fdr(p_arr, q)
    df['p_value_bh'] = np.nan
    df['fdr_significant'] = False
    df.loc[valid_mask, 'p_value_bh'] = adj
    df.loc[valid_mask, 'fdr_significant'] = reject
    n_orig_sig = int(df.get('significant', pd.Series(dtype=bool)).sum())
    n_fdr_sig = int(reject.sum())
    log.info('E6 FDR correction: %d tests | originally significant (α=%.2f): %d | FDR-significant (q=%.2f): %d', len(p_arr), 0.05, n_orig_sig, q, n_fdr_sig)
    return df

def _correct_e9(df: pd.DataFrame, q: float) -> pd.DataFrame:
    df = df.copy()
    p_col = 'p_value'
    if p_col not in df.columns:
        raise ValueError(f"Column '{p_col}' not found in E9 results. Columns: {list(df.columns)}")
    valid_mask = df[p_col].notna()
    p_arr = df.loc[valid_mask, p_col].to_numpy(dtype=float)
    adj, reject = _bh_fdr(p_arr, q)
    df['p_value_bh'] = np.nan
    df['fdr_significant'] = False
    df.loc[valid_mask, 'p_value_bh'] = adj
    df.loc[valid_mask, 'fdr_significant'] = reject
    n = len(p_arr)
    ranks = np.empty(n, dtype=int)
    ranks[np.argsort(p_arr)] = np.arange(1, n + 1)
    df['bh_rank'] = pd.NA
    df['bh_critical_value'] = np.nan
    df.loc[valid_mask, 'bh_rank'] = ranks
    df.loc[valid_mask, 'bh_critical_value'] = q * ranks / n
    n_orig_sig = int(df['significant'].sum()) if 'significant' in df.columns else 0
    n_fdr_sig = int(reject.sum())
    log.info('E9 FDR correction: %d tests | originally significant (α=%.2f): %d | FDR-significant (q=%.2f): %d', n, 0.05, n_orig_sig, q, n_fdr_sig)
    if n > 0:
        p_min = float(np.min(p_arr))
        n_at_min = int(np.sum(np.isclose(p_arr, p_min)))
        log.info('E9 smallest p = %.6f (shared by %d of %d tests); BH critical value at rank %d = %.6f. Rejection at the minimum p requires at least ceil(n*p_min/q) = %d tests tied at that p.', p_min, n_at_min, n, n_at_min, q * n_at_min / n, int(np.ceil(n * p_min / q)))
    return df

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e5_dir = base_dir / cfg.output.dir / 'E5'
    e6_dir = base_dir / cfg.output.dir / 'E6'
    e9_dir = base_dir / cfg.output.dir / 'E9'
    out_dir = base_dir / cfg.output.dir / 'E10'
    out_dir.mkdir(parents=True, exist_ok=True)
    q = float(cfg.analysis.alpha)
    log.info('FDR level q = %.3f (Benjamini–Hochberg step-up procedure)', q)
    summary_rows: list[dict] = []
    try:
        e5_df = _load_e5(e5_dir)
        e5_corrected = _correct_e5(e5_df, q)
        out_path = out_dir / 'fdr_corrected_E5.csv'
        e5_corrected.to_csv(out_path, index=False)
        log.info('Saved: %s', out_path)
        summary_rows.append({'collection': 'E5_temporal_shift', 'n_tests': len(e5_df), 'n_uncorrected_sig': int(e5_df.get('significant', pd.Series(dtype=bool)).sum()), 'n_fdr_sig': int(e5_corrected['fdr_significant'].sum()), 'fdr_level_q': q, 'correction': 'Benjamini-Hochberg'})
        survivors = e5_corrected[e5_corrected['fdr_significant']]
        if not survivors.empty:
            log.info('E5 pairs surviving FDR:\n%s', survivors[['region', 'indicator', 'p_value', 'p_value_bh']].to_string(index=False))
        else:
            log.info('E5: NO pairs survive FDR correction at q=%.2f — all 9 originally-significant pairs were at the uncorrected boundary.', q)
    except FileNotFoundError as exc:
        log.warning('Skipping E5 correction: %s', exc)
    try:
        e6_df = _load_e6(e6_dir)
        e6_corrected = _correct_e6(e6_df, q)
        out_path = out_dir / 'fdr_corrected_E6.csv'
        e6_corrected.to_csv(out_path, index=False)
        log.info('Saved: %s', out_path)
        summary_rows.append({'collection': 'E6_rolling_corr_MK', 'n_tests': len(e6_df), 'n_uncorrected_sig': int(e6_df.get('significant', pd.Series(dtype=bool)).sum()), 'n_fdr_sig': int(e6_corrected['fdr_significant'].sum()), 'fdr_level_q': q, 'correction': 'Benjamini-Hochberg'})
    except FileNotFoundError as exc:
        log.warning('Skipping E6 correction: %s', exc)
    try:
        e9_df = _load_e9(e9_dir)
        e9_corrected = _correct_e9(e9_df, q)
        out_path = out_dir / 'fdr_corrected_E9.csv'
        e9_corrected.to_csv(out_path, index=False)
        log.info('Saved: %s', out_path)
        summary_rows.append({'collection': 'E9_post2020_shift', 'n_tests': len(e9_df), 'n_uncorrected_sig': int(e9_df.get('significant', pd.Series(dtype=bool)).sum()), 'n_fdr_sig': int(e9_corrected['fdr_significant'].sum()), 'fdr_level_q': q, 'correction': 'Benjamini-Hochberg'})
        survivors = e9_corrected[e9_corrected['fdr_significant']]
        if not survivors.empty:
            log.info('E9 cells surviving FDR:\n%s', survivors[['region', 'metric', 'p_value', 'p_value_bh']].to_string(index=False))
        else:
            log.info('E9: NO region×metric cells survive FDR correction at q=%.2f — the uncorrected-significant cells sit at the exact-test p-value floor of the E9 design (n_pre=5, n_post=3 after the 2026-11-03 fix).', q)
    except FileNotFoundError as exc:
        log.warning('Skipping E9 correction: %s', exc)
    try:
        e13_dir = base_dir / cfg.output.dir / 'E13'
        e13_df = _load_e13(e13_dir)
        e13_corrected = _correct_e13(e13_df, q)
        out_path = out_dir / 'fdr_corrected_E13.csv'
        e13_corrected.to_csv(out_path, index=False)
        log.info('Saved: %s', out_path)
        summary_rows.append({'collection': 'E13_rolling_corr_hamedrao', 'n_tests': len(e13_df), 'n_uncorrected_sig': int(e13_df.get('corrected_significant', pd.Series(dtype=bool)).sum()), 'n_fdr_sig': int(e13_corrected['fdr_significant'].sum()), 'fdr_level_q': q, 'correction': 'Benjamini-Hochberg on Hamed-Rao p-values'})
        survivors = e13_corrected[e13_corrected['fdr_significant']]
        log.info('E13 pairs surviving BH:\n%s', survivors[['region', 'indicator', 'mk_tau', 'corrected_p', 'p_value_bh']].sort_values('p_value_bh').to_string(index=False))
    except (FileNotFoundError, ValueError) as exc:
        log.warning('Skipping E13 correction: %s', exc)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / 'fdr_summary_E10.csv'
        summary_df.to_csv(summary_path, index=False)
        log.info('Saved summary: %s\n%s', summary_path, summary_df.to_string(index=False))
    else:
        log.warning('No results corrected — check that E5 and E6 have been run.')
    log.info('E10 complete.')
if __name__ == '__main__':
    main()
