from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_E10_fdr_correction import _bh_fdr
from exp_E3_overlap import _climate_mp, _detect_overlaps, _find_climate_motif_starts, _load_climate, _load_kndvi_motifs, _load_kndvi_ts
log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent

def _e3_inputs(region: str, indicator: str, e2_dir: Path, climate_csv: Path, m: int, n_motifs: int, max_lag: int) -> dict[str, Any]:
    kndvi_ts, kndvi_dates = _load_kndvi_ts(region, e2_dir)
    kndvi_starts = [int(x) for x in _load_kndvi_motifs(region, e2_dir)['start_idx']]
    clim = _load_climate(climate_csv)
    clim = clim[(clim['date'] >= kndvi_dates[0]) & (clim['date'] <= kndvi_dates[-1])]
    clim = clim.reset_index(drop=True)
    cts = clim[indicator].to_numpy(dtype=np.float64)
    if np.any(np.isnan(cts)):
        cts = pd.Series(cts).interpolate(method='linear').to_numpy(dtype=np.float64)
    cmp_vals, _ = _climate_mp(cts, m)
    clim_starts = [int(x) for x in _find_climate_motif_starts(cmp_vals, n_motifs=n_motifs, m=m)]
    recs = _detect_overlaps(kndvi_starts, clim_starts, m, max_lag, kndvi_dates, pd.DatetimeIndex(clim['date']))
    return {'T_kndvi': len(kndvi_ts), 'T_clim': len(cts), 'n_valid_kndvi': len(kndvi_ts) - m + 1, 'n_valid_clim': len(cmp_vals), 'kndvi_starts': kndvi_starts, 'clim_starts': clim_starts, 'observed_pairs': len(recs), 'observed_kmotifs_with_partner': len({r['kndvi_motif_start'] for r in recs})}

def _series_length(region: str, e2_dir: Path) -> int:
    mp_path = e2_dir / f'matrix_profile_{region}.parquet'
    if mp_path.exists():
        df = pd.read_parquet(mp_path)
        return len(df) + 23
    kndvi_dir = BASE_DIR / 'data' / 'raw' / 'kndvi'
    parquet_path = kndvi_dir / f'kndvi_{region}_2000_2025.parquet'
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return int(df['date'].nunique())
    raise FileNotFoundError(f'Cannot determine series length for {region}: no E2 parquet and no raw kNDVI parquet.')

def _observed_count(region: str, indicator: str, e3_dir: Path) -> int:
    path = e3_dir / f'overlap_{region}_{indicator}.csv'
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if df.empty or 'overlap_flag' not in df.columns:
        return 0
    return int(df['overlap_flag'].sum())

def _draw_starts(rng: np.random.Generator, n_valid: int, k: int, excl: int | None) -> np.ndarray:
    while True:
        x = rng.choice(n_valid, size=k, replace=False)
        if excl is None or k < 2:
            return x
        d = np.abs(x[:, None] - x[None, :])
        if (d[np.triu_indices(k, 1)] > excl).all():
            return x

def _run_permutation(n_valid_kndvi: int, n_valid_clim: int, k_kndvi: int, k_clim: int, max_lag: int, excl: int | None, n_permutations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs = np.zeros(n_permutations, dtype=int)
    kmot = np.zeros(n_permutations, dtype=int)
    if n_valid_kndvi < k_kndvi or n_valid_clim < k_clim or k_kndvi == 0 or (k_clim == 0):
        return (pairs, kmot)
    for i in range(n_permutations):
        a = _draw_starts(rng, n_valid_kndvi, k_kndvi, excl)
        b = _draw_starts(rng, n_valid_clim, k_clim, excl)
        hit = np.abs(a[:, None] - b[None, :]) <= max_lag
        pairs[i] = int(hit.sum())
        kmot[i] = int(hit.any(axis=1).sum())
    return (pairs, kmot)

def _analyse_pair(region: str, indicator: str, inp: dict[str, Any], m: int, max_lag: int, excl: int | None, n_permutations: int, seed: int, observed: int, out_dir: Path) -> dict[str, Any]:
    k_kndvi = len(inp['kndvi_starts'])
    k_clim = len(inp['clim_starts'])
    null_counts, null_kmot = _run_permutation(inp['n_valid_kndvi'], inp['n_valid_clim'], k_kndvi, k_clim, max_lag, excl, n_permutations, seed)
    null_mean = float(np.mean(null_counts))
    null_std = float(np.std(null_counts))
    null_p95 = float(np.percentile(null_counts, 95))
    null_p99 = float(np.percentile(null_counts, 99))
    p_perm = float(np.mean(null_counts >= observed))
    p_perm_kmot = float(np.mean(null_kmot >= inp['observed_kmotifs_with_partner']))
    exceeds_p95 = bool(observed > null_p95)
    exceeds_p99 = bool(observed > null_p99)
    bins = np.arange(0, max(null_counts.max(), observed) + 2)
    hist, _ = np.histogram(null_counts, bins=bins)
    dist_df = pd.DataFrame({'count_value': bins[:-1], 'null_frequency': hist})
    dist_df['region'] = region
    dist_df['indicator'] = indicator
    dist_path = out_dir / f'null_model_{region}_{indicator}.csv'
    dist_df.to_csv(dist_path, index=False)
    log.info('  %s × %s | k_kndvi=%d k_clim=%d | observed=%d | null μ=%.2f σ=%.2f p95=%.1f | p_perm=%.4f', region, indicator, k_kndvi, k_clim, observed, null_mean, null_std, null_p95, p_perm)
    return {'region': region, 'indicator': indicator, 'T': inp['T_kndvi'], 'T_clim': inp['T_clim'], 'm': m, 'k_kndvi_motifs': k_kndvi, 'k_climate_motifs': k_clim, 'kndvi_starts': ' '.join(map(str, inp['kndvi_starts'])), 'climate_starts': ' '.join(map(str, inp['clim_starts'])), 'max_lag': max_lag, 'exclusion_zone': excl if excl is not None else -1, 'observed_count': observed, 'observed_kmotifs_with_partner': inp['observed_kmotifs_with_partner'], 'null_mean': round(null_mean, 3), 'null_std': round(null_std, 3), 'null_p95': round(null_p95, 2), 'null_p99': round(null_p99, 2), 'p_perm': p_perm, 'p_perm_kmotif_stat': p_perm_kmot, 'exceeds_null_p95': exceeds_p95, 'exceeds_null_p99': exceeds_p99, 'n_permutations': n_permutations, 'seed': seed}

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    log.info('Config:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    base_dir = Path(hydra.utils.get_original_cwd())
    e2_dir = base_dir / cfg.output.dir / 'E2'
    e3_dir = base_dir / cfg.output.dir / 'E3'
    out_dir = base_dir / cfg.output.dir / 'E11'
    out_dir.mkdir(parents=True, exist_ok=True)
    m: int = int(cfg.matrix_profile.subsequence_length)
    n_motifs_req: int = int(cfg.matrix_profile.n_motifs)
    max_lag: int = m
    seed: int = int(cfg.analysis.seed)
    n_permutations: int = int(OmegaConf.select(cfg, 'e11.n_permutations', default=10000))
    excl: int | None = m // 4 if bool(OmegaConf.select(cfg, 'e11.enforce_exclusion_zone', default=True)) else None
    q: float = float(OmegaConf.select(cfg, 'e11.fdr_q', default=0.05))
    climate_csv = base_dir / cfg.data.climate_dir / 'climate_indicators_2000_2025.csv'
    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions
    indicators = ['AO', 'ENSO', 'NAO', 'PNA']
    if cfg.run_mvp_only:
        log.info('MVP mode: africa × NAO only.')
        all_regions = ['africa']
        indicators = ['NAO']
    log.info('Parameters: m=%d, k = data-supplied per pair, max_lag=±%d months, exclusion zone=%s, n_permutations=%d, BH q=%.2f', m, max_lag, excl, n_permutations, q)
    summaries: list[dict] = []
    for region in all_regions:
        for indicator in indicators:
            observed = _observed_count(region, indicator, e3_dir)
            inp = _e3_inputs(region, indicator, e2_dir, climate_csv, m, n_motifs_req, max_lag)
            if inp['observed_pairs'] != observed:
                raise AssertionError(f"{region} × {indicator}: recomputed E3 overlap {inp['observed_pairs']} != stored E3 overlap {observed} — E3 outputs are stale.")
            row = _analyse_pair(region, indicator, inp, m, max_lag, excl, n_permutations, seed, observed, out_dir)
            summaries.append(row)
    if summaries:
        summary_df = pd.DataFrame(summaries)
        adj, rej = _bh_fdr(summary_df['p_perm'].to_numpy(dtype=float), q)
        summary_df['p_perm_bh'] = adj
        summary_df['significant_bh'] = rej
        summary_df['bh_family_size'] = len(summary_df)
        n_stat_diff = int((summary_df['observed_count'] != summary_df['observed_kmotifs_with_partner']).sum())
        log.info('Pairs where pair-count != motifs-with-partner on real data: %d', n_stat_diff)
        log.info('min p_perm = %.4f | n(p_perm < 0.05) = %d | BH survivors = %d of %d', summary_df['p_perm'].min(), int((summary_df['p_perm'] < 0.05).sum()), int(rej.sum()), len(summary_df))
        summary_path = out_dir / 'null_model_summary_E11.csv'
        summary_df.to_csv(summary_path, index=False)
        log.info('Saved summary: %s (%d rows)', summary_path, len(summary_df))
        pivot = summary_df.pivot_table(index='indicator', values='exceeds_null_p95', aggfunc='sum')
        log.info('Regions exceeding null p95 per indicator:\n%s', pivot.to_string())
    else:
        log.warning('No pairs processed — check E2 and E3 outputs.')
    log.info('E11 complete.')
if __name__ == '__main__':
    main()
