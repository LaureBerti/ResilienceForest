from __future__ import annotations
import logging
from itertools import combinations
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
log = logging.getLogger(__name__)
E9_DESIGNS: tuple[tuple[int, int], ...] = ((5, 4), (5, 3))

def _exact_u_null(n1: int, n2: int) -> np.ndarray:
    n = n1 + n2
    offset = n1 * (n1 + 1) / 2
    return np.array([sum(ranks) - offset for ranks in combinations(range(1, n + 1), n1)], dtype=float)

def _exact_two_sided_p(u_obs: float, null: np.ndarray, n1: int, n2: int) -> float:
    lo = min(u_obs, n1 * n2 - u_obs)
    hi = max(u_obs, n1 * n2 - u_obs)
    n_perm = null.size
    p = (np.sum(null <= lo) + np.sum(null >= hi)) / n_perm
    return float(min(1.0, p))

def _u_grid(n1: int, n2: int, alpha: float) -> pd.DataFrame:
    null = _exact_u_null(n1, n2)
    rows = []
    for u in range(0, n1 * n2 + 1):
        p = _exact_two_sided_p(float(u), null, n1, n2)
        rows.append({'n1': n1, 'n2': n2, 'n_permutations': int(null.size), 'U': u, 'p_two_sided_exact': p, 'effect_size_rbc': 1.0 - 2.0 * u / (n1 * n2), 'significant_at_alpha': bool(p < alpha)})
    return pd.DataFrame(rows)

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e9_dir = base_dir / cfg.output.dir / 'E9'
    out_dir = base_dir / cfg.output.dir / 'E15'
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(cfg.analysis.alpha)
    log.info('Exact Mann–Whitney U power floor | α = %.3f', alpha)
    grids = [_u_grid(n1, n2, alpha) for n1, n2 in E9_DESIGNS]
    grid = pd.concat(grids, ignore_index=True)
    out_path = out_dir / 'power_floor_E15.csv'
    grid.to_csv(out_path, index=False, float_format='%.10f')
    log.info('Saved: %s', out_path)
    md_lines: list[str] = ['# E15 — Exact power floor of the E9 Mann–Whitney U design', '', f'Exact enumeration of the null distribution of *U*; α = {alpha}.', '']
    for (n1, n2), g in zip(E9_DESIGNS, grids):
        n_perm = int(g['n_permutations'].iloc[0])
        p_min = float(g['p_two_sided_exact'].min())
        sig = g[g['significant_at_alpha']]
        log.info('Design n1=%d, n2=%d: %d rank assignments | minimum attainable two-sided p = %.10f | %d of %d attainable U values reach p < %.2f', n1, n2, n_perm, p_min, len(sig), len(g), alpha)
        if sig.empty:
            log.info('  → NO value of U can reach p < %.2f: the design is powerless at this α regardless of the data.', alpha)
        else:
            log.info('  → Significant cells (U, p, r):\n%s', sig[['U', 'p_two_sided_exact', 'effect_size_rbc']].to_string(index=False))
            log.info('  → Rank-biserial correlations attainable at significance: %s. These are the extreme tail of the r scale; |r| is fully determined by U, so a significant result CANNOT report a moderate effect size.', sorted(sig['effect_size_rbc'].round(4).unique().tolist()))
        md_lines += [f'## Design n1={n1}, n2={n2}', '', f'- Rank assignments under H0: C({n1 + n2},{n1}) = {n_perm}', f'- Minimum attainable two-sided p: **{p_min:.10f}** (= 2/{n_perm})', f'- Attainable U values reaching p < {alpha}: **{len(sig)}** of {len(g)}', '', '| U | exact two-sided p | rank-biserial r | p < α |', '|---|---|---|---|']
        for _, r in sig.iterrows():
            md_lines.append(f"| {int(r['U'])} | {r['p_two_sided_exact']:.6f} | {r['effect_size_rbc']:+.4f} | yes |")
        if sig.empty:
            md_lines.append('| — | — | — | none attainable |')
        md_lines.append('')
    e9_path = e9_dir / 'recent_trend_E9.csv'
    if e9_path.exists():
        e9 = pd.read_csv(e9_path)
        sig_e9 = e9[e9['significant']] if 'significant' in e9.columns else e9.iloc[0:0]
        log.info('Observed E9: %d of %d cells significant at α=%.2f; their p-values: %s; their |r|: %s', len(sig_e9), len(e9), alpha, sorted(sig_e9['p_value'].unique().tolist()), sorted(sig_e9['effect_size_rbc'].abs().unique().tolist()))
        md_lines += ['## Cross-check against observed E9 results', '', f'- Cells tested: {len(e9)}', f'- Significant at raw α={alpha}: {len(sig_e9)}', f"- Distinct p-values among them: {sorted(sig_e9['p_value'].unique().tolist())}", f"- Distinct |r| among them: {sorted(sig_e9['effect_size_rbc'].abs().unique().tolist())}", '']
    else:
        log.warning('E9 results not found for cross-check: %s', e9_path)
    md_path = out_dir / 'power_floor_E15.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    log.info('Saved: %s', md_path)
    log.info('E15 complete.')
if __name__ == '__main__':
    main()
