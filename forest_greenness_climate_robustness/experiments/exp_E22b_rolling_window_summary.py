from __future__ import annotations
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from statsmodels.stats.multitest import multipletests
BASE_DIR = Path(__file__).resolve().parent.parent

def _arm(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path / 'hamed_rao_mk_E13.csv').set_index(['region', 'indicator'])
    t['p_bh'] = multipletests(t['corrected_p'], method='fdr_bh')[1]
    return t

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    q = float(cfg.analysis.alpha)
    o = BASE_DIR / cfg.output.dir
    arms = {36: (o / 'E13', o / 'sens_E6' / 'stl_raw' / 'E13')}
    for w in (24, 48, 60):
        arms[w] = (o / 'sens_E6' / f'win{w}_climatology' / 'E13', o / 'sens_E6' / f'win{w}_stl' / 'E13')
    rows, summ, robust = ([], [], {})
    for w in sorted(arms):
        c, s = (_arm(arms[w][0]), _arm(arms[w][1]))
        both = (c.p_bh < q) & (s.p_bh < q) & (np.sign(c.mk_tau) == np.sign(s.mk_tau))
        robust[w] = set(both.index[both])
        for k in c.index:
            rows.append(dict(window=w, region=k[0], mode=k[1], tau_clim=c.loc[k, 'mk_tau'], p_bh_clim=c.loc[k, 'p_bh'], tau_stl=s.loc[k, 'mk_tau'], p_bh_stl=s.loc[k, 'p_bh'], robust=bool(both.loc[k])))
        summ.append(dict(window=w, n_bh_clim=int((c.p_bh < q).sum()), n_bh_stl=int((s.p_bh < q).sum()), n_robust=len(robust[w]), robust_pairs='; '.join((f'{r}x{m}' for r, m in sorted(robust[w])))))
    ref = robust[36]
    S = pd.DataFrame(summ)
    S['n_of_36_robust_retained'] = [len(ref & robust[w]) for w in S.window]
    R = pd.DataFrame(rows)
    sub = R[R.set_index(['region', 'mode']).index.isin(list(ref))]
    S['n_36_robust_same_sign_both'] = [int(((np.sign(g.tau_clim) == np.sign(sub[sub.window == 36].set_index(['region', 'mode']).loc[list(zip(g.region, g['mode'])), 'tau_clim'].to_numpy())) & (np.sign(g.tau_clim) == np.sign(g.tau_stl))).sum()) for _, g in sub.groupby('window')]
    (o / 'E22').mkdir(parents=True, exist_ok=True)
    R.to_csv(o / 'E22' / 'rolling_window_E22.csv', index=False)
    S.to_csv(o / 'E22' / 'rolling_window_summary_E22.csv', index=False)
    print(S.to_string(index=False))
if __name__ == '__main__':
    main()
