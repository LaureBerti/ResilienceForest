from __future__ import annotations
import logging
import re
import sys
import zipfile
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_E6_rolling_correlation import _kndvi_anomaly, _rolling_pearson
from exp_E13_hamed_rao_mk import hamed_rao_mk
from exp_E16_resilience_benchmark import REGION_ORDER
log = logging.getLogger(__name__)
MODES = ['AO', 'ENSO', 'NAO', 'PNA']
DEFAULTS = {'avhrr_dir': 'data/raw/avhrr_pku', 'qc_keep': [0, 1], 'qc_keep_modis': [990, 991], 'fill': 65535, 'res': 1.0 / 12.0, 'common_start': '2000-02-01', 'common_end': '2015-12-01', 'long_start': '1982-01-01', 'climate_long': 'data/raw/climate_1982/climate_indicators_1982_2024.csv', 'rolling_window': 36, 'm': 24, 'mpi_window': 24, 'min_valid_frac': 0.0, 'out_dir': 'E23'}

def _bh(p: np.ndarray) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return out

def _footprints(cfg: DictConfig, res: float) -> dict[str, pd.DataFrame]:
    fps = {}
    for region in REGION_ORDER:
        df = pd.read_parquet(BASE_DIR / cfg.data.kndvi_dir / f'kndvi_{region}_2000_2025.parquet', columns=['pixel_id', 'latitude', 'longitude']).drop_duplicates('pixel_id')
        row = np.floor((90.0 - df['latitude'].to_numpy()) / res).astype(int)
        col = np.floor((df['longitude'].to_numpy() + 180.0) / res).astype(int)
        fp = pd.DataFrame({'row': row, 'col': col}).value_counts().rename('w').reset_index()
        fps[region] = fp
        log.info('  footprint %s: %d MODIS pixels -> %d AVHRR cells', region, len(df), len(fp))
    return fps
SOURCES = {'AVHRR': ('PKU_GIMMS_NDVI_AVHRR_solely_*.zip', re.compile('PKU_GIMMS_NDVI\\(AVHRR\\)_V1\\.2_(\\d{4})(\\d{2})(0[12])\\.tif$'), 1982), 'MODIS_PKU': ('PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_*.zip', re.compile('PKU_GIMMS_NDVI_V1\\.2_(\\d{4})(\\d{2})(0[12])\\.tif$'), 2003)}

def _pku_series(cfg: DictConfig, e23: DictConfig, source: str) -> pd.DataFrame:
    import tifffile
    zglob, fname, first_year = SOURCES[source]
    out_dir = BASE_DIR / cfg.output.dir / str(e23.out_dir)
    fps = _footprints(cfg, float(e23.res))
    qc_keep = np.array(list(e23.qc_keep if source == 'AVHRR' else e23.qc_keep_modis), dtype=np.uint16)
    fill = int(e23.fill)
    cache = BASE_DIR / str(e23.avhrr_dir) / f'footprint_cells_{source}_E23.parquet'
    cells = pd.concat(fps.values())[['row', 'col']].drop_duplicates().sort_values(['row', 'col']).reset_index(drop=True)
    if cache.exists():
        raw = pd.read_parquet(cache)
        log.info('Read footprint cache %s (%d rows)', cache, len(raw))
    else:
        zips = sorted((BASE_DIR / str(e23.avhrr_dir)).glob(zglob))
        if not zips:
            raise FileNotFoundError(f'No {source} zips; run ./experiments/download_E23_avhrr_pku.sh')
        rr, cc = (cells['row'].to_numpy(), cells['col'].to_numpy())
        parts = []
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                names = sorted((n for n in zf.namelist() if fname.search(n) and int(fname.search(n).group(1)) >= first_year))
                for n in names:
                    y, mth, h = fname.search(n).groups()
                    with zf.open(n) as fh:
                        a = tifffile.imread(fh)
                    parts.append(pd.DataFrame({'row': rr.astype(np.int16), 'col': cc.astype(np.int16), 'ym': np.int32(int(y) * 100 + int(mth)), 'half': np.int8(int(h)), 'ndvi_raw': a[rr, cc, 0], 'qc': a[rr, cc, 1]}))
                log.info('  read %s (%d files)', z.name, len(names))
        raw = pd.concat(parts, ignore_index=True)
        raw.to_parquet(cache, index=False, compression='zstd')
        log.info('Wrote footprint cache %s (%d rows, %.0f MB)', cache, len(raw), cache.stat().st_size / 1000000.0)
    qc_counts = {int(k): int(v) for k, v in raw['qc'].value_counts().items()}
    log.info('%s QC value counts over footprint cells x half-months: %s', source, qc_counts)
    halves: dict[tuple[str, str], list[np.ndarray]] = {}
    cell_pos = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_frame(cells))
    reg_idx = {r: cell_pos.loc[list(zip(fp['row'], fp['col']))].to_numpy() for r, fp in fps.items()}
    for (ym, _h), g in raw.groupby(['ym', 'half'], sort=True):
        g = g.sort_values(['row', 'col'])
        if len(g) != len(cells):
            raise ValueError(f'{source} {ym}: {len(g)} cells, expected {len(cells)}')
        nd_all = g['ndvi_raw'].to_numpy().astype(float)
        qc_all = g['qc'].to_numpy()
        for region, idx in reg_idx.items():
            v, q = (nd_all[idx].copy(), qc_all[idx])
            v[(v == fill) | ~np.isin(q, qc_keep) | (v > 1000)] = np.nan
            halves.setdefault((region, f'{ym // 100}-{ym % 100:02d}'), []).append(v / 1000.0)
    rows = []
    for (region, ym), hs in halves.items():
        w = fps[region]['w'].to_numpy(dtype=float)
        stack = np.vstack(hs)
        with np.errstate(all='ignore'):
            ndvi = np.full(stack.shape[1], np.nan)
            okc = ~np.isnan(stack).all(0)
            ndvi[okc] = np.nanmax(stack[:, okc], axis=0)
        ok = ~np.isnan(ndvi)
        kn = np.tanh(ndvi ** 2)
        rows.append(dict(source=source, region=region, date=pd.Timestamp(ym + '-01'), n_halves=len(hs), valid_weight_frac=float(w[ok].sum() / w.sum()), ndvi=float(np.average(ndvi[ok], weights=w[ok])) if ok.any() else np.nan, kndvi=float(np.average(kn[ok], weights=w[ok])) if ok.any() else np.nan))
    S = pd.DataFrame(rows).sort_values(['region', 'date']).reset_index(drop=True)
    S.loc[S.valid_weight_frac < float(e23.min_valid_frac), ['ndvi', 'kndvi']] = np.nan
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(qc_counts).rename('count').to_csv(out_dir / f'qc_counts_{source}_E23.csv')
    return S

def _modis_series(cfg: DictConfig, region: str) -> pd.Series:
    df = pd.read_parquet(BASE_DIR / cfg.data.kndvi_dir / f'kndvi_{region}_2000_2025.parquet', columns=['date', 'kndvi'])
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date')['kndvi'].mean().sort_index()

def _coupling(anom: pd.Series, clim: pd.DataFrame, window: int) -> list[dict]:
    out = []
    for mode in MODES:
        rr = _rolling_pearson(anom, clim[mode].dropna(), window=window)
        r = rr['rolling_r'].to_numpy(dtype=float)
        r = r[~np.isnan(r)]
        tau, p, _ = hamed_rao_mk(r)
        out.append(dict(mode=mode, n_windows=len(r), tau=tau, p_hr=p, r_first=float(np.mean(r[:12])), r_last=float(np.mean(r[-12:]))))
    return out

def _mpi_trend(anom: pd.Series, m: int, w: int) -> dict:
    import stumpy
    a = anom.dropna().to_numpy(dtype=np.float64)
    mp = stumpy.stump(a, m)[:, 0].astype(np.float64)
    mpi = pd.Series(mp).rolling(window=w, min_periods=w // 2).min().dropna().to_numpy()
    tau, p, _ = hamed_rao_mk(mpi)
    return dict(n=len(mpi), tau=tau, p_hr=p)
ROBUST8 = [('africa', 'ENSO'), ('amazon', 'ENSO'), ('asia', 'ENSO'), ('congo', 'ENSO'), ('europe', 'ENSO'), ('russian_far_east', 'NAO'), ('south_america', 'ENSO'), ('southeast_asia', 'NAO')]
PERIODS = [('2000-2015', '2000-02-01', '2015-12-01', ['MODIS_paper', 'AVHRR'], 'v5'), ('2003-2015', '2003-01-01', '2015-12-01', ['MODIS_paper', 'MODIS_PKU', 'AVHRR'], 'v5'), ('2003-2022', '2003-01-01', '2022-12-01', ['MODIS_paper', 'MODIS_PKU'], 'v5'), ('1982-2015', '1982-01-01', '2015-12-01', ['AVHRR'], 'v6')]

@hydra.main(config_path='../conf', config_name='resilience_forest', version_base='1.3')
def main(cfg: DictConfig) -> None:
    e23 = OmegaConf.merge(OmegaConf.create(DEFAULTS), cfg.get('e23', {}))
    out = BASE_DIR / cfg.output.dir / str(e23.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = float(cfg.analysis.alpha)
    win, m, mw = (int(e23.rolling_window), int(e23.m), int(e23.mpi_window))
    S = pd.concat([_pku_series(cfg, e23, src) for src in SOURCES], ignore_index=True)
    S.to_csv(out / 'pku_regional_monthly_E23.csv', index=False)
    climf = {'v5': pd.read_csv(BASE_DIR / cfg.data.climate_dir / 'climate_indicators_2000_2025.csv', parse_dates=['date']).set_index('date')[MODES], 'v6': pd.read_csv(BASE_DIR / str(e23.climate_long), parse_dates=['date']).set_index('date')[MODES]}
    agree, coup, mpis, oni = ([], [], [], [])
    for region in REGION_ORDER:
        ser = {'MODIS_paper': _modis_series(cfg, region)}
        for src in SOURCES:
            ser[src] = S[(S.source == src) & (S.region == region)].set_index('date')['kndvi'].dropna()
        for label, p0, p1, names, cv in PERIODS:
            clim = climf[cv]
            for anom_kind in ('climatology', 'stl'):
                an = {n: _kndvi_anomaly(ser[n].loc[p0:p1].dropna(), anom_kind) for n in names}
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        j = pd.concat([an[a].rename('a'), an[b].rename('b')], axis=1).dropna()
                        raw = pd.concat([ser[a].loc[p0:p1], ser[b].loc[p0:p1]], axis=1).dropna()
                        jy = j.groupby(j.index.year).mean()
                        agree.append(dict(period=label, region=region, anomaly=anom_kind, a=a, b=b, n_months=len(j), r_anomaly=float(j.a.corr(j.b)), r_annual=float(jy.a.corr(jy.b)), r_raw=float(raw.iloc[:, 0].corr(raw.iloc[:, 1]))))
                for n in names:
                    for rec in _coupling(an[n], clim, win):
                        coup.append(dict(period=label, region=region, anomaly=anom_kind, series=n, **rec))
                    mpis.append(dict(period=label, region=region, anomaly=anom_kind, series=n, **_mpi_trend(an[n], m, mw)))
        for anom_kind in ('climatology', 'stl'):
            am_full = _kndvi_anomaly(ser['MODIS_paper'].loc[:'2024-12-01'], anom_kind)
            for ver, clim in climf.items():
                rr = _rolling_pearson(am_full, clim['ENSO'].loc['2000-01-01':'2024-12-01'], window=win)
                tau, p, _ = hamed_rao_mk(rr['rolling_r'].dropna().to_numpy())
                oni.append(dict(region=region, anomaly=anom_kind, oni=ver, tau=tau, p_hr=p))
        log.info('  %s done', region)
    G = pd.DataFrame(agree)
    G.to_csv(out / 'agreement_E23.csv', index=False)
    fam = ['period', 'anomaly', 'series']
    C = pd.DataFrame(coup)
    C['p_bh'] = C.groupby(fam)['p_hr'].transform(lambda s_: _bh(s_.to_numpy()))
    C.to_csv(out / 'coupling_E23.csv', index=False)
    P = pd.DataFrame(mpis)
    P['p_bh'] = P.groupby(fam)['p_hr'].transform(lambda s_: _bh(s_.to_numpy()))
    P.to_csv(out / 'mpi_E23.csv', index=False)
    O = pd.DataFrame(oni).pivot_table(index=['region', 'anomaly'], columns='oni', values=['tau', 'p_hr'])
    O.columns = [f'{a}_{b}' for a, b in O.columns]
    O = O.reset_index()
    O.to_csv(out / 'oni_version_check_E23.csv', index=False)
    lines = ['Median agreement (Pearson r) by period, pair, anomaly:', G.groupby(['period', 'a', 'b', 'anomaly'])[['r_anomaly', 'r_annual', 'r_raw']].median().round(2).to_string(), '']
    pair_rows = []
    for label, _, _, names, _ in PERIODS:
        for anom_kind in ('climatology', 'stl'):
            cc = C[(C.period == label) & (C.anomaly == anom_kind)]
            for i, a in enumerate(names):
                ca = cc[cc.series == a].set_index(['region', 'mode'])
                row = dict(period=label, anomaly=anom_kind, series=a, n_bh=int((ca.p_bh < q).sum()))
                for b in names[i + 1:]:
                    cb = cc[cc.series == b].set_index(['region', 'mode'])
                    same = np.sign(ca.tau) == np.sign(cb.tau)
                    pair_rows.append(dict(period=label, anomaly=anom_kind, a=a, b=b, tau_corr=float(ca.tau.corr(cb.tau)), sign_agree=int(same.sum()), n_bh_a=int((ca.p_bh < q).sum()), n_bh_b=int((cb.p_bh < q).sum()), n_bh_both_same_sign=int(((ca.p_bh < q) & (cb.p_bh < q) & same).sum()), robust8_sign_agree=int(same.loc[ROBUST8].sum())))
    PR = pd.DataFrame(pair_rows)
    PR.to_csv(out / 'coupling_pairs_E23.csv', index=False)
    lines += ['Coupling-trend agreement between series (56 region x mode pairs):', PR.round(2).to_string(index=False), '']
    R8 = C[C.set_index(['region', 'mode']).index.isin(ROBUST8)].pivot_table(index=['region', 'mode'], columns=['period', 'series', 'anomaly'], values='tau')
    R8.to_csv(out / 'robust8_tau_E23.csv')
    R8b = C[C.set_index(['region', 'mode']).index.isin(ROBUST8)].pivot_table(index=['region', 'mode'], columns=['period', 'series', 'anomaly'], values='p_bh')
    R8b.to_csv(out / 'robust8_pbh_E23.csv')
    lines += ['The 8 robust MODIS pairs: tau by period/series/anomaly:', R8.round(2).to_string(), '']
    for label, _, _, names, _ in PERIODS:
        for anom_kind in ('climatology', 'stl'):
            pp = P[(P.period == label) & (P.anomaly == anom_kind)].pivot_table(index='region', columns='series', values='tau')
            sig = P[(P.period == label) & (P.anomaly == anom_kind) & (P.p_bh < q)]
            cors = '; '.join((f'{a}~{b} {pp[a].corr(pp[b]):.2f}' for i, a in enumerate(names) for b in names[i + 1:]))
            sigs = '; '.join((f"{n}: {', '.join(sorted(g.region))}" for n, g in sig.groupby('series'))) or 'none'
            lines.append(f"MPI [{label} {anom_kind}] tau corr {cors or '-'} | BH-sig {sigs}")
    lines.append(f'ONI v5 vs v6 (MODIS 2000-2024 rolling-ENSO trends): max |dtau| = {(O.tau_v5 - O.tau_v6).abs().max():.3f}, tau corr {O.tau_v5.corr(O.tau_v6):.3f}, same sign {int((np.sign(O.tau_v5) == np.sign(O.tau_v6)).sum())}/{len(O)}')
    txt = '\n'.join(lines)
    (out / 'summary_E23.txt').write_text(txt + '\n')
    print(txt)
if __name__ == '__main__':
    main()
