"""
exp_E12_m_sensitivity.py — Subsequence length (m) sensitivity analysis.

Tests whether the indicator coupling hierarchy (NAO > PNA > AO ≈ ENSO) and
the significant region-counts are stable across m ∈ {12, 24, 36, 48} months.

For each (m, region, indicator):
  1. Load raw kNDVI time series.
  2. Deseasonalise (centred 12-month rolling mean subtraction).
  3. Compute self-join matrix profile (STUMPY, window=m).
  4. Extract top-k kNDVI motif starts (exclusion zone = m//4).
  5. Load and align climate indicator series; compute climate MP.
  6. Extract top-k climate motif starts.
  7. Count overlaps within ±m months (same criterion as E3).
  8. Flag whether any overlap was found (binary coupling signal).

Outputs:
  outputs/E12/m_sensitivity_E12.csv
      columns: m, region, indicator, n_overlaps, coupled (bool),
               overlap_fraction, mean_mp, motif_distance_best

Usage (Hydra):
    python experiments/exp_E12_m_sensitivity.py
    python experiments/exp_E12_m_sensitivity.py run_mvp_only=true
    python experiments/exp_E12_m_sensitivity.py analysis.m_values=[12,24,36,48]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import stumpy
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data loaders (same convention as E3/E6)
# ---------------------------------------------------------------------------

def _load_kndvi_ts(region: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    kndvi_dir = BASE_DIR / "data" / "raw" / "kndvi"
    path = kndvi_dir / f"kndvi_{region}_2000_2025.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw kNDVI parquet not found: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    ts = df.groupby("date")["kndvi"].mean().sort_index()
    return ts.to_numpy(dtype=np.float64), pd.DatetimeIndex(ts.index)


def _load_climate(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "climate" / "climate_indicators_2000_2025.csv"
    if not path.exists():
        raise FileNotFoundError(f"Climate CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def _deseasonalise(arr: np.ndarray) -> np.ndarray:
    """
    Centred 12-month rolling mean subtraction (same as exp_E6_rolling_correlation).
    NaN edges are filled by nearest valid value.
    """
    s = pd.Series(arr)
    seasonal = s.rolling(window=12, center=True, min_periods=6).mean()
    residual = s - seasonal
    # Forward/backward fill boundary NaNs
    residual = residual.ffill().bfill()
    return residual.to_numpy(dtype=np.float64)


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr)
    return s.interpolate(method="linear").ffill().bfill().to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Matrix profile and motif extraction
# ---------------------------------------------------------------------------

def _top_k_motif_starts(mp_values: np.ndarray, k: int, m: int) -> list[int]:
    """Return start indices of top-k motifs using an exclusion zone = m//4."""
    excl = max(m // 4, 1)
    vals = mp_values.copy()
    starts: list[int] = []
    for _ in range(k):
        idx = int(np.argmin(vals))
        starts.append(idx)
        lo = max(0, idx - excl)
        hi = min(len(vals), idx + excl + 1)
        vals[lo:hi] = np.inf
    return starts


def _count_overlaps(
    kndvi_starts: list[int],
    clim_starts: list[int],
    max_lag: int,
) -> int:
    """Count kNDVI motifs that have at least one climate motif within ±max_lag."""
    clim_arr = np.array(clim_starts)
    count = 0
    for ks in kndvi_starts:
        if np.any(np.abs(clim_arr - ks) <= max_lag):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per (m, region, indicator) analysis
# ---------------------------------------------------------------------------

def _analyse(
    m: int,
    region: str,
    indicator: str,
    k: int,
    raw_dir: Path,
) -> dict[str, Any]:
    max_lag = m  # same convention as E3

    # kNDVI
    kndvi_raw, kndvi_dates = _load_kndvi_ts(region)
    kndvi_deseason = _deseasonalise(kndvi_raw)

    # Climate
    clim_df = _load_climate(raw_dir)
    clim_df = clim_df[
        (clim_df["date"] >= kndvi_dates[0]) & (clim_df["date"] <= kndvi_dates[-1])
    ].reset_index(drop=True)
    clim_ts = _interpolate_nans(clim_df[indicator].to_numpy(dtype=np.float64))
    clim_deseason = _deseasonalise(clim_ts)

    T = len(kndvi_deseason)

    if T < 2 * m:
        log.warning("  %s × %s × m=%d: T=%d < 2m — skipping.", region, indicator, m, T)
        return {
            "m": m, "region": region, "indicator": indicator,
            "T": T, "n_overlaps": None, "coupled": None,
            "overlap_fraction": None, "mean_mp": None, "motif_distance_best": None,
        }

    # kNDVI matrix profile
    kndvi_mp = stumpy.stump(kndvi_deseason, m)
    kndvi_mp_vals = kndvi_mp[:, 0].astype(np.float64)
    kndvi_starts = _top_k_motif_starts(kndvi_mp_vals, k, m)
    mean_mp = float(np.nanmean(kndvi_mp_vals))
    best_motif_dist = float(np.nanmin(kndvi_mp_vals))

    # Climate matrix profile
    if len(clim_deseason) < 2 * m:
        log.warning("  Climate series too short for m=%d at %s × %s.", m, region, indicator)
        return {
            "m": m, "region": region, "indicator": indicator,
            "T": T, "n_overlaps": 0, "coupled": False,
            "overlap_fraction": 0.0, "mean_mp": mean_mp,
            "motif_distance_best": best_motif_dist,
        }

    clim_mp = stumpy.stump(clim_deseason, m)
    clim_mp_vals = clim_mp[:, 0].astype(np.float64)
    clim_starts = _top_k_motif_starts(clim_mp_vals, k, m)

    # Overlap count
    n_overlaps = _count_overlaps(kndvi_starts, clim_starts, max_lag)
    overlap_fraction = round(n_overlaps / max(k, 1), 3)
    coupled = n_overlaps > 0

    log.info(
        "  m=%2d | %s × %-5s | overlaps=%d/%d | coupled=%s | mean_mp=%.3f",
        m, region, indicator, n_overlaps, k, coupled, mean_mp,
    )

    return {
        "m": m, "region": region, "indicator": indicator,
        "T": T, "n_overlaps": n_overlaps, "coupled": coupled,
        "overlap_fraction": overlap_fraction, "mean_mp": round(mean_mp, 4),
        "motif_distance_best": round(best_motif_dist, 4),
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    base_dir = Path(hydra.utils.get_original_cwd())
    raw_dir = base_dir / cfg.data.raw_dir
    out_dir = base_dir / cfg.output.dir / "E12"
    out_dir.mkdir(parents=True, exist_ok=True)

    # m values: override via analysis.m_values=[12,24,36,48] if desired
    m_values_raw = OmegaConf.select(cfg, "analysis.m_values", default=None)
    if m_values_raw is not None:
        m_values = [int(x) for x in m_values_raw]
    else:
        m_values = [12, 24, 36, 48]

    k: int = int(cfg.matrix_profile.n_motifs)  # 3

    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions
    indicators = ["AO", "ENSO", "NAO", "PNA"]

    if cfg.run_mvp_only:
        log.info("MVP mode: africa × NAO only, m=[12,24].")
        all_regions = ["africa"]
        indicators = ["NAO"]
        m_values = [12, 24]

    log.info("m values to sweep: %s", m_values)
    log.info("Regions: %d | Indicators: %d", len(all_regions), len(indicators))
    log.info("Total cells: %d", len(m_values) * len(all_regions) * len(indicators))

    rows: list[dict] = []

    for m in m_values:
        log.info("── m = %d ──", m)
        for region in all_regions:
            for indicator in indicators:
                try:
                    row = _analyse(m, region, indicator, k, raw_dir)
                    rows.append(row)
                except FileNotFoundError as exc:
                    log.warning("Skipping m=%d %s × %s: %s", m, region, indicator, exc)
                except Exception:
                    log.exception("Error at m=%d %s × %s", m, region, indicator)

    if not rows:
        log.warning("No results produced — check that raw kNDVI and climate data exist.")
        return

    df = pd.DataFrame(rows)
    out_path = out_dir / "m_sensitivity_E12.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved: %s (%d rows)", out_path, len(df))

    # Pivot: indicator coupling count per m
    try:
        pivot = df[df["coupled"].notna()].groupby(["m", "indicator"])["coupled"].sum().unstack()
        log.info("Regions with coupling (coupled=True) per m and indicator:\n%s",
                 pivot.to_string())
    except Exception:
        pass

    # Stability check: NAO ranking across m values
    try:
        nao_counts = df[df["indicator"] == "NAO"].groupby("m")["coupled"].sum()
        log.info("NAO coupling count by m:\n%s", nao_counts.to_string())
    except Exception:
        pass

    log.info("E12 complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
