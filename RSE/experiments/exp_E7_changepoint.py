"""
exp_E7_changepoint.py — Structural break detection in intact forest kNDVI.

Temporal claim T2: intact forest kNDVI shows detectable structural breaks at
multiple continents, clustered around known climate extreme years.

Methods:
  - PELT (model="rbf", pen=10)
  - BinSeg (model="l2", n_bkps = analysis.n_changepoints)

Cross-references detected break years against known climate extreme events.

Outputs:
  outputs/E7/changepoints_{region}.csv
  outputs/E7/breakpoint_summary_E7.csv

Dependency: E2 kNDVI parquet files (data/raw/kndvi/).

Usage (Hydra):
    python experiments/exp_E7_changepoint.py
    python experiments/exp_E7_changepoint.py run_mvp_only=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known climate extreme events (year ranges, ±12-month window applied later)
# ---------------------------------------------------------------------------

CLIMATE_EXTREMES: list[dict] = [
    {"label": "ENSO_drought_2002_2003", "year_start": 2002, "year_end": 2003},
    {"label": "amazon_megadrought_russian_heatwave_2010", "year_start": 2010, "year_end": 2010},
    {"label": "strong_el_nino_2015_2016", "year_start": 2015, "year_end": 2016},
    {"label": "australia_bushfires_amazon_drought_2019_2020", "year_start": 2019, "year_end": 2020},
]


def _extreme_year_range(tolerance_months: int = 12) -> set[int]:
    """Return set of calendar years within ±tolerance_months of any extreme event."""
    tol_yr = tolerance_months / 12.0
    years: set[int] = set()
    for ev in CLIMATE_EXTREMES:
        lo = int(np.floor(ev["year_start"] - tol_yr))
        hi = int(np.ceil(ev["year_end"] + tol_yr))
        years.update(range(lo, hi + 1))
    return years


EXTREME_YEARS = _extreme_year_range(tolerance_months=12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kndvi_mean(raw_dir: Path, region: str) -> pd.Series:
    path = raw_dir / "kndvi" / f"kndvi_{region}_2000_2025.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["kndvi"].mean().sort_index()


def _detect_pelt(signal_arr: np.ndarray, pen: float = 10.0) -> list[int]:
    """PELT change-point detection; returns list of break indices (0-based)."""
    import ruptures as rpt  # noqa: PLC0415

    algo = rpt.Pelt(model="rbf", min_size=3, jump=1)
    algo.fit(signal_arr.reshape(-1, 1))
    try:
        result = algo.predict(pen=pen)
    except Exception as exc:  # noqa: BLE001
        log.warning("PELT failed: %s", exc)
        return []
    # ruptures returns indices as (end of segment, exclusive); last = len(signal)
    # Remove the final index which marks series end
    return [bp - 1 for bp in result if bp < len(signal_arr)]


def _detect_binseg(signal_arr: np.ndarray, n_bkps: int = 3) -> list[int]:
    """BinSeg change-point detection."""
    import ruptures as rpt  # noqa: PLC0415

    algo = rpt.Binseg(model="l2", min_size=3, jump=1)
    algo.fit(signal_arr.reshape(-1, 1))
    try:
        result = algo.predict(n_bkps=n_bkps)
    except Exception as exc:  # noqa: BLE001
        log.warning("BinSeg failed: %s", exc)
        return []
    return [bp - 1 for bp in result if bp < len(signal_arr)]


def _breakpoints_to_df(
    indices: list[int],
    dates: pd.DatetimeIndex,
    region: str,
    method: str,
    penalty: float | int | str,
) -> pd.DataFrame:
    rows = []
    for idx in indices:
        if 0 <= idx < len(dates):
            rows.append(
                {
                    "region": region,
                    "break_index": idx,
                    "break_date": dates[idx],
                    "break_year": dates[idx].year,
                    "method": method,
                    "penalty": str(penalty),
                    "near_climate_extreme": dates[idx].year in EXTREME_YEARS,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    raw_dir = base_dir / cfg.data.raw_dir
    out_dir = base_dir / cfg.output.dir / "E7"
    out_dir.mkdir(parents=True, exist_ok=True)

    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions

    n_bkps = int(cfg.analysis.n_changepoints)
    pelt_pen = 10.0  # fixed; sufficient sensitivity for monthly kNDVI

    if cfg.run_mvp_only:
        log.info("MVP mode: processing africa + amazon only.")
        all_regions = ["africa", "amazon"]

    all_cp_frames: list[pd.DataFrame] = []

    for region in all_regions:
        kndvi_path = raw_dir / "kndvi" / f"kndvi_{region}_2000_2025.parquet"
        if not kndvi_path.exists():
            log.warning("kNDVI file missing for region=%s, skipping.", region)
            continue

        kndvi_mean = _load_kndvi_mean(raw_dir, region)
        signal_arr = kndvi_mean.to_numpy(dtype=np.float64)
        dates = kndvi_mean.index

        if len(signal_arr) < 24:
            log.warning("Too few observations for region=%s (%d), skipping.", region, len(signal_arr))
            continue

        log.info("Detecting change points for region=%s (n=%d)...", region, len(signal_arr))

        pelt_indices = _detect_pelt(signal_arr, pen=pelt_pen)
        binseg_indices = _detect_binseg(signal_arr, n_bkps=n_bkps)

        pelt_df = _breakpoints_to_df(pelt_indices, dates, region, "PELT", pelt_pen)
        binseg_df = _breakpoints_to_df(binseg_indices, dates, region, "BinSeg", n_bkps)

        region_df = pd.concat([pelt_df, binseg_df], ignore_index=True)
        region_df = region_df.sort_values(["method", "break_date"])

        region_out = out_dir / f"changepoints_{region}.csv"
        region_df.to_csv(region_out, index=False)
        log.info("  %s: PELT found %d, BinSeg found %d break points.", region, len(pelt_df), len(binseg_df))

        all_cp_frames.append(region_df)

    if not all_cp_frames:
        log.warning("No change points detected; summary will be empty.")
        summary_df = pd.DataFrame()
    else:
        all_cp = pd.concat(all_cp_frames, ignore_index=True)

        # Fraction near climate extremes per region × method
        summary_rows = []
        for (region, method), grp in all_cp.groupby(["region", "method"]):
            n_total = len(grp)
            n_near = int(grp["near_climate_extreme"].sum())
            fraction = n_near / n_total if n_total > 0 else float("nan")
            # Unique break years
            break_years = sorted(grp["break_year"].unique().tolist())
            summary_rows.append(
                {
                    "region": region,
                    "method": method,
                    "n_breakpoints": n_total,
                    "n_near_climate_extreme": n_near,
                    "fraction_near_extreme": round(fraction, 4),
                    "break_years": str(break_years),
                    "extreme_years_reference": str(sorted(EXTREME_YEARS)),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(["region", "method"])

    summary_path = out_dir / "breakpoint_summary_E7.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Saved breakpoint summary: %s (%d rows).", summary_path, len(summary_df))

    if not summary_df.empty:
        mean_frac = summary_df["fraction_near_extreme"].mean()
        log.info(
            "Mean fraction of break points near a known climate extreme: %.1f %%.",
            mean_frac * 100,
        )


if __name__ == "__main__":
    main()
