"""
exp_E6_rolling_correlation.py — Rolling correlation temporal analysis.

Temporal claim T1: rolling coupling strength between kNDVI and climate indicators
shows a significant post-2010 decline.

For each region × indicator:
  1. Load mean kNDVI and climate indicator.
  2. Deseasonalise both: subtract 12-month rolling mean.
  3. Compute rolling Pearson r (window = analysis.rolling_window months, default 36).
  4. Test for monotonic trend via Mann-Kendall test (pymannkendall).

Outputs:
  outputs/E6/rolling_corr_{region}_{indicator}.csv
  outputs/E6/rolling_corr_trend_E6.csv

Usage (Hydra):
    python experiments/exp_E6_rolling_correlation.py
    python experiments/exp_E6_rolling_correlation.py run_mvp_only=true
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
# Helpers
# ---------------------------------------------------------------------------

def _load_kndvi_mean(raw_dir: Path, region: str) -> pd.Series:
    path = raw_dir / "kndvi" / f"kndvi_{region}_2000_2025.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["kndvi"].mean().sort_index()


def _load_climate(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "climate" / "climate_indicators_2000_2025.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").set_index("date")


def _deseasonalise(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Remove the seasonal component by subtracting a centred 12-month rolling mean.
    Returns the residual (anomaly) series.
    """
    seasonal = series.rolling(window=window, center=True, min_periods=window // 2).mean()
    return series - seasonal


def _rolling_pearson(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling Pearson correlation at lag-0.

    Uses a manual rolling window to get both the correlation and a rough
    point-in-time p-value from the t-distribution.
    """
    from scipy.stats import t as t_dist  # noqa: PLC0415

    combined = pd.concat([s1.rename("x"), s2.rename("y")], axis=1).dropna()
    n_total = len(combined)
    if n_total < window:
        log.warning("Series length (%d) < window (%d); returning empty.", n_total, window)
        return pd.DataFrame(columns=["date", "rolling_r", "p_value_instantaneous"])

    dates: list[pd.Timestamp] = []
    rs: list[float] = []
    ps: list[float] = []

    for start in range(n_total - window + 1):
        chunk = combined.iloc[start : start + window]
        x_c = chunk["x"].to_numpy()
        y_c = chunk["y"].to_numpy()

        # Mask NaN
        valid = ~(np.isnan(x_c) | np.isnan(y_c))
        n_valid = int(valid.sum())
        if n_valid < 4:
            r, p = float("nan"), float("nan")
        else:
            xv, yv = x_c[valid], y_c[valid]
            # Pearson r
            xm, ym = xv.mean(), yv.mean()
            num = float(np.sum((xv - xm) * (yv - ym)))
            den = float(np.sqrt(np.sum((xv - xm) ** 2) * np.sum((yv - ym) ** 2)))
            r = num / (den + 1e-14)
            r = float(np.clip(r, -1.0, 1.0))
            # t-statistic for significance
            if abs(r) >= 1.0:
                p = 0.0
            else:
                t_stat = r * np.sqrt(n_valid - 2) / np.sqrt(1 - r**2 + 1e-14)
                p = float(2 * t_dist.sf(abs(t_stat), df=n_valid - 2))

        # Date = last date of window
        dates.append(combined.index[start + window - 1])
        rs.append(r)
        ps.append(p)

    result = pd.DataFrame({"date": dates, "rolling_r": rs, "p_value_instantaneous": ps})
    return result


def _mann_kendall(series: pd.Series) -> dict:
    """Run Mann-Kendall trend test on a series of values."""
    try:
        import pymannkendall as mk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pymannkendall is required: pip install pymannkendall") from exc

    arr = series.dropna().to_numpy()
    if len(arr) < 4:
        return {
            "trend": "insufficient_data",
            "p": float("nan"),
            "Tau": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }

    res = mk.original_test(arr)
    return {
        "trend": res.trend,
        "p": float(res.p),
        "Tau": float(res.Tau),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
    }


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_e3_outputs(base_dir: Path, out_cfg: str) -> None:
    e3_dir = base_dir / out_cfg / "E3"
    if not e3_dir.exists():
        raise RuntimeError(
            f"E3 output directory not found: {e3_dir}\n"
            "Run exp_E3_overlap.py before exp_E6_rolling_correlation.py."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    raw_dir = base_dir / cfg.data.raw_dir
    out_dir = base_dir / cfg.output.dir / "E6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Soft dependency check (warn, don't hard-fail, in case E3 not done)
    e3_dir = base_dir / cfg.output.dir / "E3"
    if not e3_dir.exists():
        log.warning(
            "E3 output directory not found (%s). E6 does not directly depend on E3 "
            "outputs but you should ensure E3 has been run for consistency.",
            e3_dir,
        )

    indicators = ["AO", "ENSO", "NAO", "PNA"]
    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions

    rolling_window = int(cfg.analysis.rolling_window)
    alpha = float(cfg.analysis.alpha)

    if cfg.run_mvp_only:
        log.info("MVP mode: processing africa continent only.")
        all_regions = [r for r in continents if r == "africa"]
        if not all_regions:
            all_regions = ["africa"]

    climate_df = _load_climate(raw_dir)

    trend_rows = []

    for region in all_regions:
        kndvi_path = raw_dir / "kndvi" / f"kndvi_{region}_2000_2025.parquet"
        if not kndvi_path.exists():
            log.warning("kNDVI file missing for region=%s, skipping.", region)
            continue

        kndvi_mean = _load_kndvi_mean(raw_dir, region)
        kndvi_anom = _deseasonalise(kndvi_mean)

        for indicator in indicators:
            if indicator not in climate_df.columns:
                log.warning("Indicator %s not in climate data, skipping.", indicator)
                continue

            ind_series = climate_df[indicator].dropna()
            ind_anom = _deseasonalise(ind_series)

            log.info("Computing rolling r for %s × %s (window=%d)...", region, indicator, rolling_window)

            roll_df = _rolling_pearson(kndvi_anom, ind_anom, window=rolling_window)

            if roll_df.empty:
                log.warning("No rolling correlation computed for %s × %s.", region, indicator)
                continue

            roll_df["region"] = region
            roll_df["indicator"] = indicator
            roll_df = roll_df[["date", "region", "indicator", "rolling_r", "p_value_instantaneous"]]
            roll_df.to_csv(
                out_dir / f"rolling_corr_{region}_{indicator}.csv", index=False
            )

            # Mann-Kendall trend test on rolling_r series
            mk_result = _mann_kendall(roll_df["rolling_r"])
            trend_rows.append(
                {
                    "region": region,
                    "indicator": indicator,
                    "mk_trend": mk_result["trend"],
                    "mk_p": round(mk_result["p"], 6),
                    "mk_tau": round(mk_result["Tau"], 4),
                    "mk_slope": round(mk_result["slope"], 6),
                    "mk_intercept": round(mk_result["intercept"], 4),
                    "significant": bool(mk_result["p"] < alpha)
                    if not (mk_result["p"] != mk_result["p"])  # NaN check
                    else False,
                    "n_windows": len(roll_df),
                }
            )

    trend_df = pd.DataFrame(trend_rows)
    trend_path = out_dir / "rolling_corr_trend_E6.csv"
    trend_df.to_csv(trend_path, index=False)
    log.info("Saved Mann-Kendall trend summary: %s (%d rows).", trend_path, len(trend_df))

    n_sig = (trend_df["significant"] == True).sum()  # noqa: E712
    n_total = len(trend_df)
    n_declining = (
        (trend_df["mk_trend"].isin(["decreasing", "decreasing (pre-whitened)"]))
        & trend_df["significant"]
    ).sum()
    log.info(
        "Significant MK trends: %d / %d; declining: %d.",
        n_sig, n_total, n_declining,
    )


if __name__ == "__main__":
    main()
