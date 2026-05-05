"""
exp_E8_motif_evolution.py — Motif predictability index temporal analysis.

Temporal claim T3: motif predictability index (inverse of minimum matrix profile
distance) shows a temporal trend toward greater irregularity post-2015.

For each region:
  1. Load matrix profile from E2 outputs.
  2. Compute MPI = rolling 24-month minimum of MP values (low = high predictability).
  3. Compute rolling 24-month mean MP value (overall irregularity trend).
  4. Test for monotonic trend in MPI via Mann-Kendall test.
  5. Count "anomalous windows" per year (MP > mean + 1.5 × std).

Outputs:
  outputs/E8/mpi_{region}.csv
  outputs/E8/mpi_trend_E8.csv

Dependency: outputs/E2/matrix_profile_{region}.parquet (from E2).

Usage (Hydra):
    python experiments/exp_E8_motif_evolution.py
    python experiments/exp_E8_motif_evolution.py run_mvp_only=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# MPI rolling window (months)
MPI_WINDOW = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_matrix_profile(e2_dir: Path, region: str) -> pd.DataFrame:
    """
    Load E2 matrix profile parquet.

    Expected columns: date, matrix_profile_value (or 'mp_value').
    Returns a DataFrame with columns: date, mp_value — sorted by date.
    """
    path = e2_dir / f"matrix_profile_{region}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"E2 matrix profile not found: {path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalise column name
    for candidate in ("matrix_profile_value", "mp_value", "mp", "value"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "mp_value"})
            break
    else:
        raise ValueError(
            f"Cannot find matrix profile value column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[["date", "mp_value"]]


def _compute_mpi(df: pd.DataFrame, window: int = MPI_WINDOW) -> pd.DataFrame:
    """
    Compute MPI (rolling minimum), rolling mean, and per-year anomaly count.

    Returns a DataFrame with columns:
      date, mp_value, mpi_rolling_min, mp_rolling_mean, anomalous_flag, year.
    """
    mp = df["mp_value"].to_numpy(dtype=np.float64)

    global_mean = float(np.nanmean(mp))
    global_std = float(np.nanstd(mp))
    anomaly_threshold = global_mean + 1.5 * global_std

    result = df.copy()
    result["mpi_rolling_min"] = (
        result["mp_value"]
        .rolling(window=window, min_periods=window // 2)
        .min()
    )
    result["mp_rolling_mean"] = (
        result["mp_value"]
        .rolling(window=window, min_periods=window // 2)
        .mean()
    )
    result["anomalous_flag"] = result["mp_value"] > anomaly_threshold
    result["year"] = result["date"].dt.year

    return result


def _annual_anomaly_counts(df: pd.DataFrame) -> pd.Series:
    """Return per-year count of anomalous windows."""
    return df.groupby("year")["anomalous_flag"].sum().rename("n_anomalous_windows")


def _mann_kendall(series: pd.Series) -> dict:
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
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e2_dir = base_dir / cfg.output.dir / "E2"
    out_dir = base_dir / cfg.output.dir / "E8"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dependency check
    if not e2_dir.exists():
        raise RuntimeError(
            f"E2 output directory not found: {e2_dir}\n"
            "Run exp_E2_matrix_profile.py before exp_E8_motif_evolution.py."
        )

    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions

    alpha = float(cfg.analysis.alpha)

    if cfg.run_mvp_only:
        log.info("MVP mode: processing africa + amazon only.")
        all_regions = ["africa", "amazon"]

    trend_rows = []

    for region in all_regions:
        try:
            mp_df = _load_matrix_profile(e2_dir, region)
        except FileNotFoundError as exc:
            log.warning("Skipping region=%s: %s", region, exc)
            continue
        except ValueError as exc:
            log.warning("Skipping region=%s: %s", region, exc)
            continue

        log.info("Computing MPI for region=%s (n=%d)...", region, len(mp_df))

        result_df = _compute_mpi(mp_df, window=MPI_WINDOW)
        annual_anom = _annual_anomaly_counts(result_df)

        # Build per-month output with annual anomaly count merged in
        out_df = result_df[["date", "mpi_rolling_min", "mp_rolling_mean", "anomalous_flag", "year"]].copy()
        out_df = out_df.merge(
            annual_anom.reset_index(), on="year", how="left"
        )
        out_df = out_df.drop(columns=["anomalous_flag", "year"])
        out_df.to_csv(out_dir / f"mpi_{region}.csv", index=False)

        # Mann-Kendall on MPI (rolling min — increase = greater irregularity)
        mk_mpi = _mann_kendall(result_df["mpi_rolling_min"])
        # Mann-Kendall on rolling mean (overall trend)
        mk_mean = _mann_kendall(result_df["mp_rolling_mean"])

        # Post-2015 anomaly rate
        total_n = int((result_df["date"].dt.year >= 2000).sum())
        post2015_n = int((result_df["date"].dt.year >= 2015).sum())
        pre2015_anom = int(
            result_df.loc[result_df["date"].dt.year < 2015, "anomalous_flag"].sum()
        )
        post2015_anom = int(
            result_df.loc[result_df["date"].dt.year >= 2015, "anomalous_flag"].sum()
        )
        pre2015_base = max(total_n - post2015_n, 1)
        post2015_base = max(post2015_n, 1)

        trend_rows.append(
            {
                "region": region,
                "n_months": len(result_df),
                # MPI trend
                "mpi_mk_trend": mk_mpi["trend"],
                "mpi_mk_p": round(mk_mpi["p"], 6),
                "mpi_mk_tau": round(mk_mpi["Tau"], 4),
                "mpi_mk_slope": round(mk_mpi["slope"], 8),
                "mpi_significant": bool(mk_mpi["p"] < alpha)
                if not np.isnan(mk_mpi["p"])
                else False,
                # Overall irregularity trend
                "mean_mp_mk_trend": mk_mean["trend"],
                "mean_mp_mk_p": round(mk_mean["p"], 6),
                "mean_mp_mk_tau": round(mk_mean["Tau"], 4),
                "mean_mp_significant": bool(mk_mean["p"] < alpha)
                if not np.isnan(mk_mean["p"])
                else False,
                # Anomaly counts
                "pre2015_anomaly_rate": round(pre2015_anom / pre2015_base, 4),
                "post2015_anomaly_rate": round(post2015_anom / post2015_base, 4),
                "anomaly_rate_increase": round(
                    (post2015_anom / post2015_base) - (pre2015_anom / pre2015_base), 4
                ),
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    trend_path = out_dir / "mpi_trend_E8.csv"
    trend_df.to_csv(trend_path, index=False)
    log.info("Saved MPI trend summary: %s (%d rows).", trend_path, len(trend_df))

    if not trend_df.empty:
        n_sig_mpi = (trend_df["mpi_significant"] == True).sum()  # noqa: E712
        n_increasing = (
            trend_df["mpi_mk_trend"].str.contains("increasing", na=False) & trend_df["mpi_significant"]
        ).sum()
        log.info(
            "Significant MPI trends: %d / %d; increasing (greater irregularity): %d.",
            n_sig_mpi, len(trend_df), n_increasing,
        )


if __name__ == "__main__":
    main()
