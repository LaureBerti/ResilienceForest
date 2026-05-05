"""
exp_E9_recent_trend.py — Pre/post 2020 comparison of forest resilience metrics.

Temporal claim T4: forest resilience metrics changed significantly between the
pre-2020 (2015–2019) and post-2020 (2020–2025) periods across major forest regions.

For each region × metric:
  1. Load outputs/E8/mpi_{region}.csv
  2. Aggregate to annual means (mpi_rolling_min, mp_rolling_mean) / sum (n_anomalous_windows)
  3. Split into pre (2015–2019) and post (2020–2025) groups
  4. Two-sided Mann–Whitney U test
  5. Rank-biserial effect size: r = 1 - 2U / (n1 * n2)
  6. Direction: increasing / decreasing

Output:
  outputs/E9/recent_trend_E9.csv

Columns: region, metric, n_pre, n_post, mean_pre, mean_post,
         U_stat, p_value, effect_size_rbc, direction, significant

Usage (Hydra):
    python experiments/exp_E9_recent_trend.py
    python experiments/exp_E9_recent_trend.py run_mvp_only=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import mannwhitneyu

log = logging.getLogger(__name__)

# Fixed region display order: continents first, then subregions
REGION_ORDER = [
    "africa",
    "asia",
    "europe",
    "north_america",
    "south_america",
    "australia_oceania",
    "amazon",
    "congo",
    "boreal_siberia",
    "southeast_asia",
    "canadian_boreal",
    "scandinavian_boreal",
    "papua_new_guinea",
    "russian_far_east",
]

METRICS = ["mpi_min", "mpi_mean", "n_anomalous"]

# Aggregation for each metric
AGG_MAP = {
    "mpi_min": ("mpi_rolling_min", "mean"),
    "mpi_mean": ("mp_rolling_mean", "mean"),
    "n_anomalous": ("n_anomalous_windows", "sum"),
}

PRE_YEARS = range(2015, 2020)   # 2015–2019 inclusive
POST_YEARS = range(2020, 2026)  # 2020–2025 inclusive


def _load_annual(e8_dir: Path, region: str) -> pd.DataFrame | None:
    """Load mpi_{region}.csv and return annual aggregates."""
    path = e8_dir / f"mpi_{region}.csv"
    if not path.exists():
        log.warning("E8 file not found for region=%s: %s", region, path)
        return None

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # Annual aggregations
    annual = (
        df.groupby("year")
        .agg(
            mpi_rolling_min=("mpi_rolling_min", "mean"),
            mp_rolling_mean=("mp_rolling_mean", "mean"),
            n_anomalous_windows=("n_anomalous_windows", "sum"),
        )
        .reset_index()
    )
    return annual


def _mann_whitney_rbc(
    pre: np.ndarray,
    post: np.ndarray,
    alpha: float,
) -> dict:
    """Two-sided MWU + rank-biserial effect size."""
    n1, n2 = len(pre), len(post)
    if n1 == 0 or n2 == 0:
        return {
            "U_stat": np.nan,
            "p_value": np.nan,
            "effect_size_rbc": np.nan,
            "direction": "insufficient_data",
            "significant": False,
        }

    try:
        # Exact permutation enumeration (C(n1+n2, n1) combinations; valid for n≤8)
        stat, p = mannwhitneyu(pre, post, alternative="two-sided", method="exact")
    except ValueError:
        # Ties prevent exact enumeration; fall back to normal approximation
        stat, p = mannwhitneyu(pre, post, alternative="two-sided", method="asymptotic")
    rbc = float(1.0 - 2.0 * stat / (n1 * n2))
    mean_pre = float(np.mean(pre))
    mean_post = float(np.mean(post))
    direction = "increasing" if mean_post > mean_pre else "decreasing"

    return {
        "U_stat": float(stat),
        "p_value": float(p),
        "effect_size_rbc": rbc,
        "direction": direction,
        "significant": bool(p < alpha),
    }


@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e8_dir = base_dir / cfg.output.dir / "E8"
    out_dir = base_dir / cfg.output.dir / "E9"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(cfg.analysis.alpha)

    if cfg.run_mvp_only:
        log.info("MVP mode: processing africa only.")
        regions = ["africa"]
    else:
        regions = REGION_ORDER

    rows = []

    for region in regions:
        annual = _load_annual(e8_dir, region)
        if annual is None:
            continue

        log.info("Processing region=%s (annual rows=%d)", region, len(annual))

        for metric in METRICS:
            col, _ = AGG_MAP[metric]

            pre_vals = annual.loc[annual["year"].isin(PRE_YEARS), col].dropna().to_numpy()
            post_vals = annual.loc[annual["year"].isin(POST_YEARS), col].dropna().to_numpy()

            stats = _mann_whitney_rbc(pre_vals, post_vals, alpha)

            rows.append(
                {
                    "region": region,
                    "metric": metric,
                    "n_pre": len(pre_vals),
                    "n_post": len(post_vals),
                    "mean_pre": round(float(np.mean(pre_vals)), 6) if len(pre_vals) > 0 else np.nan,
                    "mean_post": round(float(np.mean(post_vals)), 6) if len(post_vals) > 0 else np.nan,
                    "U_stat": round(stats["U_stat"], 4) if not np.isnan(stats["U_stat"]) else np.nan,
                    "p_value": round(stats["p_value"], 6) if not np.isnan(stats["p_value"]) else np.nan,
                    "effect_size_rbc": round(stats["effect_size_rbc"], 4) if not np.isnan(stats["effect_size_rbc"]) else np.nan,
                    "direction": stats["direction"],
                    "significant": stats["significant"],
                }
            )

    results = pd.DataFrame(rows)
    out_path = out_dir / "recent_trend_E9.csv"
    results.to_csv(out_path, index=False)
    log.info("Saved E9 results: %s (%d rows)", out_path, len(results))

    if not results.empty:
        n_total = len(results)
        n_sig = results["significant"].sum()
        log.info(
            "Significant pairs: %d / %d (%.1f%%)",
            n_sig, n_total, 100.0 * n_sig / n_total,
        )
        sig_df = results[results["significant"]]
        if not sig_df.empty:
            log.info("Significant region–metric pairs:\n%s",
                     sig_df[["region", "metric", "direction", "effect_size_rbc", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
