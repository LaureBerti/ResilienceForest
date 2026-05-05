"""
exp_E5_temporal_test.py — Temporal shift test for forest resilience paper.

Claim supported: C4 — significantly more motif–climate overlaps pre-2010 vs post-2010.

For each region × indicator:
  1. Load per-year overlap counts from E3 outputs.
  2. Split at analysis.pre_period_end (default 2010-01-01).
  3. Mann-Whitney U test (pre > post, one-sided).
  4. Effect size: rank-biserial correlation.
  5. Pearson linear trend on annual overlap counts (monotonic decline?).

Usage (Hydra):
    python experiments/exp_E5_temporal_test.py
    python experiments/exp_E5_temporal_test.py run_mvp_only=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import mannwhitneyu, pearsonr

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_overlap_counts(
    e3_dir: Path, region: str, indicator: str,
    first_year: int = 2000, last_year: int = 2025,
) -> pd.DataFrame:
    """
    Load per-row overlap flags from E3 and return annual counts for every year
    in [first_year, last_year], filling years with no overlap events with 0.

    E3 only writes rows for motif pairs that DID overlap, so years with zero
    overlaps are absent from the CSV.  Reindexing to the full year range is
    essential so the Mann-Whitney test sees the correct group sizes.

    Returns DataFrame with columns: year (int), overlap_count (int).
    """
    path = e3_dir / f"overlap_{region}_{indicator}.csv"
    if not path.exists():
        raise FileNotFoundError(f"E3 overlap file not found: {path}")

    all_years = pd.DataFrame({"year": range(first_year, last_year + 1)})

    df = pd.read_csv(path)
    if df.empty or "overlap_flag" not in df.columns:
        all_years["overlap_count"] = 0
        return all_years

    # Ensure year column exists
    if "year" not in df.columns:
        if "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year
        elif "month" in df.columns:
            raise ValueError(f"E3 file {path} has no year or date column.")

    annual = (
        df.groupby("year")["overlap_flag"]
        .sum()
        .reset_index()
        .rename(columns={"overlap_flag": "overlap_count"})
    )
    # Merge to fill missing years with 0
    annual = all_years.merge(annual, on="year", how="left").fillna(0)
    annual["overlap_count"] = annual["overlap_count"].astype(int)
    return annual


def _rank_biserial(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation: r = 1 - 2U / (n1 * n2)."""
    denom = n1 * n2
    if denom == 0:
        return float("nan")
    return 1.0 - 2.0 * U / denom


def _pearson_trend(years: np.ndarray, counts: np.ndarray) -> dict:
    """Linear trend via Pearson r on annual data."""
    if len(years) < 3:
        return {"trend_r": float("nan"), "trend_p": float("nan"), "trend_direction": "insufficient_data"}
    r, p = pearsonr(years.astype(float), counts.astype(float))
    direction = "decreasing" if r < 0 else "increasing"
    return {"trend_r": float(r), "trend_p": float(p), "trend_direction": direction}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e3_dir = base_dir / cfg.output.dir / "E3"
    out_dir = base_dir / cfg.output.dir / "E5"
    out_dir.mkdir(parents=True, exist_ok=True)

    indicators = ["AO", "ENSO", "NAO", "PNA"]
    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions

    pre_end = pd.Timestamp(cfg.analysis.pre_period_end)
    pre_end_year = pre_end.year  # 2010

    alpha = float(cfg.analysis.alpha)

    if cfg.run_mvp_only:
        log.info("MVP mode: processing africa × AO only.")
        all_regions = ["africa"]
        indicators = ["AO"]

    rows = []

    for region in all_regions:
        for indicator in indicators:
            try:
                annual = _load_overlap_counts(e3_dir, region, indicator)
            except FileNotFoundError as exc:
                log.warning("Skipping %s × %s: %s", region, indicator, exc)
                continue

            pre = annual.loc[annual["year"] < pre_end_year, "overlap_count"].to_numpy()
            post = annual.loc[annual["year"] >= pre_end_year, "overlap_count"].to_numpy()

            n_pre = len(pre)
            n_post = len(post)

            if n_pre < 2 or n_post < 2:
                log.warning(
                    "%s × %s: insufficient data (n_pre=%d, n_post=%d), skipping.",
                    region, indicator, n_pre, n_post,
                )
                continue

            # Mann-Whitney U test: H1 = pre-2010 overlap counts > post-2010
            try:
                stat_result = mannwhitneyu(pre, post, alternative="greater")
                U_stat = float(stat_result.statistic)
                p_value = float(stat_result.pvalue)
            except Exception as exc:  # noqa: BLE001
                log.warning("mannwhitneyu failed for %s × %s: %s", region, indicator, exc)
                U_stat, p_value = float("nan"), float("nan")

            effect_size = _rank_biserial(U_stat, n_pre, n_post)
            significant = bool(p_value < alpha) if not np.isnan(p_value) else False
            direction = (
                "pre>post" if np.nanmean(pre) > np.nanmean(post) else "pre≤post"
            )

            # Pearson linear trend on full time series
            trend = _pearson_trend(annual["year"].to_numpy(), annual["overlap_count"].to_numpy())

            rows.append(
                {
                    "region": region,
                    "indicator": indicator,
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "mean_pre": round(float(np.mean(pre)), 3),
                    "mean_post": round(float(np.mean(post)), 3),
                    "U_stat": round(U_stat, 2),
                    "p_value": round(p_value, 6),
                    "effect_size_rbc": round(effect_size, 4),
                    "direction": direction,
                    "significant": significant,
                    "trend_r": round(trend["trend_r"], 4),
                    "trend_p": round(trend["trend_p"], 6),
                    "trend_direction": trend["trend_direction"],
                }
            )

    results_df = pd.DataFrame(rows)
    out_path = out_dir / "temporal_test_E5.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Saved %s (%d rows).", out_path, len(results_df))

    if results_df.empty:
        log.warning("No region×indicator pairs had sufficient data for the temporal test.")
        return

    n_sig = results_df["significant"].sum()
    n_total = len(results_df)
    log.info(
        "Significant pre>post pairs (α=%.2f): %d / %d (%.1f %%).",
        alpha, n_sig, n_total, 100 * n_sig / max(n_total, 1),
    )


if __name__ == "__main__":
    main()
