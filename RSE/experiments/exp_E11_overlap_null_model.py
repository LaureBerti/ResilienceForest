"""
exp_E11_overlap_null_model.py — Permutation null model for motif overlap.

For each region × indicator pair, computes the expected overlap count between
two independent sets of k motifs drawn uniformly at random on the observed
time-series length, using a ±max_lag coincidence window.

This provides the null distribution against which the observed overlap counts
(from E3) are compared, allowing the NAO coupling claim to be stated as:
"observed overlap exceeds the null 95th percentile in 13/14 regions."

Algorithm (per pair):
  1. Let T = kNDVI series length, m = subsequence length, k = n_motifs.
  2. Valid motif start positions: [0, T - m].
  3. For each of N=10,000 permutations:
       - Draw k kNDVI motif starts uniformly from [0, T-m] without replacement.
       - Draw k climate motif starts uniformly from [0, T-m] without replacement.
       - Count pairs (a, b) where |a - b| ≤ max_lag (same definition as E3).
  4. Report: null_mean, null_std, null_p95, null_p99, observed_count,
             p_perm (fraction of permutations with count ≥ observed).

Outputs:
  outputs/E11/null_model_{region}_{indicator}.csv   — per-pair permutation dist
  outputs/E11/null_model_summary_E11.csv            — all pairs in one table

Usage (Hydra):
    python experiments/exp_E11_overlap_null_model.py
    python experiments/exp_E11_overlap_null_model.py run_mvp_only=true
    python experiments/exp_E11_overlap_null_model.py analysis.n_permutations=1000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Series-length lookup (re-uses E2 parquet to determine T per region)
# ---------------------------------------------------------------------------

def _series_length(region: str, e2_dir: Path) -> int:
    """Return the number of months in the kNDVI time series for a region."""
    mp_path = e2_dir / f"matrix_profile_{region}.parquet"
    if mp_path.exists():
        df = pd.read_parquet(mp_path)
        return len(df) + 23  # MP length = T - m + 1, m=24 → T = len(mp) + 23
    kndvi_dir = BASE_DIR / "data" / "raw" / "kndvi"
    parquet_path = kndvi_dir / f"kndvi_{region}_2000_2025.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return int(df["date"].nunique())
    raise FileNotFoundError(
        f"Cannot determine series length for {region}: "
        f"no E2 parquet and no raw kNDVI parquet."
    )


def _observed_count(region: str, indicator: str, e3_dir: Path) -> int:
    """Load observed overlap count from E3."""
    path = e3_dir / f"overlap_{region}_{indicator}.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if df.empty or "overlap_flag" not in df.columns:
        return 0
    return int(df["overlap_flag"].sum())


# ---------------------------------------------------------------------------
# Permutation null model
# ---------------------------------------------------------------------------

def _run_permutation(
    T: int,
    m: int,
    k: int,
    max_lag: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """
    Returns array of overlap counts under the null (uniform random placement).

    Parameters
    ----------
    T : total time-series length (months)
    m : subsequence length
    k : number of motifs per series
    max_lag : coincidence window (±max_lag months)
    n_permutations : number of Monte Carlo draws
    seed : random seed

    Returns
    -------
    counts : shape (n_permutations,) — overlap count in each permutation
    """
    rng = np.random.default_rng(seed)
    n_valid = T - m + 1  # number of valid start positions
    if n_valid < k:
        # Series too short to draw k non-overlapping starts
        return np.zeros(n_permutations, dtype=int)

    counts = np.empty(n_permutations, dtype=int)
    for i in range(n_permutations):
        kndvi_starts = rng.choice(n_valid, size=k, replace=False)
        clim_starts = rng.choice(n_valid, size=k, replace=False)
        # Count pairs within ±max_lag (vectorised)
        diff = np.abs(kndvi_starts[:, None] - clim_starts[None, :])
        counts[i] = int((diff <= max_lag).any(axis=1).sum())

    return counts


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def _analyse_pair(
    region: str,
    indicator: str,
    T: int,
    m: int,
    k: int,
    max_lag: int,
    n_permutations: int,
    seed: int,
    observed: int,
    out_dir: Path,
) -> dict[str, Any]:
    null_counts = _run_permutation(T, m, k, max_lag, n_permutations, seed)

    null_mean = float(np.mean(null_counts))
    null_std = float(np.std(null_counts))
    null_p95 = float(np.percentile(null_counts, 95))
    null_p99 = float(np.percentile(null_counts, 99))

    # Empirical p-value: fraction of permutations with count ≥ observed
    p_perm = float(np.mean(null_counts >= observed))
    exceeds_p95 = bool(observed > null_p95)
    exceeds_p99 = bool(observed > null_p99)

    # Save per-pair permutation distribution (histogram bins)
    bins = np.arange(0, max(null_counts.max(), observed) + 2)
    hist, _ = np.histogram(null_counts, bins=bins)
    dist_df = pd.DataFrame({
        "count_value": bins[:-1],
        "null_frequency": hist,
    })
    dist_df["region"] = region
    dist_df["indicator"] = indicator
    dist_path = out_dir / f"null_model_{region}_{indicator}.csv"
    dist_df.to_csv(dist_path, index=False)

    log.info(
        "  %s × %s | observed=%d | null μ=%.2f σ=%.2f p95=%.1f | "
        "p_perm=%.4f | exceeds_p95=%s",
        region, indicator, observed, null_mean, null_std, null_p95,
        p_perm, exceeds_p95,
    )

    return {
        "region": region,
        "indicator": indicator,
        "T": T,
        "m": m,
        "k_motifs": k,
        "max_lag": max_lag,
        "observed_count": observed,
        "null_mean": round(null_mean, 3),
        "null_std": round(null_std, 3),
        "null_p95": round(null_p95, 2),
        "null_p99": round(null_p99, 2),
        "p_perm": round(p_perm, 4),
        "exceeds_null_p95": exceeds_p95,
        "exceeds_null_p99": exceeds_p99,
        "n_permutations": n_permutations,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    base_dir = Path(hydra.utils.get_original_cwd())
    e2_dir = base_dir / cfg.output.dir / "E2"
    e3_dir = base_dir / cfg.output.dir / "E3"
    out_dir = base_dir / cfg.output.dir / "E11"
    out_dir.mkdir(parents=True, exist_ok=True)

    m: int = int(cfg.matrix_profile.subsequence_length)  # 24
    k: int = int(cfg.matrix_profile.n_motifs)            # 3
    max_lag: int = m                                       # ±24 months (same as E3)
    seed: int = int(cfg.analysis.seed)                    # 42
    # Allow override via config; default 10000
    n_permutations: int = int(OmegaConf.select(cfg, "analysis.n_permutations", default=10_000))

    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions
    indicators = ["AO", "ENSO", "NAO", "PNA"]

    if cfg.run_mvp_only:
        log.info("MVP mode: africa × NAO only.")
        all_regions = ["africa"]
        indicators = ["NAO"]

    log.info(
        "Parameters: m=%d, k=%d, max_lag=±%d months, n_permutations=%d",
        m, k, max_lag, n_permutations,
    )

    summaries: list[dict] = []

    for region in all_regions:
        try:
            T = _series_length(region, e2_dir)
        except FileNotFoundError as exc:
            log.warning("Skipping %s — cannot determine T: %s", region, exc)
            continue

        for indicator in indicators:
            observed = _observed_count(region, indicator, e3_dir)
            try:
                row = _analyse_pair(
                    region, indicator, T, m, k, max_lag,
                    n_permutations, seed, observed, out_dir,
                )
                summaries.append(row)
            except Exception:
                log.exception("Error processing %s × %s", region, indicator)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = out_dir / "null_model_summary_E11.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info("Saved summary: %s (%d rows)", summary_path, len(summary_df))

        # Count regions per indicator where observed exceeds null p95
        pivot = summary_df.pivot_table(
            index="indicator", values="exceeds_null_p95",
            aggfunc="sum"
        )
        log.info("Regions exceeding null p95 per indicator:\n%s", pivot.to_string())
    else:
        log.warning("No pairs processed — check E2 and E3 outputs.")

    log.info("E11 complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
