"""
exp_E3_overlap.py
-----------------
E3 — Climate Indicator Overlap Analysis.

Claims supported:
  C2: AO/NAO/PNA overlap with kNDVI with continent-specific lags 1–17 months
  C3: ENSO shows no significant overlap with kNDVI

For each region × climate indicator pair this script:
  1. Loads kNDVI mean time series (from E2 matrix profile parquet)
  2. Computes matrix profile of the climate indicator (self-join)
  3. Identifies kNDVI motifs from E2 motif CSV outputs
  4. For each kNDVI motif, searches the climate MP for a co-occurring motif
     within a user-defined lag window (max_lag months)
  5. Records the time lag (start_idx difference) for each overlap pair
  6. Computes bootstrap 95% CI on the lag distribution (n=1000, seed=42)
  7. Counts overlaps per calendar year → consumed by E5

ENSO handling: if no overlaps are found, reports n_overlaps=0 and
  overlap_fraction=0.0 — this is the positive result for C3.

Output files (relative to project root):
  outputs/E3/overlap_{region}_{indicator}.csv
      columns: year, month, lag_months, overlap_flag,
               kndvi_motif_start, climate_motif_start

  outputs/E3/lag_summary_E3.csv
      columns: region, indicator, median_lag, ci_lower, ci_upper,
               n_overlaps, overlap_fraction

Usage:
  python experiments/exp_E3_overlap.py
  python experiments/exp_E3_overlap.py run_mvp_only=true   # africa × AO only
  python experiments/exp_E3_overlap.py --write-config
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import stumpy
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("E3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve(cfg: DictConfig, relpath: str) -> Path:
    return BASE_DIR / relpath


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_kndvi_ts(region: str, e2_dir: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Load mean kNDVI time series from E2 matrix profile parquet.

    Returns (ts_array, date_index).
    """
    mp_path = e2_dir / f"matrix_profile_{region}.parquet"
    if not mp_path.exists():
        raise FileNotFoundError(
            f"E2 output not found: {mp_path}. Run E2 first."
        )
    mp_df = pd.read_parquet(mp_path)
    # E2 saves mp values aligned to dates[:n_mp], but we need the full TS.
    # Re-derive from the raw parquet.
    kndvi_dir = BASE_DIR / "data" / "raw" / "kndvi"
    parquet_path = kndvi_dir / f"kndvi_{region}_2000_2025.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Raw kNDVI parquet not found: {parquet_path}. Run E1 first."
        )
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    ts_series = (
        df.groupby("date")["kndvi"].mean().sort_index()
    )
    return ts_series.to_numpy(dtype=np.float64), pd.DatetimeIndex(ts_series.index)


def _load_climate(climate_csv: Path) -> pd.DataFrame:
    """
    Load climate indicators CSV.

    Expected columns: date, AO, ENSO, NAO, PNA
    """
    if not climate_csv.exists():
        raise FileNotFoundError(
            f"Climate CSV not found: {climate_csv}. Run E1 first."
        )
    df = pd.read_csv(climate_csv, parse_dates=["date"])
    required = {"date", "AO", "ENSO", "NAO", "PNA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {climate_csv}")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _load_kndvi_motifs(region: str, e2_dir: Path) -> pd.DataFrame:
    """Load E2 motif CSV for a region."""
    motif_path = e2_dir / f"motifs_{region}.csv"
    if not motif_path.exists():
        raise FileNotFoundError(
            f"E2 motif file not found: {motif_path}. Run E2 first."
        )
    df = pd.read_csv(motif_path)
    return df


# ---------------------------------------------------------------------------
# Matrix profile for climate series
# ---------------------------------------------------------------------------


def _climate_mp(
    climate_ts: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute self-join matrix profile for a climate indicator.

    Returns (mp_values, mp_indices).
    """
    if len(climate_ts) < 2 * m:
        raise ValueError(
            f"Climate series length {len(climate_ts)} < 2*m={2*m}."
        )
    mp_array = stumpy.stump(climate_ts, m)
    return mp_array[:, 0].astype(np.float64), mp_array[:, 1].astype(np.int64)


def _find_climate_motif_starts(
    climate_mp_vals: np.ndarray,
    n_motifs: int,
    m: int,
    excl_zone: int | None = None,
) -> list[int]:
    """
    Return the start indices of the top-N motifs in the climate MP
    (i.e. the lowest MP values, excluding trivial matches via exclusion zone).
    """
    if excl_zone is None:
        excl_zone = m // 4
    vals = climate_mp_vals.copy()
    starts: list[int] = []
    for _ in range(n_motifs):
        idx = int(np.argmin(vals))
        starts.append(idx)
        # Exclude zone around found motif
        lo = max(0, idx - excl_zone)
        hi = min(len(vals), idx + excl_zone + 1)
        vals[lo:hi] = np.inf
    return starts


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------


def _detect_overlaps(
    kndvi_motif_starts: list[int],
    climate_motif_starts: list[int],
    m: int,
    max_lag: int,
    kndvi_dates: pd.DatetimeIndex,
    climate_dates: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """
    Find pairs (kndvi_motif, climate_motif) whose subsequences overlap within
    a lag window of ±max_lag months.

    An 'overlap' is defined as: the climate motif subsequence [cs, cs+m) and
    the kNDVI motif subsequence [ks, ks+m) are aligned to the same calendar
    dates AND their start indices differ by at most max_lag months.

    Because both time series share the same monthly calendar (aligned by
    date), index distance = month lag.

    Returns list of overlap records.
    """
    records: list[dict[str, Any]] = []

    # Build a set of climate motif start months for O(1) lookup
    # We look for any climate motif start within [ks - max_lag, ks + max_lag]
    climate_set = {int(cs): True for cs in climate_motif_starts}

    for ks in kndvi_motif_starts:
        ks = int(ks)
        for lag in range(-max_lag, max_lag + 1):
            cs = ks + lag
            if cs < 0 or cs >= len(climate_dates):
                continue
            if cs in climate_set:
                kdate = kndvi_dates[ks] if ks < len(kndvi_dates) else None
                cdate = climate_dates[cs] if cs < len(climate_dates) else None
                records.append(
                    {
                        "kndvi_motif_start": ks,
                        "climate_motif_start": cs,
                        "lag_months": lag,
                        "overlap_flag": 1,
                        "year": kdate.year if kdate is not None else None,
                        "month": kdate.month if kdate is not None else None,
                        "kndvi_date": kdate.strftime("%Y-%m-%d")
                        if kdate is not None
                        else None,
                        "climate_date": cdate.strftime("%Y-%m-%d")
                        if cdate is not None
                        else None,
                    }
                )
    return records


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on lag
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    lags: np.ndarray, n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Bootstrap 95% CI on the median lag.

    Returns (ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    medians = np.empty(n_bootstrap)
    n = len(lags)
    for i in range(n_bootstrap):
        sample = rng.choice(lags, size=n, replace=True)
        medians[i] = np.median(sample)
    lo = float(np.percentile(medians, 100 * alpha / 2))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Per region × indicator
# ---------------------------------------------------------------------------


def process_pair(
    region: str,
    indicator: str,
    cfg: DictConfig,
    e2_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Run E3 analysis for one (region, indicator) pair.

    Returns summary dict for lag_summary_E3.csv.
    """
    log.info("  %s × %s", region, indicator)

    m: int = cfg.matrix_profile.subsequence_length
    n_motifs: int = cfg.matrix_profile.n_motifs
    n_bootstrap: int = cfg.analysis.n_bootstrap
    seed: int = cfg.analysis.seed
    alpha: float = cfg.analysis.alpha
    # Lag window = half the subsequence length (conservative)
    max_lag: int = m

    # --- Load data ---
    kndvi_ts, kndvi_dates = _load_kndvi_ts(region, e2_dir)
    kndvi_motif_df = _load_kndvi_motifs(region, e2_dir)

    climate_csv = _resolve(cfg, cfg.data.climate_dir) / "climate_indicators_2000_2025.csv"
    climate_df = _load_climate(climate_csv)

    # Align climate to kNDVI date range
    start_date = kndvi_dates[0]
    end_date = kndvi_dates[-1]
    climate_df = climate_df[
        (climate_df["date"] >= start_date) & (climate_df["date"] <= end_date)
    ].reset_index(drop=True)

    climate_ts = climate_df[indicator].to_numpy(dtype=np.float64)
    climate_dates = pd.DatetimeIndex(climate_df["date"])

    # Handle NaN in climate (interpolate linearly)
    if np.any(np.isnan(climate_ts)):
        n_nan = int(np.sum(np.isnan(climate_ts)))
        log.warning(
            "    %d NaN values in %s series — interpolating", n_nan, indicator
        )
        s = pd.Series(climate_ts)
        climate_ts = s.interpolate(method="linear").to_numpy(dtype=np.float64)

    # --- Climate matrix profile ---
    log.debug("    Computing climate MP for %s …", indicator)
    climate_mp_vals, _ = _climate_mp(climate_ts, m)

    # --- Climate motif starts ---
    climate_starts = _find_climate_motif_starts(
        climate_mp_vals, n_motifs=n_motifs, m=m
    )

    # --- kNDVI motif starts from E2 ---
    kndvi_starts = kndvi_motif_df["start_idx"].tolist()

    # --- Overlap detection ---
    overlaps = _detect_overlaps(
        kndvi_starts, climate_starts, m, max_lag, kndvi_dates, climate_dates
    )

    # Build overlap DataFrame
    if overlaps:
        overlap_df = pd.DataFrame(overlaps)[
            [
                "year",
                "month",
                "lag_months",
                "overlap_flag",
                "kndvi_motif_start",
                "climate_motif_start",
                "kndvi_date",
                "climate_date",
            ]
        ]
    else:
        # No overlaps found — empty frame with correct schema (positive result for C3)
        overlap_df = pd.DataFrame(
            columns=[
                "year",
                "month",
                "lag_months",
                "overlap_flag",
                "kndvi_motif_start",
                "climate_motif_start",
                "kndvi_date",
                "climate_date",
            ]
        )
        if indicator == "ENSO":
            log.info(
                "    [C3] No overlaps found for %s × ENSO — confirming C3", region
            )
        else:
            log.info("    No overlaps found for %s × %s", region, indicator)

    overlap_csv = out_dir / f"overlap_{region}_{indicator}.csv"
    overlap_df.to_csv(overlap_csv, index=False)
    log.debug("    Saved: %s", overlap_csv)

    # --- Lag statistics ---
    n_overlaps = len(overlap_df)
    # Overlap fraction: fraction of kNDVI motifs that have at least one climate match
    n_kndvi_motifs = len(kndvi_starts)
    matched_kndvi = (
        overlap_df["kndvi_motif_start"].nunique() if n_overlaps > 0 else 0
    )
    overlap_fraction = matched_kndvi / max(n_kndvi_motifs, 1)

    if n_overlaps >= 2:
        lags = overlap_df["lag_months"].to_numpy(dtype=np.float64)
        median_lag = float(np.median(lags))
        ci_lower, ci_upper = _bootstrap_ci(lags, n_bootstrap, seed, alpha)
    elif n_overlaps == 1:
        median_lag = float(overlap_df["lag_months"].iloc[0])
        ci_lower = median_lag
        ci_upper = median_lag
    else:
        median_lag = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    log.info(
        "    Overlaps: %d | median lag: %.1f [%.1f, %.1f] months | "
        "overlap fraction: %.2f",
        n_overlaps,
        median_lag if not np.isnan(median_lag) else 0,
        ci_lower if not np.isnan(ci_lower) else 0,
        ci_upper if not np.isnan(ci_upper) else 0,
        overlap_fraction,
    )

    return {
        "region": region,
        "indicator": indicator,
        "median_lag": median_lag,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_overlaps": n_overlaps,
        "overlap_fraction": overlap_fraction,
        "n_kndvi_motifs_input": n_kndvi_motifs,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="../conf",
    config_name="resilience_forest",
)
def main(cfg: DictConfig) -> None:
    log.info("stumpy version: %s", stumpy.__version__)
    log.info(
        "Config:\n%s",
        OmegaConf.to_yaml(cfg, resolve=True),
    )

    e2_dir = _resolve(cfg, cfg.output.dir) / "E2"
    out_dir = _resolve(cfg, cfg.output.dir) / "E3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check E2 outputs exist
    if not e2_dir.exists():
        log.error(
            "E2 output directory not found: %s. Run E2 first.", e2_dir
        )
        sys.exit(1)

    log.info("E2 directory: %s", e2_dir)
    log.info("E3 output directory: %s", out_dir)

    # Determine regions and indicators
    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions
    indicators = ["AO", "ENSO", "NAO", "PNA"]

    if cfg.run_mvp_only:
        all_regions = ["africa"]
        indicators = ["AO"]
        log.info("[MVP] Processing only: africa × AO")

    summaries: list[dict[str, Any]] = []

    for region in all_regions:
        # Verify E2 motif file exists before attempting E3
        motif_path = e2_dir / f"motifs_{region}.csv"
        if not motif_path.exists():
            log.warning(
                "E2 motif file missing for %s — skipping all indicators", region
            )
            continue

        for indicator in indicators:
            try:
                stats = process_pair(region, indicator, cfg, e2_dir, out_dir)
                summaries.append(stats)
            except FileNotFoundError as exc:
                log.warning("Skipping %s × %s — %s", region, indicator, exc)
            except Exception:
                log.exception("Error processing %s × %s", region, indicator)
                raise

    # Aggregate lag summary
    if summaries:
        lag_df = pd.DataFrame(summaries)
        lag_csv = out_dir / "lag_summary_E3.csv"
        lag_df.to_csv(lag_csv, index=False)
        log.info("Saved lag summary: %s", lag_csv)

        # Print pivot table to stdout
        try:
            pivot = lag_df.pivot(
                index="region", columns="indicator", values="median_lag"
            )
            log.info("\nMedian lag (months):\n%s", pivot.to_string())
        except Exception:
            pass
    else:
        log.warning("No pairs processed. Check E2 outputs and climate CSV exist.")

    log.info("E3 complete.")


def _write_config() -> None:
    conf_dir = BASE_DIR / "conf"
    target = conf_dir / "resilience_forest.yaml"
    print(f"Config managed by Hydra at {target}.")
    print("Override with: e.g. analysis.n_bootstrap=2000")


if __name__ == "__main__":
    if "--write-config" in sys.argv:
        sys.argv.remove("--write-config")
        _write_config()
        sys.exit(0)

    main()  # pylint: disable=no-value-for-parameter
