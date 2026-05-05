"""
exp_E13_hamed_rao_mk.py — Autocorrelation-corrected Mann-Kendall trend test.

Motivation
----------
Standard Mann-Kendall assumes exchangeability under H0.  Rolling 24-month
Pearson correlation windows share 23 of 24 observations, so adjacent values
have lag-1 autocorrelation ≈ 1 by construction.  This inflates the MK
variance estimate, making p-values anti-conservative and overstating
significance.

This experiment re-tests every (region × indicator) rolling correlation
series from E6 using:

1. Hamed & Rao (1998) modified MK — variance is inflated by the effective
   sample size correction for autocorrelation.  Applied via `pymannkendall`
   if the package is importable.

2. Block-bootstrap MK — if `pymannkendall` is not available, resample
   contiguous blocks (length = block_size) to preserve local autocorrelation
   and estimate the p-value empirically.

Inputs
------
outputs/E6/rolling_corr_{region}_{indicator}.csv
    columns: date, region, indicator, rolling_r, p_value_instantaneous

Outputs
-------
outputs/E13/hamed_rao_mk_E13.csv
    columns: region, indicator, mk_tau, n_obs,
             uncorrected_p, corrected_p,
             uncorrected_significant, corrected_significant, method

Usage
-----
    python experiments/exp_E13_hamed_rao_mk.py
    python experiments/exp_E13_hamed_rao_mk.py alpha=0.01
    python experiments/exp_E13_hamed_rao_mk.py block_size=48 n_bootstrap=10000

Hydra config keys (all have defaults, all overridable via CLI)
--------------------------------------------------------------
    alpha         : significance level (default 0.05)
    block_size    : block length for bootstrap fallback (default 36)
    n_bootstrap   : bootstrap iterations for fallback (default 5000)
    e6_dir        : path to E6 outputs, relative to project root
    out_dir       : path to E13 outputs, relative to project root
    seed          : RNG seed for reproducibility (default 42)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Hydra config (dataclass)
# ---------------------------------------------------------------------------

@dataclass
class E13Config:
    alpha: float = 0.05
    block_size: int = 36
    n_bootstrap: int = 5000
    e6_dir: str = "outputs/E6"
    out_dir: str = "outputs/E13"
    seed: int = 42


# ---------------------------------------------------------------------------
# Block-bootstrap Mann-Kendall (fallback when pymannkendall is unavailable)
# ---------------------------------------------------------------------------

def block_bootstrap_mk(
    series: np.ndarray,
    n_bootstrap: int = 5000,
    block_size: int = 36,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Block-bootstrap Mann-Kendall trend test.

    Parameters
    ----------
    series       : 1-D array of observations (NaN-free)
    n_bootstrap  : number of bootstrap replicates
    block_size   : length of each resampled block
    rng          : numpy Generator for reproducibility

    Returns
    -------
    obs_tau   : observed Kendall tau
    p_value   : two-tailed bootstrap p-value
    uncorrected_p : p-value from the standard scipy kendalltau
    """
    from scipy.stats import kendalltau  # noqa: PLC0415

    if rng is None:
        rng = np.random.default_rng()

    n = len(series)
    t = np.arange(n)
    obs_tau, uncorrected_p = kendalltau(t, series)

    max_start = n - block_size
    if max_start < 1:
        # Series shorter than block_size — fall back to simple permutation
        bootstrap_taus: list[float] = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(series)
            tau, _ = kendalltau(t, perm)
            bootstrap_taus.append(tau)
    else:
        bootstrap_taus = []
        for _ in range(n_bootstrap):
            indices: list[int] = []
            while len(indices) < n:
                start = int(rng.integers(0, max_start + 1))
                indices.extend(range(start, min(start + block_size, n)))
            bootstrapped = series[np.array(indices[:n])]
            tau, _ = kendalltau(t, bootstrapped)
            bootstrap_taus.append(tau)

    boot_arr = np.array(bootstrap_taus)
    p_value = float(np.mean(np.abs(boot_arr) >= np.abs(obs_tau)))
    # Clamp to avoid exact 0 (minimum resolution is 1/n_bootstrap)
    p_value = max(p_value, 1.0 / n_bootstrap)

    return float(obs_tau), p_value, float(uncorrected_p)


# ---------------------------------------------------------------------------
# Hamed-Rao MK via pymannkendall
# ---------------------------------------------------------------------------

def hamed_rao_mk(series: np.ndarray) -> tuple[float, float, float]:
    """
    Apply Hamed & Rao (1998) variance-corrected Mann-Kendall test.

    Returns
    -------
    tau           : Kendall tau
    corrected_p   : H-R corrected p-value
    uncorrected_p : standard MK p-value (original_test)
    """
    import pymannkendall as mk  # noqa: PLC0415

    hr = mk.hamed_rao_modification_test(series)
    orig = mk.original_test(series)
    return float(hr.Tau), float(hr.p), float(orig.p)


# ---------------------------------------------------------------------------
# Dispatch: choose method based on availability
# ---------------------------------------------------------------------------

def _try_import_pymannkendall() -> bool:
    try:
        import pymannkendall  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def run_corrected_mk(
    series: np.ndarray,
    use_hamed_rao: bool,
    block_size: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, str]:
    """
    Run corrected MK test.

    Returns
    -------
    tau, corrected_p, uncorrected_p, method_name
    """
    if use_hamed_rao:
        tau, corrected_p, uncorrected_p = hamed_rao_mk(series)
        return tau, corrected_p, uncorrected_p, "hamed_rao"
    else:
        tau, corrected_p, uncorrected_p = block_bootstrap_mk(
            series, n_bootstrap=n_bootstrap, block_size=block_size, rng=rng
        )
        return tau, corrected_p, uncorrected_p, f"block_bootstrap_b{block_size}_n{n_bootstrap}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())

    # Read E13-specific overrides merged on top of the base config.
    # We store them under cfg.e13 if present, otherwise fall back to defaults.
    e13_defaults = OmegaConf.structured(E13Config())
    e13_cfg = OmegaConf.merge(e13_defaults, cfg.get("e13", {}))

    alpha: float = float(e13_cfg.alpha)
    block_size: int = int(e13_cfg.block_size)
    n_bootstrap: int = int(e13_cfg.n_bootstrap)
    e6_dir: Path = base_dir / str(e13_cfg.e6_dir)
    out_dir: Path = base_dir / str(e13_cfg.out_dir)
    seed: int = int(e13_cfg.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    use_hamed_rao = _try_import_pymannkendall()
    method_label = "hamed_rao" if use_hamed_rao else f"block_bootstrap (b={block_size})"
    log.info("Correction method: %s", method_label)

    rng = np.random.default_rng(seed)

    # Discover all per-pair CSVs in E6 (exclude the trend summary file)
    csv_files = sorted(
        p for p in e6_dir.glob("rolling_corr_*.csv")
        if "trend" not in p.name
    )
    if not csv_files:
        raise FileNotFoundError(f"No rolling correlation CSVs found in {e6_dir}")

    log.info("Found %d rolling correlation CSVs in %s", len(csv_files), e6_dir)

    rows: list[dict] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if df.empty:
            log.warning("Empty file: %s — skipping.", csv_path.name)
            continue

        region = str(df["region"].iloc[0])
        indicator = str(df["indicator"].iloc[0])
        series_raw = df["rolling_r"].to_numpy()

        # Drop NaN before passing to MK
        series = series_raw[~np.isnan(series_raw)]
        n_obs = len(series)

        if n_obs < 4:
            log.warning("Too few observations (%d) for %s × %s — skipping.", n_obs, region, indicator)
            continue

        tau, corrected_p, uncorrected_p, method_used = run_corrected_mk(
            series,
            use_hamed_rao=use_hamed_rao,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

        rows.append(
            {
                "region": region,
                "indicator": indicator,
                "n_obs": n_obs,
                "mk_tau": round(tau, 4),
                "uncorrected_p": round(uncorrected_p, 6),
                "corrected_p": round(corrected_p, 6),
                "uncorrected_significant": bool(uncorrected_p < alpha),
                "corrected_significant": bool(corrected_p < alpha),
                "method": method_used,
            }
        )

    results = pd.DataFrame(rows).sort_values(["region", "indicator"]).reset_index(drop=True)
    out_path = out_dir / "hamed_rao_mk_E13.csv"
    results.to_csv(out_path, index=False)
    log.info("Saved corrected MK results: %s (%d rows)", out_path, len(results))

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    n_total = len(results)
    n_uncorrected = int(results["uncorrected_significant"].sum())
    n_corrected = int(results["corrected_significant"].sum())

    print("\n" + "=" * 60)
    print(f"E13 Autocorrelation-Corrected Mann-Kendall Summary")
    print(f"Correction method : {method_label}")
    print(f"Significance level: alpha = {alpha}")
    print(f"Total pairs tested: {n_total}")
    print("=" * 60)
    print(f"  Standard MK significant : {n_uncorrected:3d} / {n_total}")
    print(f"  Corrected MK significant: {n_corrected:3d} / {n_total}")
    print(f"  Pairs that LOSE significance after correction: {n_uncorrected - n_corrected}")
    print("=" * 60)

    # Pairs that lose significance
    lost = results[results["uncorrected_significant"] & ~results["corrected_significant"]]
    if lost.empty:
        print("\nNo pairs lose significance after correction.")
    else:
        print(f"\nPairs losing significance ({len(lost)}):")
        print(
            lost[["region", "indicator", "mk_tau", "uncorrected_p", "corrected_p"]]
            .to_string(index=False)
        )

    # Pairs that gain significance (unexpected)
    gained = results[~results["uncorrected_significant"] & results["corrected_significant"]]
    if not gained.empty:
        print(f"\nPairs gaining significance ({len(gained)}) [unexpected — check series]:")
        print(
            gained[["region", "indicator", "mk_tau", "uncorrected_p", "corrected_p"]]
            .to_string(index=False)
        )

    # Full table sorted by corrected_p
    print("\nFull results (sorted by corrected_p):")
    display_cols = ["region", "indicator", "n_obs", "mk_tau",
                    "uncorrected_p", "corrected_p",
                    "uncorrected_significant", "corrected_significant"]
    print(results.sort_values("corrected_p")[display_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
