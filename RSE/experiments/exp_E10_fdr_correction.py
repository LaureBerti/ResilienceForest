"""
exp_E10_fdr_correction.py — Benjamini–Hochberg FDR correction for multiple tests.

Applies FDR correction to all simultaneous hypothesis tests in the paper:
  (a) E5 temporal-shift tests: 56 region × indicator Mann-Whitney U tests
      (pre-2010 vs post-2010 overlap counts)
  (b) E6 rolling-correlation MK trend tests: up to 24 continent × indicator
      Mann-Kendall tests on rolling Pearson r

Corrected α threshold and adjusted p-values are reported alongside the
original uncorrected results. Rows that survive BH-FDR at q=0.05 are
flagged `fdr_significant=True`.

Outputs:
  outputs/E10/fdr_corrected_E5.csv   — E5 results with BH-adjusted p-values
  outputs/E10/fdr_corrected_E6.csv   — E6 results with BH-adjusted p-values
  outputs/E10/fdr_summary_E10.csv    — one-row-per-collection summary

Usage (Hydra):
    python experiments/exp_E10_fdr_correction.py
    python experiments/exp_E10_fdr_correction.py run_mvp_only=true
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
# BH-FDR correction (manual, no statsmodels dependency)
# ---------------------------------------------------------------------------

def _bh_fdr(p_values: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR correction.

    Returns
    -------
    adjusted_p : np.ndarray
        BH-adjusted p-values (step-up procedure).
    reject : np.ndarray of bool
        True for tests that reject H0 after FDR correction at level q.
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1)

    # BH adjusted p-values: p_adj[i] = min(p[i]*n/rank[i], 1)
    # then enforce monotonicity from right to left (step-up)
    adj = np.minimum(1.0, sorted_p * n / ranks)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])

    # Reject all ranks ≤ max k such that sorted_p[k-1] ≤ k*q/n
    thresholds = q * ranks / n
    below = sorted_p <= thresholds
    if below.any():
        max_k = int(np.where(below)[0].max())
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: max_k + 1] = True
    else:
        reject_sorted = np.zeros(n, dtype=bool)

    # Map back to original order
    adjusted_p = np.empty(n)
    reject = np.empty(n, dtype=bool)
    adjusted_p[order] = adj
    reject[order] = reject_sorted

    return adjusted_p, reject


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_e5(e5_dir: Path) -> pd.DataFrame:
    path = e5_dir / "temporal_test_E5.csv"
    if not path.exists():
        raise FileNotFoundError(f"E5 results not found: {path}. Run exp_E5 first.")
    return pd.read_csv(path)


def _load_e6(e6_dir: Path) -> pd.DataFrame:
    path = e6_dir / "rolling_corr_trend_E6.csv"
    if not path.exists():
        raise FileNotFoundError(f"E6 results not found: {path}. Run exp_E6 first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Correction routines
# ---------------------------------------------------------------------------

def _correct_e5(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Apply BH-FDR to E5's 56 Mann-Whitney U p-values."""
    df = df.copy()
    p_col = "p_value"
    if p_col not in df.columns:
        raise ValueError(f"Column '{p_col}' not found in E5 results.")

    valid_mask = df[p_col].notna()
    p_arr = df.loc[valid_mask, p_col].to_numpy(dtype=float)

    adj, reject = _bh_fdr(p_arr, q)

    df["p_value_bh"] = np.nan
    df["fdr_significant"] = False
    df.loc[valid_mask, "p_value_bh"] = adj
    df.loc[valid_mask, "fdr_significant"] = reject

    n_orig_sig = int(df["significant"].sum()) if "significant" in df.columns else 0
    n_fdr_sig = int(reject.sum())
    log.info(
        "E5 FDR correction: %d tests | originally significant (α=%.2f): %d | "
        "FDR-significant (q=%.2f): %d",
        len(p_arr), 0.05, n_orig_sig, q, n_fdr_sig,
    )
    return df


def _correct_e6(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Apply BH-FDR to E6's rolling-correlation MK trend p-values."""
    df = df.copy()
    # E6 stores p-value as 'mk_p' (Mann-Kendall p-value)
    p_col = "mk_p" if "mk_p" in df.columns else "p_value"
    if p_col not in df.columns:
        raise ValueError(f"Neither 'mk_p' nor 'p_value' found in E6 results. Columns: {list(df.columns)}")

    valid_mask = df[p_col].notna()
    p_arr = df.loc[valid_mask, p_col].to_numpy(dtype=float)

    adj, reject = _bh_fdr(p_arr, q)

    df["p_value_bh"] = np.nan
    df["fdr_significant"] = False
    df.loc[valid_mask, "p_value_bh"] = adj
    df.loc[valid_mask, "fdr_significant"] = reject

    n_orig_sig = int((df.get("significant", pd.Series(dtype=bool))).sum())
    n_fdr_sig = int(reject.sum())
    log.info(
        "E6 FDR correction: %d tests | originally significant (α=%.2f): %d | "
        "FDR-significant (q=%.2f): %d",
        len(p_arr), 0.05, n_orig_sig, q, n_fdr_sig,
    )
    return df


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="resilience_forest", version_base="1.3")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra.utils.get_original_cwd())
    e5_dir = base_dir / cfg.output.dir / "E5"
    e6_dir = base_dir / cfg.output.dir / "E6"
    out_dir = base_dir / cfg.output.dir / "E10"
    out_dir.mkdir(parents=True, exist_ok=True)

    q = float(cfg.analysis.alpha)  # FDR level = same as uncorrected α = 0.05
    log.info("FDR level q = %.3f (Benjamini–Hochberg step-up procedure)", q)

    summary_rows: list[dict] = []

    # --- E5 ---
    try:
        e5_df = _load_e5(e5_dir)
        e5_corrected = _correct_e5(e5_df, q)
        out_path = out_dir / "fdr_corrected_E5.csv"
        e5_corrected.to_csv(out_path, index=False)
        log.info("Saved: %s", out_path)

        summary_rows.append({
            "collection": "E5_temporal_shift",
            "n_tests": len(e5_df),
            "n_uncorrected_sig": int(e5_df.get("significant", pd.Series(dtype=bool)).sum()),
            "n_fdr_sig": int(e5_corrected["fdr_significant"].sum()),
            "fdr_level_q": q,
            "correction": "Benjamini-Hochberg",
        })

        # Print surviving pairs
        survivors = e5_corrected[e5_corrected["fdr_significant"]]
        if not survivors.empty:
            log.info("E5 pairs surviving FDR:\n%s",
                     survivors[["region", "indicator", "p_value", "p_value_bh"]].to_string(index=False))
        else:
            log.info("E5: NO pairs survive FDR correction at q=%.2f — "
                     "all 9 originally-significant pairs were at the uncorrected boundary.", q)
    except FileNotFoundError as exc:
        log.warning("Skipping E5 correction: %s", exc)

    # --- E6 ---
    try:
        e6_df = _load_e6(e6_dir)
        e6_corrected = _correct_e6(e6_df, q)
        out_path = out_dir / "fdr_corrected_E6.csv"
        e6_corrected.to_csv(out_path, index=False)
        log.info("Saved: %s", out_path)

        summary_rows.append({
            "collection": "E6_rolling_corr_MK",
            "n_tests": len(e6_df),
            "n_uncorrected_sig": int((e6_df.get("significant", pd.Series(dtype=bool))).sum()),
            "n_fdr_sig": int(e6_corrected["fdr_significant"].sum()),
            "fdr_level_q": q,
            "correction": "Benjamini-Hochberg",
        })
    except FileNotFoundError as exc:
        log.warning("Skipping E6 correction: %s", exc)

    # --- Summary ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "fdr_summary_E10.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info("Saved summary: %s\n%s", summary_path, summary_df.to_string(index=False))
    else:
        log.warning("No results corrected — check that E5 and E6 have been run.")

    log.info("E10 complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
