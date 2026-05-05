"""
exp_E4_baselines.py — Baseline method comparison for forest resilience paper.

Claim supported: C1 — matrix profile adds value over cross-correlation / wavelet.

For each region × indicator pair:
  1. Cross-correlation (Pearson, lags 0–24 months, Fisher z 95 % CI).
  2. Wavelet coherence (pycwt) — mean coherence in 6-, 12-, 24-month bands.
  3. Comparison table against E3 matrix-profile lag estimates.

Usage (Hydra):
    python experiments/exp_E4_baselines.py
    python experiments/exp_E4_baselines.py run_mvp_only=true
    python experiments/exp_E4_baselines.py output.dir=custom_outputs
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import signal
from scipy.stats import t as t_dist

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kndvi_mean(raw_dir: Path, region: str) -> pd.Series:
    """Return monthly mean kNDVI across all pixels for *region*."""
    path = raw_dir / "kndvi" / f"kndvi_{region}_2000_2025.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["kndvi"].mean().sort_index()


def _load_climate(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "climate" / "climate_indicators_2000_2025.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df


def _align(s1: pd.Series, s2: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join on date index and return aligned numpy arrays."""
    combined = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    return combined["a"].to_numpy(), combined["b"].to_numpy()


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def compute_xcorr(
    kndvi: pd.Series,
    indicator: pd.Series,
    max_lag: int = 24,
    alpha: float = 0.05,
) -> dict:
    """
    Compute normalised Pearson cross-correlation at lags 0…max_lag.

    Returns peak_lag, peak_r, ci_lower, ci_upper and the full lag-r array.
    Positive lag means kNDVI lags the climate indicator.
    """
    x, y = _align(kndvi, indicator)
    n = len(x)

    # Zero-mean, unit-variance
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    # scipy.signal.correlate — mode='full', then extract positive lags
    corr_full = signal.correlate(x, y, mode="full")
    # Normalise to [-1, 1]: divide by (n * std_x * std_y) but after z-score above
    # that equals n exactly
    corr_full /= n

    mid = len(corr_full) // 2
    # lags[0] = 0, lags[1..max_lag] = positive lags (kNDVI lags indicator)
    lags = np.arange(0, max_lag + 1)
    r_values = corr_full[mid : mid + max_lag + 1]

    peak_idx = int(np.argmax(np.abs(r_values)))
    peak_lag = int(lags[peak_idx])
    peak_r = float(r_values[peak_idx])

    # Fisher z-transformation 95 % CI
    z = np.arctanh(np.clip(peak_r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_crit = t_dist.ppf(1 - alpha / 2, df=max(n - 2, 1))
    ci_lower = float(np.tanh(z - z_crit * se))
    ci_upper = float(np.tanh(z + z_crit * se))

    return {
        "peak_lag": peak_lag,
        "peak_r": peak_r,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "lags": lags.tolist(),
        "r_values": r_values.tolist(),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Wavelet coherence
# ---------------------------------------------------------------------------

def compute_wavelet_coherence(
    kndvi: pd.Series,
    indicator: pd.Series,
    dt: float = 1.0,  # monthly sampling
    bands: dict | None = None,
) -> dict:
    """
    Compute wavelet transform coherence using pycwt.

    Returns mean coherence and dominant period for each target band.
    Also returns the raw WTC array and period vector for scalogram export.
    """
    try:
        import pycwt as wavelet  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pycwt is required for wavelet coherence: pip install pycwt") from exc

    if bands is None:
        bands = {"6m": 6.0, "12m": 12.0, "24m": 24.0}

    x, y = _align(kndvi, indicator)
    n = len(x)

    # Standardise
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    mother = wavelet.Morlet(6)
    s0 = 2 * dt
    dj = 1 / 12
    J = int(np.log2(n * dt / s0) / dj)

    # Cross-wavelet transform
    try:
        WCT, aWCT, coi, freq, sig95 = wavelet.wct(
            x, y, dt, dj=dj, s0=s0, J=J, sig=True, wavelet=mother, normalize=True
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("pycwt.wct failed (%s); returning NaN coherence.", exc)
        result = {"n": n, "band_coherence": {}, "dominant_period": None,
                  "WTC": None, "periods": None, "coi": None}
        return result

    periods = 1.0 / freq  # in months

    band_coherence: dict[str, float] = {}
    for band_name, target_period in bands.items():
        # Find period indices within ±50 % of target
        lo, hi = target_period * 0.5, target_period * 1.5
        idx = np.where((periods >= lo) & (periods <= hi))[0]
        if idx.size == 0:
            band_coherence[band_name] = float("nan")
        else:
            band_coherence[band_name] = float(np.nanmean(WCT[idx, :]))

    # Dominant period: period with highest mean coherence over COI-masked region
    mean_coh_per_period = np.nanmean(WCT, axis=1)
    dom_idx = int(np.argmax(mean_coh_per_period))
    dominant_period = float(periods[dom_idx])

    return {
        "n": n,
        "band_coherence": band_coherence,
        "dominant_period": dominant_period,
        "WTC": WCT.tolist(),
        "periods": periods.tolist(),
        "coi": coi.tolist(),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    xcorr_results: dict,
    wavelet_results: dict,
    e3_lag_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (region, indicator) pair assemble:
      mp_lag | xcorr_lag | wavelet_dominant_period | agreement
    Agreement: |mp_lag - xcorr_lag| ≤ 2 months.
    """
    rows = []
    for (region, indicator), xr in xcorr_results.items():
        wr = wavelet_results.get((region, indicator), {})

        # Look up E3 matrix-profile lag
        mask = (e3_lag_summary["region"] == region) & (
            e3_lag_summary["indicator"] == indicator
        )
        mp_lag_row = e3_lag_summary[mask]
        if mp_lag_row.empty:
            mp_lag = float("nan")
        else:
            mp_lag = float(mp_lag_row["best_lag"].iloc[0])

        xcorr_lag = float(xr["peak_lag"])
        wav_dom = wr.get("dominant_period", float("nan"))
        if wav_dom is None:
            wav_dom = float("nan")

        agree = (
            "agree"
            if (not np.isnan(mp_lag)) and abs(mp_lag - xcorr_lag) <= 2
            else "disagree"
        )

        rows.append(
            {
                "region": region,
                "indicator": indicator,
                "mp_lag_months": mp_lag,
                "xcorr_peak_lag_months": xcorr_lag,
                "xcorr_peak_r": round(xr["peak_r"], 4),
                "xcorr_ci_lower_95": round(xr["ci_lower_95"], 4),
                "xcorr_ci_upper_95": round(xr["ci_upper_95"], 4),
                "wavelet_dominant_period_months": round(wav_dom, 1),
                "wavelet_coh_6m": round(wr.get("band_coherence", {}).get("6m", float("nan")), 4),
                "wavelet_coh_12m": round(wr.get("band_coherence", {}).get("12m", float("nan")), 4),
                "wavelet_coh_24m": round(wr.get("band_coherence", {}).get("24m", float("nan")), 4),
                "lag_agreement": agree,
                "n_obs": xr["n"],
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
    out_dir = base_dir / cfg.output.dir / "E4"
    out_dir.mkdir(parents=True, exist_ok=True)

    indicators = ["AO", "ENSO", "NAO", "PNA"]
    continents: list[str] = list(cfg.regions.continents)
    subregions: list[str] = list(cfg.regions.subregions)
    all_regions = continents + subregions

    if cfg.run_mvp_only:
        log.info("MVP mode: processing africa × AO only.")
        all_regions = ["africa"]
        indicators = ["AO"]

    # Load climate once
    climate_df = _load_climate(raw_dir)

    # Load E3 lag summary (may not exist in MVP)
    e3_lag_path = base_dir / cfg.output.dir / "E3" / "lag_summary_E3.csv"
    if e3_lag_path.exists():
        e3_lag_summary = pd.read_csv(e3_lag_path)
    else:
        log.warning("E3 lag summary not found at %s; mp_lag will be NaN.", e3_lag_path)
        e3_lag_summary = pd.DataFrame(columns=["region", "indicator", "best_lag"])

    xcorr_results: dict = {}
    wavelet_results: dict = {}

    for region in all_regions:
        kndvi_path = raw_dir / "kndvi" / f"kndvi_{region}_2000_2025.parquet"
        if not kndvi_path.exists():
            log.warning("kNDVI file missing for region=%s, skipping.", region)
            continue

        kndvi_mean = _load_kndvi_mean(raw_dir, region)

        for indicator in indicators:
            if indicator not in climate_df.columns:
                log.warning("Indicator %s not in climate data, skipping.", indicator)
                continue

            ind_series = climate_df[indicator].dropna()
            key = (region, indicator)

            log.info("Processing %s × %s ...", region, indicator)

            # --- Cross-correlation ---
            xr = compute_xcorr(
                kndvi_mean,
                ind_series,
                max_lag=24,
                alpha=cfg.analysis.alpha,
            )
            xcorr_results[key] = xr

            # Save per-pair CSV
            xcorr_df = pd.DataFrame({"lag": xr["lags"], "r": xr["r_values"]})
            xcorr_df["peak_lag"] = xr["peak_lag"]
            xcorr_df["peak_r"] = xr["peak_r"]
            xcorr_df["ci_lower_95"] = xr["ci_lower_95"]
            xcorr_df["ci_upper_95"] = xr["ci_upper_95"]
            xcorr_df.to_csv(out_dir / f"xcorr_{region}_{indicator}.csv", index=False)

            # --- Wavelet coherence ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wr = compute_wavelet_coherence(kndvi_mean, ind_series)
            wavelet_results[key] = wr

            # Save wavelet coherence summary (WTC array is too large for CSV)
            wav_rows = [
                {
                    "region": region,
                    "indicator": indicator,
                    "dominant_period_months": wr.get("dominant_period"),
                    **{f"coh_{k}": v for k, v in wr.get("band_coherence", {}).items()},
                    "n_obs": wr.get("n"),
                }
            ]
            pd.DataFrame(wav_rows).to_csv(
                out_dir / f"wavelet_coherence_{region}_{indicator}.csv", index=False
            )

            # Save scalogram data as parquet if available
            if wr.get("WTC") is not None:
                try:
                    periods = wr["periods"]
                    wtc_arr = np.array(wr["WTC"])
                    times = np.arange(wtc_arr.shape[1])
                    wtc_df = pd.DataFrame(
                        wtc_arr, index=pd.Index(periods, name="period_months")
                    )
                    wtc_df.columns = [f"t{i}" for i in times]
                    wtc_df.to_parquet(
                        out_dir / f"scalogram_{region}_{indicator}.parquet"
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not save scalogram for %s×%s: %s", region, indicator, exc)

    # Build and save comparison table
    comparison = build_comparison_table(xcorr_results, wavelet_results, e3_lag_summary)
    comparison.to_csv(out_dir / "comparison_table_E4.csv", index=False)
    log.info("Saved comparison_table_E4.csv with %d rows.", len(comparison))

    # Summary stats
    agree_pct = (comparison["lag_agreement"] == "agree").mean() * 100
    log.info(
        "Lag agreement (|mp_lag - xcorr_lag| ≤ 2 months): %.1f %% of pairs.", agree_pct
    )


if __name__ == "__main__":
    main()
