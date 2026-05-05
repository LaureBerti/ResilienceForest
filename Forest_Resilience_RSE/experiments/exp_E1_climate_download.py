"""
E1 — Climate indicator download from NOAA CPC.

Downloads monthly time series for AO, NAO, PNA, and ENSO (ONI) from the
NOAA Climate Prediction Center (CPC).  Aligns all indicators to a common
monthly DatetimeIndex and writes:

    data/raw/climate/<indicator>_monthly.csv   (one file per indicator)
    data/raw/climate/climate_indicators_<start>_<end>.csv  (combined)
    data/raw/climate/data_card_climate.md      (data card)

ASCII format notes (verified 2026-04)
--------------------------------------
AO  : long format — YEAR  MONTH  VALUE  (one row per month)
NAO : same long format as AO
PNA : same long format as AO
ENSO (ONI): SEAS  YR  TOTAL  ANOM  (season codes DJF…NDJ, one row per season)

Usage (Hydra)
-------------
    python experiments/exp_E1_climate_download.py
    python experiments/exp_E1_climate_download.py run_mvp_only=true
    python experiments/exp_E1_climate_download.py --write-config

References
----------
NOAA CPC: https://www.cpc.ncep.noaa.gov
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root, not CWD, so Hydra output dir is safe)
# ---------------------------------------------------------------------------

# Repository root = parent of experiments/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONF_PATH = _REPO_ROOT / "conf" / "resilience_forest.yaml"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDICATOR_NAMES = ["AO", "NAO", "PNA", "ENSO"]

# Maps each 3-letter ONI season code to the centre month number.
# DJF 1950 = Dec-49/Jan-50/Feb-50 → centre Jan 1950 → month 1.
_SEASON_TO_MONTH: dict[str, int] = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _fetch_url(url: str, timeout: int = 60) -> str:
    """Fetch a URL and return the response body as a string.

    Raises a descriptive RuntimeError on failure so the caller can decide
    whether to abort or skip this indicator.
    """
    log.info("Fetching: %s", url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_long_format(raw: str, indicator: str) -> pd.Series:
    """Parse AO / NAO / PNA long ASCII format: YEAR  MONTH  VALUE per line.

    Missing values are represented as -9999 or -99.99.
    Returns a pd.Series with a monthly PeriodIndex.
    """
    records: list[tuple[str, float]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 3:
            log.debug("%s: skipping short line: %s", indicator, line)
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            value = float(parts[2])
        except (ValueError, IndexError):
            log.debug("%s: could not parse line: %s", indicator, line)
            continue
        if value <= -99.0:
            value = float("nan")
        records.append((f"{year}-{month:02d}", value))

    if not records:
        raise ValueError(f"No data parsed for {indicator} (long format).")

    index = pd.PeriodIndex([r[0] for r in records], freq="M")
    series = pd.Series([r[1] for r in records], index=index, name=indicator, dtype=float)
    return series


def _parse_oni_format(raw: str) -> pd.Series:
    """Parse ENSO ONI ASCII format: SEAS  YR  TOTAL  ANOM per line.

    SEAS is a 3-letter season code (DJF … NDJ); we map it to its centre
    month via _SEASON_TO_MONTH.  We use the ANOM column (col index 3).
    Returns a pd.Series with a monthly PeriodIndex.
    """
    records: list[tuple[str, float]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("SEAS") or line.startswith("YR"):
            continue
        parts = line.split()
        if len(parts) < 4:
            log.debug("ONI: skipping short line: %s", line)
            continue
        seas = parts[0].upper()
        if seas not in _SEASON_TO_MONTH:
            log.debug("ONI: unknown season code %s, skipping", seas)
            continue
        try:
            year = int(parts[1])
            anom = float(parts[3])
        except (ValueError, IndexError):
            log.debug("ONI: could not parse line: %s", line)
            continue
        if anom <= -99.0:
            anom = float("nan")
        month = _SEASON_TO_MONTH[seas]
        records.append((f"{year}-{month:02d}", anom))

    if not records:
        raise ValueError("No data parsed for ENSO (ONI format).")

    index = pd.PeriodIndex([r[0] for r in records], freq="M")
    series = pd.Series([r[1] for r in records], index=index, name="ENSO", dtype=float)
    return series


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def download_indicator(
    indicator: str,
    url: str,
    start: str,
    end: str,
) -> Optional[pd.Series]:
    """Download and parse a single climate indicator.

    Parameters
    ----------
    indicator : str
        One of AO, NAO, PNA, ENSO.
    url : str
        Source URL from Hydra config.
    start, end : str
        ISO date strings "YYYY-MM-DD" for filtering.

    Returns
    -------
    pd.Series with monthly PeriodIndex, or None on failure.
    """
    try:
        raw = _fetch_url(url)
    except RuntimeError as exc:
        log.error("Download failed for %s: %s", indicator, exc)
        return None

    try:
        if indicator in ("AO", "NAO", "PNA"):
            series = _parse_long_format(raw, indicator)
        else:
            series = _parse_oni_format(raw)
    except ValueError as exc:
        log.error("Parse failed for %s: %s", indicator, exc)
        return None

    # Restrict to analysis window
    start_period = pd.Period(start[:7], freq="M")
    end_period = pd.Period(end[:7], freq="M")
    series = series.loc[start_period:end_period]

    n_missing = series.isna().sum()
    log.info(
        "%s: %d months, %d missing values (%.1f%%)",
        indicator,
        len(series),
        n_missing,
        100.0 * n_missing / max(len(series), 1),
    )
    return series


def align_to_monthly_index(
    series_dict: Dict[str, pd.Series],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Align all indicator series to a common monthly DatetimeIndex.

    Missing months (download gaps or short series) are filled with NaN.
    The index is a DatetimeIndex at month-start frequency for compatibility
    with downstream pandas / statsmodels calls.
    """
    full_index = pd.period_range(start=start[:7], end=end[:7], freq="M")
    aligned: Dict[str, pd.Series] = {}
    for name, series in series_dict.items():
        if series is None or series.empty:
            aligned[name] = pd.Series(float("nan"), index=full_index, name=name)
        else:
            aligned[name] = series.reindex(full_index)

    df = pd.DataFrame(aligned)
    # Convert PeriodIndex to DatetimeIndex (start of month) for parquet compatibility
    df.index = df.index.to_timestamp(how="start")
    df.index.name = "date"
    return df


def write_data_card(
    climate_dir: Path,
    indicators: list[str],
    sources: Dict[str, str],
    start: str,
    end: str,
) -> None:
    """Write a Markdown data card documenting provenance."""
    access_date = date.today().isoformat()
    lines = [
        "# Data Card — NOAA CPC Climate Indices",
        "",
        f"**Date accessed:** {access_date}",
        f"**Analysis window:** {start[:7]} to {end[:7]}",
        f"**Temporal resolution:** Monthly",
        "",
        "## Indicators",
        "",
    ]
    descriptions = {
        "AO":   ("Arctic Oscillation (AO)",
                 "Normalised monthly index. Positive = stronger polar vortex."),
        "NAO":  ("North Atlantic Oscillation (NAO)",
                 "Normalised monthly index. Positive = stronger westerlies over N. Atlantic."),
        "PNA":  ("Pacific–North American pattern (PNA)",
                 "Normalised monthly index. Positive = amplified ridge over W. North America."),
        "ENSO": ("El Niño–Southern Oscillation (ENSO / ONI)",
                 "Oceanic Niño Index — 3-month running mean of ERSST v5 anomalies in Niño-3.4 "
                 "region (5°N–5°S, 120–170°W). Base period: 1991–2020."),
    }
    for ind in indicators:
        name, desc = descriptions.get(ind, (ind, ""))
        lines += [
            f"### {name}",
            "",
            f"**URL:** {sources[ind]}",
            f"**Description:** {desc}",
            "**Format:** NOAA CPC ASCII",
            "**License:** Public domain (NOAA/CPC)",
            "**Missing value sentinel:** -9999 or -99.99 → replaced with NaN",
            "",
        ]
    lines += [
        "## Processing",
        "",
        "- Downloaded via `urllib.request` (no authentication required).",
        "- AO / NAO / PNA: wide format (year + 12 monthly columns) parsed row by row.",
        "- ENSO (ONI): long format (YR MON TOTAL CLIMO ANOM); ANOM column retained.",
        "- All series aligned to a complete monthly DatetimeIndex; gaps filled with NaN.",
        "- Combined output: `climate_indicators_2000_2025.csv`.",
        "",
        "## Citation",
        "",
        "NOAA Climate Prediction Center. Monthly climate indices. "
        "Retrieved from https://www.cpc.ncep.noaa.gov",
    ]
    card_path = climate_dir / "data_card_climate.md"
    card_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Data card written to %s", card_path)


# ---------------------------------------------------------------------------
# Config write helper
# ---------------------------------------------------------------------------


def _write_default_config() -> None:
    """Dump the default Hydra config to conf/resilience_forest.yaml.

    Reads the existing file (if any) and prints it; does not overwrite the
    hand-edited config.  Follows the --write-config convention used by
    Restauration/foremost.py.
    """
    if _CONF_PATH.exists():
        print(f"Config already exists at {_CONF_PATH}")
        print(_CONF_PATH.read_text(encoding="utf-8"))
    else:
        print(f"Config not found at {_CONF_PATH}.")
        print("Create conf/resilience_forest.yaml before running this script.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(_CONF_PATH.parent),
    config_name=_CONF_PATH.stem,
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("=== E1 Climate Download ===")
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    run_mvp = bool(cfg.run_mvp_only)
    if run_mvp:
        log.warning("MVP mode: downloading 2000-2005 only.")

    start_date: str = cfg.gee.start_date          # "2000-01-01"
    end_date: str   = "2005-12-31" if run_mvp else cfg.gee.end_date  # "2024-12-31"

    # Resolve output directory relative to repo root (not Hydra's cwd).
    climate_dir = _REPO_ROOT / cfg.data.climate_dir
    climate_dir.mkdir(parents=True, exist_ok=True)

    sources: Dict[str, str] = dict(cfg.climate_sources)

    series_dict: Dict[str, Optional[pd.Series]] = {}
    for indicator in INDICATOR_NAMES:
        url = sources.get(indicator)
        if url is None:
            log.error("No URL configured for indicator: %s", indicator)
            series_dict[indicator] = None
            continue
        series = download_indicator(indicator, url, start_date, end_date)
        series_dict[indicator] = series

        # Save per-indicator CSV (even if partial)
        if series is not None and not series.empty:
            out_path = climate_dir / f"{indicator}_monthly.csv"
            # DatetimeIndex for CSV (convert from PeriodIndex)
            ts_index = series.index.to_timestamp(how="start")
            df_ind = pd.DataFrame({indicator: series.values}, index=ts_index)
            df_ind.index.name = "date"
            df_ind.to_csv(out_path)
            log.info("Saved %s → %s", indicator, out_path)
        else:
            log.warning("No data saved for %s.", indicator)

    # Align all indicators to a common monthly index.
    combined = align_to_monthly_index(
        {k: v for k, v in series_dict.items()},
        start_date,
        end_date,
    )

    combined_path = climate_dir / f"climate_indicators_{start_date[:4]}_{end_date[:4]}.csv"
    combined.to_csv(combined_path)
    log.info("Combined table saved → %s  (%d rows × %d cols)",
             combined_path, len(combined), combined.shape[1])

    # Write per-IFL-period slices so each MODIS period has a matching climate file.
    # Periods are read from the config; fall back to hard-coded defaults if absent.
    ifl_periods: Dict[str, tuple] = {}
    try:
        for yr, bounds in cfg.modis.ifl_periods.items():
            ifl_periods[str(yr)] = (str(bounds.start), str(bounds.end))
    except Exception:
        ifl_periods = {
            "2000": ("2000-01-01", "2012-12-31"),
            "2013": ("2013-01-01", "2015-12-31"),
            "2016": ("2016-01-01", "2019-12-31"),
            "2020": ("2020-01-01", "2024-12-31"),
            "2025": ("2025-01-01", "2025-12-31"),
        }

    for ifl_yr, (p_start, p_end) in ifl_periods.items():
        slice_df = combined.loc[p_start:p_end]
        if slice_df.empty:
            log.warning("No climate data for IFL %s period (%s–%s).", ifl_yr, p_start, p_end)
            continue
        slice_path = climate_dir / f"climate_indicators_ifl{ifl_yr}.csv"
        slice_df.to_csv(slice_path)
        log.info("IFL %s period climate saved → %s  (%d rows)", ifl_yr, slice_path.name, len(slice_df))

    # Validate: check value ranges (climate indices typically in [-5, 5]).
    for col in combined.columns:
        col_data = combined[col].dropna()
        if col_data.empty:
            log.warning("Column %s is entirely NaN — check download.", col)
            continue
        out_of_range = ((col_data < -10) | (col_data > 10)).sum()
        if out_of_range > 0:
            log.warning(
                "%s: %d values outside [-10, 10] — possible parse error.",
                col, out_of_range,
            )
        log.info(
            "%s: min=%.3f  max=%.3f  mean=%.3f",
            col, col_data.min(), col_data.max(), col_data.mean(),
        )

    # Write data card.
    write_data_card(climate_dir, INDICATOR_NAMES, sources, start_date, end_date)

    log.info("=== E1 Climate Download complete ===")


# ---------------------------------------------------------------------------
# CLI entry point (handles --write-config before Hydra takes over)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--write-config", action="store_true",
                        help="Print the default config and exit.")
    known, remaining = parser.parse_known_args()

    if known.write_config:
        _write_default_config()

    # Pass remaining args to Hydra (sys.argv must not include --write-config).
    sys.argv = [sys.argv[0]] + remaining
    main()
