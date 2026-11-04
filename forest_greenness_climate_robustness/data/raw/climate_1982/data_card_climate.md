# Data Card — NOAA CPC Climate Indices

**Date accessed:** 2026-11-03
**Analysis window:** 1982-01 to 2024-12
**Temporal resolution:** Monthly

## Indicators

### Arctic Oscillation (AO)

**URL:** https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii
**Description:** Normalised monthly index. Positive = stronger polar vortex.
**Format:** NOAA CPC ASCII
**License:** Public domain (NOAA/CPC)
**Missing value sentinel:** -9999 or -99.99 → replaced with NaN

### North Atlantic Oscillation (NAO)

**URL:** https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii
**Description:** Normalised monthly index. Positive = stronger westerlies over N. Atlantic.
**Format:** NOAA CPC ASCII
**License:** Public domain (NOAA/CPC)
**Missing value sentinel:** -9999 or -99.99 → replaced with NaN

### Pacific–North American pattern (PNA)

**URL:** https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii
**Description:** Normalised monthly index. Positive = amplified ridge over W. North America.
**Format:** NOAA CPC ASCII
**License:** Public domain (NOAA/CPC)
**Missing value sentinel:** -9999 or -99.99 → replaced with NaN

### El Niño–Southern Oscillation (ENSO / ONI)

**URL:** https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
**Description:** Oceanic Niño Index — 3-month running mean of ERSST v5 anomalies in Niño-3.4 region (5°N–5°S, 120–170°W). Base period: 1991–2020.
**Format:** NOAA CPC ASCII
**License:** Public domain (NOAA/CPC)
**Missing value sentinel:** -9999 or -99.99 → replaced with NaN

## Processing

- Downloaded via `urllib.request` (no authentication required).
- AO / NAO / PNA: wide format (year + 12 monthly columns) parsed row by row.
- ENSO (ONI): long format (YR MON TOTAL CLIMO ANOM); ANOM column retained.
- All series aligned to a complete monthly DatetimeIndex; gaps filled with NaN.
- Combined output: `climate_indicators_2000_2025.csv`.

## Citation

NOAA Climate Prediction Center. Monthly climate indices. Retrieved from https://www.cpc.ncep.noaa.gov