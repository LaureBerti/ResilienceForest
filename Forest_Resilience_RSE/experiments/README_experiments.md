# Experiments — Analysis of Motifs of Intact Forest Resilience and Global Climate Variables

**Authors:** Pius N. Nwachukwu, Laure Berti-Équille (IRD, ESPACE-DEV)


---

## Setup

### Step 1 — E1 data prep: IFL conversion + climate download (Python 3.12)

Uses the same `.venv_experiments` as the full pipeline. No LLVM required.

```bash
# Use the same Python 3.12 venv (create it first if needed — see Step 2 below)
source .venv_experiments/bin/activate
pip install -r experiments/requirements_E1_dataprep.txt

# 1a. Convert IFL GeoPackages to GeoParquet (one-time)
python experiments/exp_E1_convert_ifl_gpkg.py

# 1b. Download climate indices from NOAA CPC (AO, NAO, PNA, ENSO)
python experiments/exp_E1_climate_download.py
```

Outputs:
- `data/raw/ifl/IFL_<year>.parquet` — GeoParquet masks for each IFL vintage
- `data/raw/climate/AO_monthly.csv`, `NAO_monthly.csv`, `PNA_monthly.csv`, `ENSO_monthly.csv`
- `data/raw/climate/climate_indicators_2000_2025.csv` — combined table (312 rows, 0 missing values)

**Python version:** Python 3.12 required. Do not use Python 3.13 — the venv
symlinks will resolve incorrectly on macOS if pyenv has 3.13 active (see note
in Step 2).

### Step 2 — Full experiment pipeline (requires Python 3.12)

`stumpy` (matrix profile) depends on `numba` → `llvmlite`, which requires LLVM
and has no pre-built wheel for Python 3.13. Use Python 3.12 for all E2–E8 scripts.

```bash
# Install Python 3.12 via pyenv if not present
pyenv install 3.12.9
pyenv local 3.12.9           # sets .python-version in project root

# Create the experiment venv with Python 3.12
python3.12 -m venv .venv_experiments
source .venv_experiments/bin/activate
```

#### macOS 26 (Darwin 25) — llvmlite must be built from source

On macOS 26 (Sequoia / Darwin 25+), PyPI has no pre-built `llvmlite` wheel.
`numba 0.65.x` pulls in `llvmlite 0.47.x`, which requires exactly **LLVM 20**.
Homebrew's default `llvm` formula is 22 (too new); `llvm@14` is too old.

```bash
# Install LLVM 20 (one-time, ~350 MB)
brew install llvm@20

# Build llvmlite against LLVM 20 — must set both env vars
CMAKE_PREFIX_PATH=$(brew --prefix llvm@20) \
LLVM_CONFIG=$(brew --prefix llvm@20)/bin/llvm-config \
pip install llvmlite==0.47.0

# Then install everything else
pip install -r experiments/requirements_experiments.txt
```

On macOS 14 or earlier, `pip install -r experiments/requirements_experiments.txt` works directly without the LLVM step.

**Python version:** 3.12 required for E2–E8. Python 3.13 works for E1 data prep only.

---

## Datasets

All datasets must be present before running paper-result experiments.
Scripts fall back to synthetic stand-ins when `run_mvp_only=true` — smoke tests only, NOT paper results.

| Dataset | Used by | Manual download? | Path | Source |
|---------|---------|-----------------|------|--------|
| MODIS MOD13A3 V6.1 | E1, E2, E6, E7, E8 | Via GEE (see below) | `data/raw/kndvi/kndvi_{region}_2000_2024.parquet` | GEE: `MODIS/061/MOD13A3` |
| IFL 2000 / 2013 / 2016 / 2020 | E1 | Yes (already downloaded) | `data/raw/ifl/IFL_<year>.gpkg` | https://www.intactforests.org/data.ifl.html |
| AO monthly index | E1, E3–E6 | Auto (downloaded by E1) | `data/raw/climate/AO_monthly.csv` | NOAA CPC |
| NAO monthly index | E1, E3–E6 | Auto (downloaded by E1) | `data/raw/climate/NAO_monthly.csv` | NOAA CPC |
| PNA monthly index | E1, E3–E6 | Auto (downloaded by E1) | `data/raw/climate/PNA_monthly.csv` | NOAA CPC |
| ENSO ONI index | E1, E3–E6 | Auto (downloaded by E1) | `data/raw/climate/ENSO_monthly.csv` | NOAA CPC |

### Step 0 — Convert IFL GeoPackages to GeoParquet (one-time)

The IFL `.gpkg` files in `data/raw/ifl/` must be converted to GeoParquet for fast
loading and spatial joins in downstream experiments.

```bash
source .venv_experiments/bin/activate
python experiments/exp_E1_convert_ifl_gpkg.py          # converts all 5 vintages
python experiments/exp_E1_convert_ifl_gpkg.py --years 2020  # IFL 2020 only
```

Output: `data/raw/ifl/IFL_<year>.parquet` (Snappy-compressed GeoParquet, EPSG:4326).
This is a one-time step — re-run only if the source `.gpkg` files change.

---

### MODIS kNDVI — local download via NASA Earthdata (recommended)

`exp_E1_earthaccess.py` replaces the GEE JS approach. No GEE account required.
Requires a free NASA Earthdata account: https://urs.earthdata.nasa.gov

**IFL–MODIS period matching (time-varying mask)**

| IFL vintage | MODIS period | Rationale |
|-------------|-------------|-----------|
| IFL 2000 | 2000-01 – 2012-12 | IFL valid before first update |
| IFL 2013 | 2013-01 – 2015-12 | IFL valid 2013–2015 |
| IFL 2016 | 2016-01 – 2019-12 | IFL valid 2016–2019 |
| IFL 2020 | 2020-01 – 2024-12 | IFL valid 2020–2024 |

```bash
source .venv_experiments/bin/activate
pip install earthaccess pyhdf       # one-time, not in requirements_experiments.txt yet

# Full run (all 14 regions, all IFL periods — downloads ~50–80 GB of HDF4 tiles)
python experiments/exp_E1_earthaccess.py

# Specific IFL years or regions
python experiments/exp_E1_earthaccess.py --ifl-years 2020 --regions amazon congo
python experiments/exp_E1_earthaccess.py --mvp       # smoke test: Africa + Amazon, 1 month
```

Climate data is also sliced per IFL period automatically during the climate download step:
```bash
python experiments/exp_E1_climate_download.py
# Outputs: data/raw/climate/climate_indicators_ifl<year>.csv  (one per IFL vintage)
```

**Disk space:** HDF4 tiles are cached in `data/raw/modis/` (~500 MB per month globally at 1 km).
Use `--stride 5` (default) for ~5 km effective resolution; delete tiles after Parquet export to reclaim space.

---

### GEE data download (MODIS kNDVI — alternative, requires GEE account)

MODIS data is downloaded via Google Earth Engine. This requires a GEE account.

**No shapefile upload needed.** The GEE script uses public IFL raster assets:
- `projects/ee-potapovpeter/assets/IFL/IFL_2025m` — pixels intact as of 2025
- `projects/ee-potapovpeter/assets/IFL/IFL_2025_loss` — loss periods encoded 1–4

IFL 2020 extent is derived inside `exp_E1_gee.js` as: pixels intact in 2025 **or** lost only in 2020–2025.

**Run the GEE script:**
1. Open https://code.earthengine.google.com/
2. Paste the contents of `experiments/exp_E1_gee.js`
3. Set `var IFL_YEAR = 2020` and `var MVP_MODE = false` for full runs
4. Click **Run** — GEE submits one export task per region to your Google Drive (folder: `ForestResilience_E1`)
5. Monitor tasks at https://code.earthengine.google.com/tasks
6. Once complete, note the Google Drive file IDs for each exported CSV

Expected GEE export files (one CSV per region, pixel-level rows):
```
kndvi_africa_pixels_ifl2020.csv
kndvi_asia_pixels_ifl2020.csv
kndvi_europe_pixels_ifl2020.csv
kndvi_north_america_pixels_ifl2020.csv
kndvi_south_america_pixels_ifl2020.csv
kndvi_australia_oceania_pixels_ifl2020.csv
kndvi_amazon_pixels_ifl2020.csv
kndvi_congo_pixels_ifl2020.csv
kndvi_boreal_siberia_pixels_ifl2020.csv
kndvi_southeast_asia_pixels_ifl2020.csv
kndvi_canadian_boreal_pixels_ifl2020.csv
kndvi_scandinavian_boreal_pixels_ifl2020.csv
kndvi_papua_new_guinea_pixels_ifl2020.csv
kndvi_russian_far_east_pixels_ifl2020.csv
```

Each CSV has one row per pixel per month: `region, date, pixel_id, latitude, longitude, kNDVI`.

### Convert GEE CSVs to Parquet

After downloading CSVs from Google Drive, convert them to Parquet for efficient downstream use.
This step also auto-detects and corrects the MODIS integer scale (values 0–10 000 → divided by 10 000).

**1. Add your Google Drive file IDs** to `experiments/exp_E1_convert_to_parquet.py`:
```python
GDRIVE_FILE_IDS: dict[str, str] = {
    "africa":         "1M03g5BWAkyuFHKsrPPJJ2aQPDSGHxxCsp",  # your file ID
    "asia":           "<file_id_from_drive_share_link>",
    # ... one entry per region
}
```
Get each file ID from the share link: `drive.google.com/open?id=<FILE_ID>`

**2. Run the conversion:**
```bash
source .venv_experiments/bin/activate
python experiments/exp_E1_convert_to_parquet.py
# MVP smoke test (first 10 000 rows only):
python experiments/exp_E1_convert_to_parquet.py --mvp
```

Output: `data/raw/kndvi/kndvi_{region}_2000_2024.parquet` (Snappy-compressed).
Downloaded CSVs are cached in `data/raw/kndvi/_downloads/` — delete to re-download.

---

## Running Experiments

### Critical path (run in this order)

```
GEE export (exp_E1_gee.js — in browser)
  → exp_E1_convert_to_parquet.py  (download from Drive + fix scale + save Parquet)
  → E1 climate download  →  E2 (matrix profile) → E3 (overlap + lags)
                                                 → E4 (baselines)  [parallel with E2]
                            E3 → E5 (temporal test)
                            E2 → E6 (rolling correlation)
                            E1 → E7 (change points)
                            E2 → E8 (motif evolution)
```



### Hydra config overrides

```bash
# Change subsequence length
source .venv_experiments/bin/activate
python experiments/exp_E2_matrix_profile.py matrix_profile.subsequence_length=12

# Change bootstrap resamples
python experiments/exp_E3_overlap.py analysis.n_bootstrap=500

# Change pre/post split year
python experiments/exp_E5_temporal_test.py analysis.pre_period_end="2012-01-01"

# Write default config to file
python experiments/exp_E2_matrix_profile.py --write-config
```

---

## Security Notes

- No credentials are stored in any script; GEE authentication uses `earthengine authenticate` (OAuth).
- Climate index downloads use HTTPS to NOAA CPC public endpoints; no authentication required.
- `.env` files are not used in this project.
