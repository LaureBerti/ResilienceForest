# How robust are satellite-detected links between forest greenness and climate variability? A sensitivity analysis of MODIS kNDVI in intact forests

Code, processed data and analysis outputs for the paper of the same title by
Pius N. Nwachukwu and Laure Berti-Equille (IRD, ESPACE-DEV, Montpellier, France).
The paper is under review.

The analysis relates monthly MODIS kNDVI anomalies (2000–2024) in 14 intact-forest zones to four
climate modes (AO, ENSO, NAO, PNA). It uses rolling correlations, matrix-profile recurrence (STUMPY),
motif and changepoint co-occurrence, critical-slowing-down indicators, a 0.5° grid analysis, a pixel
bootstrap and an AVHRR comparison. Each result is tested under two deseasonalisations (month-of-year
climatology and robust STL), with Hamed–Rao autocorrelation-corrected trend tests and
Benjamini–Hochberg false-discovery control.

## Contents

```
README.md
requirements_experiments.txt        Python packages for the analyses
requirements_E1_dataprep.txt        Python packages for data preparation
conf/resilience_forest.yaml         Hydra configuration (regions, parameters, paths)
experiments/                        analysis scripts (see table below)
data/raw/climate/                   CPC indices 2000–2025 (AO, NAO, PNA, ONI ERSST.v5)
data/raw/climate_1982/              CPC indices 1982–2024 (ONI ERSST.v6), used for AVHRR 1982–2015
data/processed/regional_monthly_kndvi_modis.csv
                                    regional pixel-mean kNDVI per month, with valid-pixel counts
```

The analysis outputs used in the paper (`outputs/`) and the pixel-level kNDVI series
(`data/raw/kndvi/kndvi_{region}_2000_2025.parquet`, 14 files, 1.6 GB) are on Zenodo,
https://doi.org/10.5281/zenodo.20039952 (`ForestResilience_code_data_outputs.zip` and
`ForestResilience_kndvi_pixels.zip`). Unzip them at the root of this folder. The pixel series are
needed to rerun the analyses from the start; the regional CSV and `outputs/` are enough to check
every reported number.

## Setup

Python 3.12, CPU only. No GPU or cloud resources are needed.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements_experiments.txt
```

All scripts use Hydra. Parameters are overridden on the command line, e.g.
`python experiments/exp_E2_matrix_profile.py matrix_profile.subsequence_length=36`.

## Data sources

| Data | Source | Used for |
|---|---|---|
| MODIS MOD13A3 V6.1, 2000–2024 | NASA LP DAAC, via Google Earth Engine (`MODIS/061/MOD13A3`) | kNDVI pixel series |
| Intact Forest Landscapes 2020 | https://www.intactforests.org | forest mask |
| AO, NAO, PNA, ONI | NOAA CPC, https://www.cpc.ncep.noaa.gov | climate modes |
| PKU GIMMS NDVI V1.2 | Zenodo, https://doi.org/10.5281/zenodo.8253971 (CC BY 4.0) | AVHRR comparison |

kNDVI = tanh(((NIR − red) / 2σ)²) with σ = 0.5, pixels with SummaryQA ≤ 1, IFL 2020 mask, 5 km
sampling. The PKU GIMMS archives (about 3.9 GB) are not included; download them with
`./experiments/download_E23_avhrr_pku.sh --modis`.

## Reproducing the data

1. In the Earth Engine code editor, run `experiments/exp_E1_gee.js` (upload the IFL 2020 extent as an
   asset first), then convert the exported CSVs with `python experiments/exp_E1_kndvi_from_csv.py` or
   `python experiments/exp_E1_convert_to_parquet.py`.
2. Download the climate indices: `python experiments/exp_E1_climate_download.py`
   (for 1982–2024: `gee.start_date=1982-01-01 gee.end_date=2024-12-31 data.climate_dir=data/raw/climate_1982`).

## Reproducing the analyses

Run from the root of this folder, in this order. Outputs are written to `outputs/E*/`.

| Step | Command | Result |
|---|---|---|
| E2 | `python experiments/exp_E2_matrix_profile.py` | matrix profiles, motifs |
| E3 | `python experiments/exp_E3_overlap.py` | kNDVI–climate motif overlap |
| E8 | `python experiments/exp_E8_motif_evolution.py` | MPI series and trends |
| E5 | `python experiments/exp_E5_temporal_test.py` | motif co-occurrence before and after 2010 |
| E6 | `python experiments/exp_E6_rolling_correlation.py` | 36-month rolling correlations |
| E13 | `python experiments/exp_E13_hamed_rao_mk.py` | Hamed–Rao trend tests of the rolling correlations |
| E9, E14, E15 | `python experiments/exp_E9_recent_trend.py`, `..._E14_e5_floor_analysis.py`, `..._E15_power_floor.py` | recent-versus-baseline contrast and attainable p-values |
| E10 | `python experiments/exp_E10_fdr_correction.py` | Benjamini–Hochberg correction |
| E7 | `python experiments/exp_E7_changepoint.py` | changepoints and ENSO-year null model |
| E11 | `python experiments/exp_E11_overlap_null_model.py` | motif-overlap permutation null |
| E18 | `python experiments/exp_E18_mpi_trend_correction.py` | Hamed–Rao MPI trends |
| E19 | `python experiments/exp_E19_deseason_sensitivity.py` | MPI trends under four anomaly definitions |
| E16 | `python experiments/exp_E16_resilience_benchmark.py` | MPI against TAC, λ(AR1) and variance |
| E17 | `python experiments/exp_E17_grid_mpi.py` | 0.5° grid MPI, cell trends, Moran's I |
| E20 | `python experiments/exp_E20_pixel_bootstrap.py` | pixel bootstrap (200 replicates, about 15 min) |
| E21 | `python experiments/exp_E21_csd_climate_regression.py` | resilience indicators regressed on the climate modes |
| E22 | `python experiments/exp_E22_m_sensitivity_corrected.py` | subsequence lengths 12–48 months |
| E22b | `python experiments/exp_E22b_rolling_window_summary.py` | rolling windows 24–60 months (needs the E6/E13 arms below) |
| E23 | `python experiments/exp_E23_avhrr_comparison.py` | AVHRR and MODIS comparison |

Sensitivity arms:

```bash
# STL anomaly for the rolling correlations
python experiments/exp_E6_rolling_correlation.py analysis.kndvi_anomaly=stl output.dir=outputs/sens_E6/stl_raw
python experiments/exp_E13_hamed_rao_mk.py output.dir=outputs/sens_E6/stl_raw
# rolling windows 24, 48, 60 months (w and a = climatology or stl)
python experiments/exp_E6_rolling_correlation.py analysis.rolling_window=24 analysis.kndvi_anomaly=climatology output.dir=outputs/sens_E6/win24_climatology
python experiments/exp_E13_hamed_rao_mk.py output.dir=outputs/sens_E6/win24_climatology
# variants that keep the annual cycle
python experiments/exp_E6_rolling_correlation.py analysis.kndvi_anomaly=rollmean12 output.dir=outputs/sens_E6/roll_raw
python experiments/exp_E6_rolling_correlation.py analysis.kndvi_anomaly=rollmean12 analysis.deseason_climate=true output.dir=outputs/sens_E6/roll_filt
python experiments/exp_E6_rolling_correlation.py analysis.deseason_climate=true output.dir=outputs/sens_E6/clim_filt
# stricter ENSO-year thresholds for the changepoints
python experiments/exp_E7_changepoint.py +e7.oni_threshold=1.0 output.dir=outputs/sens_E7/oni_1.0
# E21 variants
python experiments/exp_E21_csd_climate_regression.py +e21.window=60 +e21.out_dir=E21_w60
python experiments/exp_E21_csd_climate_regression.py '+e21.season_months=[5,6,7,8,9]' +e21.out_dir=E21_gs24
python experiments/exp_E21_csd_climate_regression.py '+e21.season_months=[5,6,7,8,9]' +e21.window=60 +e21.out_dir=E21_gs60
# E23 with the good AVHRR flag only
python experiments/exp_E23_avhrr_comparison.py '+e23.qc_keep=[0]' '+e23.qc_keep_modis=[990]' +e23.out_dir=E23_qc0
```

E22 reproduces E19 and E11 exactly at m = 24 (`outputs/E22/acceptance_E22.csv`), and E20 checks that
its replicate with unit weights reproduces E2 and E8 before resampling.

## License

CC BY 4.0.

## Funding

European Horizon project eco2adapt (grant agreement 101059498).

## Citation

Nwachukwu, P. N., Berti-Equille, L. How robust are satellite-detected links between forest greenness
and climate variability? A sensitivity analysis of MODIS kNDVI in intact forests. Under review.
