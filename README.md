# ResilienceForest

Code for the paper

> Nwachukwu, P. N., Berti-Equille, L. **How robust are satellite-detected links between forest
> greenness and climate variability? A sensitivity analysis of MODIS kNDVI in intact forests.**
> Under review.

The paper relates monthly MODIS kNDVI anomalies (2000–2024) in 14 intact-forest zones to four climate
modes (AO, ENSO, NAO, PNA). It uses rolling correlations, matrix-profile recurrence, motif and
changepoint co-occurrence, critical-slowing-down indicators, a 0.5° grid analysis, a pixel bootstrap
and an AVHRR comparison. Each result is tested under two deseasonalisations, with autocorrelation-
corrected trend tests and false-discovery control.

## Repository

```
forest_greenness_climate_robustness/
  experiments/      analysis scripts (Python 3.12, Hydra)
  conf/             configuration
  data/raw/         CPC climate indices (2000–2025 and 1982–2024)
  data/processed/   regional monthly kNDVI series
  README.md         setup, data sources and run order
```

Setup, data sources and the full run order are in
[`forest_greenness_climate_robustness/README.md`](forest_greenness_climate_robustness/README.md).
Everything runs on a CPU; no GPU or cloud resources are needed.

## Data and outputs

The analysis outputs used in the paper and the 5 km pixel-level kNDVI series (1.6 GB) are on Zenodo:
https://doi.org/10.5281/zenodo.20039952

## License

CC BY 4.0.

## Funding

European Horizon project eco2adapt (grant agreement 101059498).

## Contact

Laure Berti-Equille, IRD, ESPACE-DEV, Montpellier, France (laure.berti@ird.fr)
