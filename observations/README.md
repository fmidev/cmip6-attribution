# Observational data

This directory contains observational datasets and derived data used to
estimate the climate-change regression coefficients.

## Original observational datasets

- `HadCRUT.5.1.0.0.analysis.anomalies.ensemble_mean.nc`
  - HadCRUT5 global temperature anomalies.

- `gistemp1200_GHCNv4_ERSSTv5.nc`
  - NASA GISTEMP global temperature data.

- `Global_TAVG_Gridded_1deg.nc`
  - Berkeley Earth global temperature data.

## Global mean temperature covariates

- `g11_HadCRUT5.nc`
- `g11_GISTEMP.nc`
- `g11_BEST.nc`

These files contain the 11-year running mean of global mean temperature
calculated from the corresponding observational dataset.

## Regression coefficients

- `obs_regr_coeffs_HadCRUT5.nc`
- `obs_regr_coeffs_GISTEMP.nc`
- `obs_regr_coeffs_BEST.nc`

These files contain the regression coefficients calculated from the
corresponding observational dataset using `calculate_obs_coeffs.py`.
