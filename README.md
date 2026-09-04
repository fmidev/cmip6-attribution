# cmip6-attribution

This repository contains Python software to calculate probability distributions of monthly, seasonal and annual mean temperatures in a changing climate. The program modifies observed temperature time series using (1) a time series of global mean temperature and (2) regression coefficients describing changes in local mean temperature and temperature variance as a function of global mean temperature.

The original CMIP6-based method is documented in Rantanen et al. (2024):

Rantanen et al. (2024): *A method for estimating the effect of climate change on monthly mean temperatures: September 2023 and other recent record-warm months in Helsinki, Finland*. Atmospheric Science Letters, 25(6), e1216. https://doi.org/10.1002/asl.1216

The development version of the software additionally allows the regression coefficient for changes in local mean temperature to be estimated directly from observations at the station being analysed.

## Preparing the input data

### 1. Calculation of the covariate and CMIP6-based regression coefficients

Run `calculate_coeffs.py`. The script calculates the 11-year running mean global temperature used as a covariate and the regression coefficients derived from CMIP6 climate models.

You need to define the emission scenario and the input and output paths in the script.

The CMIP6-based regression coefficients describe:

- `aam`: change in local mean temperature per degree of global warming
- `aav`: change in local temperature variance as a function of global warming

### 2. Observational global temperature data

The observational estimate of the local mean-temperature coefficient (`aam`) is calculated directly from the observed temperature record of the station being analysed.

The predictor in this regression is the 11-year running mean global temperature derived from HadCRUT5. The corresponding time series is stored in:

`observations/g11_HadCRUT5.nc`

The station temperature data and global temperature time series are automatically matched to their common period before the regression coefficient is calculated.

## Observational weighting

The contribution of the observational estimate to the local mean-temperature coefficient is controlled by the parameter `alpha`, which can take values between 0 and 1.

- `alpha = 0`: CMIP6-based coefficient only
- `alpha = 1`: observational coefficient only
- `0 < alpha < 1`: weighted combination of CMIP6 and observational coefficients

The weighted coefficient is calculated as:

`aam = (1 - alpha) * aam_CMIP6 + alpha * aam_obs`

where `aam_obs` is estimated by regressing the observed station temperature time series against the 11-year running mean global temperature.

The same observational adjustment is applied to the `aam` coefficient of the individual CMIP6 models used to estimate model uncertainty.

The coefficient describing changes in temperature variance (`aav`) remains entirely CMIP6-based.

With `alpha = 0`, the calculation corresponds to the original CMIP6-based method.

## Smoothing the observational temperature series

The parameter `aam_window` controls the length of the temperature window used when estimating the observational `aam` coefficient.

For example:

- `aam_window = 1`: the regression uses the temperature of the target month
- `aam_window = 3`: the regression uses a centred three-month mean temperature

With `aam_window = 3`, the observational regression coefficient for April is estimated using March-April-May mean temperatures. Similarly, the coefficient for January is estimated using December-January-February mean temperatures.

For January, December is assigned to the year of the following January. For example, the January 1901 three-month mean is defined as the mean of December 1900, January 1901 and February 1901, and is regressed against the global mean temperature for 1901.

Only the calculation of the observational regression coefficient uses the smoothed temperature series. The attribution calculation itself continues to use the actual temperature of the target month. Thus, for example, an attribution analysis of an exceptionally warm March uses the actual March temperature as the event value even if `aam_window = 3` is used to estimate the regression coefficient.

`aam_window` must be a positive odd integer. The default value of 1 reproduces the monthly regression approach.

## Run the main script

In the Python script `distribution_generator.py`, the following parameters are defined:

`fmisid`

This is the FMISID of the station. Based on this ID, the program reads monthly mean temperatures for the station. Currently, all FMI weather stations in Finland are available. Note: this works only within the FMI internal network. If you are not in the FMI internal network, you need to replace the reading of station data with data read locally from your device.

`target_mon`

Target month (1-12) / season (13-16) / annual mean (17), for example `12`.

This is the target month, season or annual mean of the event for which the probability distributions are calculated.

If `target_mon` = 13 / 14 / 15 / 16, the calculations are made for seasonal mean temperatures in winter (DJF), spring (MAM), summer (JJA) and autumn (SON), respectively. DJF includes December of the previous year.

For annual mean temperature, use `target_mon = 17`.

`y_target`

Target year of the calculation, for example `2022`.

For an attribution analysis of September 2023, use `y_target = 2023`.

`y_climate`

Future climate year. Typically, 2050 is used.

`y_preind`

Year that approximates the preindustrial climate. Typically, `y_preind = 1900` is used.

`pwarm`

Probability of warmer (`True`) or colder (`False`) temperatures.

Depending on this parameter, either the probability of higher (`pwarm = True`) or lower (`pwarm = False`) temperatures than the observed event is calculated.

`ssp`

Emission scenario for the future climate.

Available options are:

`"ssp119"`, `"ssp126"`, `"ssp245"`, `"ssp370"` and `"ssp585"`.

`alpha`

Weight given to the observational estimate of the local mean-temperature coefficient (`aam`).

`alpha` must be between 0 and 1.

- `alpha = 0` uses only the CMIP6-based coefficient
- `alpha = 1` uses only the observational estimate
- intermediate values combine the two estimates

`aam_window`

Length of the centred temperature window used to estimate the observational `aam` coefficient.

The value must be a positive odd integer.

- `aam_window = 1` uses the target month directly
- `aam_window = 3` uses a centred three-month mean
- `aam_window = 5` uses a centred five-month mean

The `aam_window` parameter affects only the estimation of the observational regression coefficient. The actual event temperature and the probability distributions remain based on the original target month, season or annual mean.

In addition to these parameters, the first and last years of observations used in the calculation of probability distributions, the number of bootstrapping samples, and the path to input data can be defined at the beginning of the script.

## Output

The program outputs the annual probabilities for ```y_preind```, ```y_target``` and ```y_climate```, probability ratio and change in intensity. For September 2023 in Helsinki Kaisaniemi, the output looks like this:

Annual probabilities:

1900: 0.19 % 0.01 0.58 %\
2023: 1.77 % 0.38 3.71 %\
2050: 5.59 % 1.27 15.02 %

Probability ratio:  9.4 (3.2-124.4)\
Change in intensity: 1.4°C (0.8-2.0)

In addition, three plots are produced.

![timeser_plot](https://github.com/fmidev/cmip6-attribution/assets/22466785/b7e6958a-b8d0-4d2f-89b0-9d2b33a81d81)
Fig. 1. Time series of observed mean temperatures. Here, September from Helsinki Kaisaniemi in 1901-2023 is used. Black line shows the actual observations, and blue dots show the pseudo-observations representing today’s (2023) climate. Red error bars in pseudo-observations indicate 5th and 95h percentiles of the model ensemble. Blue dashed line marks the 2023 monthly mean temperature, 15.8°C.

![dist_plot](https://github.com/fmidev/cmip6-attribution/assets/22466785/dfbf7382-eeda-47d7-9ad7-32384ccc2e81)
Fig. 2. a The frequency distribution of pseudo-observations representing September monthly mean temperatures in today’s (2023) climate in Helsinki (blue bars), and SGS probability distribution of September pseudo-observations (blue line). In the upper left corner of the figure, the values of the four moments are annotated: mean (μ), variance (σ²), skewness (γ) and kurtosis (κ). b SGS distributions of pseudo-observations for climates in 1900 (green line), 2023 (blue line) and 2050 (red line). Black vertical line marks the observed mean temperature in 2023.

![model_stats](https://github.com/fmidev/cmip6-attribution/assets/22466785/9fcf31bd-1577-461e-961f-ffb52d37a90d)
Fig. 3. Model-simulated probability ratios and changes in intensity for September 2023 in Helsinki Kaisaniemi. In each model, only one realization is used. Thus, the uncertainty in PR values is entirely due to internal variability, i.e. it comes from the bootstrapping. The boxes show the first and third quartiles, and whiskers extend to the 5–95th percentiles of the realizations. MMM at the bottom row refers to the multi-model mean estimate.

## More information

More information can be asked from\
Mika Rantanen\
Researcher, Weather and Climate Change Impact Research\
mika.rantanen@fmi.fi
 
