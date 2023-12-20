# cmip6-attribution
This repository contains Python software to calculate probability distributions of monthly, seasonal and annual mean temperature in a changing climate. The program modifies the temperature time series from observations, using 
(1) time series of global mean temperature and 
(2) cmip6-based coefficients for the changes in the local mean temperature and temperature variance normalized by the global mean temperature.

The method is documented in an upcoming study Rantanen et al. (2023, under review in ASL).

## Input parameters

In the Python script ```distribution_generator.py``` following parameters are defined:

```place```

This is the name of the station. This far, only Helsinki Kaisaniemi ```place```="kaisaniemi" and Sodankylä Tähtelä ```place```="sodankylä" are available. The coordinates of the stations and the years available for them are given directly in the script. 

```target_mon```

Target month (1-12) / season (13-16) / annual mean (17) (e.g., 12). This is the target month/season/annual for which the probability distributions will be calculated. If ```target_mon``` = 13 / 14 / 15 / 16, the calculations are made for seasonal mean temperatures in winter (DJF) / spring (MAM) / summer (JJA) / autumn (SON). DJF includes the December of the previous year. For calculation for the annual mean temperature, use ```target_mon``` = 17.

```y_target```

Target year (e.g., 2022). Target year of the calculation. For September 2023, ```y_target```=2023.

```y_climate```

Future climate year. Typically, 2050 is used. 

```y_preind```

Year which approximates the preindustrial climate. ```y_preind``` = 1900 is typically used.

```pwarm```

Probability of warmer (True) or colder (False) temperatures? Depending on this parameter, either the probability of higher (```pwarm```=True) or lower (```pwarm```=False) temperatures than the observed one is given in the distribution plots.

```ssp```

Emission scenario for future climate (```ssp```= "ssp119", "ssp126", "ssp245", "ssp370", and "ssp585" are available).

In addition to these parameters, the first and last years of observations used in calculation of probability distributions, the number of bootstrapping samples, and the path to input data can be defined in the beginning of the script.

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
 
