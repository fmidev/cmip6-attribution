# cmip6-attribution
This repository contains Python software to calculate probability distributions of monthly, seasonal and annual mean temperature in a changing climate. The program modifies the temperature time series from observations, using 
(1) time series of global mean temperature and 
(2) cmip6-based coefficients for the changes in the local mean temperature and temperature variance normalized by the global mean temperature.

The method is documented in an upcoming study Rantanen et al. (2023, under review in ASL).

In the Python script distribution_generator.py following parameters are defined:

```place```

This is the name of the station. This far, only Helsinki Kaisaniemi "kaisaniemi" and Sodankylä Tähtelä "sodankylä" are available. The coordinates of the stations and the years available for them are given directly in the script. 

```target_mon```

Target month (1-12) / season (13-16) / annual mean (17) (e.g., 12). This is the target month/season/annual for which the probability distributions will be calculated. If ```target_mon``` = 13 / 14 / 15 / 16, the calculations are made for seasonal mean temperatures in winter (DJF) / spring (MAM) / summer (JJA) / autumn (SON). DJF includes the December of the previous year. For calculation for the annual mean temperature, use ```target_mon``` = 17.

```y_target```

Target year (e.g., 2022). Target year of the calculation.

```y_climate```

Future climate year. Typically, 2050 is used. 

```y_preind```

Year which approximates the preindustrial climate. ```y_preind``` = 1900 is typically used.

```pwarm```

Probability of warmer (True) or colder (False) temperatures? Depending on this parameter, either the probability of higher (```pwarm```=True) or lower (```pwarm```=False) temperatures than the observed one is given in the distribution plots.

```ssp```

Emission scenario for future climate (```ssp```= "ssp119", "ssp126", "ssp245", "ssp370", and "ssp585" are available).


