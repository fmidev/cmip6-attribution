# cmip6-attribution
This repository contains Python software to calculate probability distributions of monthly, seasonal and annual mean temperature in a changing climate. The program modifies the temperature time series from observations, using 
(1) time series of global mean temperature and 
(2) cmip6-based coefficients for the changes in the local mean temperature and temperature variance normalized by the global mean temperature.

The method is documented in an upcoming study Rantanen et al. (2023, under review in ASL).

In the Python script distribution_generator.py following parameters are defined:

Station (currently "kaisaniemi" or "sodankylä" are available)
This is the name of the station. This far, only Helsinki Kaisaniemi and Sodankylä Tähtelä are available. The coordinates of the stations and the years available for them are given directly in the script. 

```target_mon```
Target month (1-12) / season (13-16) / annual mean (17) (e.g., 12) 
This is the month/season/annual for which the probability distributions will be calculated. If Month = 13 / 14 / 15 / 16, the calculations are made for seasonal mean temperatures in winter (DJF) / spring (MAM) / summer (JJA) / autumn (SON). DJF includes the
December of the previous year. For calculation for the annual mean temperature, use Month = 17.

Target year (e.g., 2022)


Probability of warmer (1) or colder (2) temperatures?
Depending on this parameter, either the probability of higher (1) or lower (2) temperatures than the observed one is given in the distribution plots.
First year of baseline period? (e.g., 1901)
Last year of baseline period? (e.g., 2022)
These parameters define the part of the observed time series that is used as the “raw material” in calculating the probability distributions. Obviously, this range of years must fall within the years that are available for the station (but a shorter baseline period may also be selected). However, the earliest possible start for the baseline period is 1855, since this is the first year for which the 11-year running mean global mean temperature used in the method is available.

