#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:56:57 2023

This script calculates probability distributions of observed monthly temperatures 
in Helsinki Kaisaniemi and Sodankylä Tähtelä in observed and future climates. 
The methdology is adopted from Jouni Räisänen (Extreme value analysis seminar 
in October 2022). The main aspects have been documented in these two papers

Räisänen, J. and L. Ruokolainen, 2008: Estimating present climate in a
warming world: a model-based approach. Climate Dynamics, 31, 573-585

and

Räisänen, J. and L. Ruokolainen, 2008: Ongoing global warming and local
warm extremes: a case study of winter 2006-2007 in Helsinki, Finland.
Geophysica, 44, 45-65.

@author: rantanem
"""

import xarray as xr
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import subroutines

"""
First, give the input parameters
"""
# Select place (kaisaniemi or sodankylä)
place='kaisaniemi'

# Target Month (1-12) / Season (13-16) / Annual mean (17)? (e.g., 12)
target_mon = 9

# Target year
y_target = 2023

# Future climate year
y_climate = 2050

# Preindustrial climate year
y_preind = 1900

# probability of warmer temperatures (colder if false)
pwarm = True

# Scenario for future climate (ssp119, ssp126, ssp245, ssp370, ssp585)
ssp = 'ssp245'


"""
################################################################
Next, specify some additional parameters used in the calculation
################################################################
"""

# first and last of observations used in calculation of probability distributions
y1base=1901
y2base=2000
# Assuming monthly mean temperatures are within -40 ... +40 C
valmin=-40.0
valmax=40.0
nbins=1601

# Number of bootstrapping in the uncertainty estimate
n_bts = 10
# input path for the datafiles
input_path = '/Users/rantanem/Documents/python/cmip6-attribution/'
# lat/lon coordinates
latlon = {'kaisaniemi':(60.2,25.0),
          'sodankylä':(67.4,26.6),
          'finland':(64.1,26.0)}

"""
################################################################
Read observed and simulated global temperatures, merge them and
smooth with 11-year rolling average
################################################################
"""
## READ observed temperature
glob_obs_ds = xr.open_dataset(input_path + 'input_data/HadCRUT5_global_annual.nc')
glob_obs_temp = glob_obs_ds['tas_mean'].squeeze().sel(time=slice('1850-01-01','2022-12-31'))

## READ simulated temperature, merge it with observed temperature and smooth with 11-year moving average
glob_temp_single = subroutines.read_sim_temp_single_models(input_path, ssp, glob_obs_temp)
glob_temp_mean = subroutines.read_sim_temp_model_mean(input_path, ssp, glob_obs_temp)

# Number of models used in the calculation
n_mod = len(glob_temp_single.columns)


"""
################################################################
Finally, read observed local temperatures and the coefficients 
for the Tglob-regressed changes in mean and variability
################################################################
"""
#### A) Observed local temperatures  
obs_temp = subroutines.read_obs_temp(input_path, place, target_mon).loc[1901:]


### B) The coefficients for the Tglob-regressed changes in mean and variability
###    The GrADS binary file includes (somewhat illogically) for each 17 seasons
coeffs_single = subroutines.read_coeffs_single_models(input_path,ssp, target_mon, latlon[place][0], latlon[place][1])
coeffs_mean = subroutines.read_coeffs_model_mean(input_path,ssp, target_mon, latlon[place][0], latlon[place][1])

"""
################################################################
Everything has been read in. Next, recalculate the observed time series    
for the target year, future year and pre-industrial climate year. 
################################################################
"""
# Calculate single models
pseudo_obs_preind_year_single_models = pd.DataFrame(index=obs_temp.index, columns=glob_temp_single.columns)
pseudo_obs_target_year_single_models = pd.DataFrame(index=obs_temp.index, columns=glob_temp_single.columns)
pseudo_obs_future_year_single_models = pd.DataFrame(index=obs_temp.index, columns=glob_temp_single.columns)


for m in glob_temp_single.columns:
    
    pseudo_obs_preind_year_single_models[m] = subroutines.modify_obs(obs_temp, glob_temp_single[m], coeffs_single.sel(lev=m), y_preind)
    pseudo_obs_target_year_single_models[m] = subroutines.modify_obs(obs_temp, glob_temp_single[m], coeffs_single.sel(lev=m), y_target)
    pseudo_obs_future_year_single_models[m] = subroutines.modify_obs(obs_temp, glob_temp_single[m], coeffs_single.sel(lev=m), y_climate)
    
# Calculate multi-model-mean
pseudo_obs_preind_year_model_mean = subroutines.modify_obs(obs_temp, glob_temp_mean, coeffs_mean, y_preind)
pseudo_obs_target_year_model_mean = subroutines.modify_obs(obs_temp, glob_temp_mean, coeffs_mean, y_target)
pseudo_obs_future_year_model_mean = subroutines.modify_obs(obs_temp, glob_temp_mean, coeffs_mean, y_climate)


"""
################################################################
Then, calculate the stochastically generated skewed (SGS) distributions
for observed climate, present climate and future climate. The calculation
of the distributions is based on Sardeshmukh et al. (2015): 
https://doi.org/10.1175/JCLI-D-15-0020.1
################################################################
"""
# Select the first year (either y1base or the first year of the observations)
y1base = np.maximum(y1base, obs_temp.index[0])

# SGS distributions for observed values (real observations)
f_obs, cp_obs = subroutines.calculate_sgs(obs_temp.loc[y1base:y2base], valmax, valmin, nbins)

# SGS distributions for pseudo observations (pre-industrial, target and future climates)
f_preind, cp_preind = subroutines.calculate_sgs(pseudo_obs_preind_year_model_mean.loc[y1base:y2base], valmax, valmin, nbins)
f_target, cp_target = subroutines.calculate_sgs(pseudo_obs_target_year_model_mean.loc[y1base:y2base], valmax, valmin, nbins)
f_future, cp_future = subroutines.calculate_sgs(pseudo_obs_future_year_model_mean.loc[y1base:y2base], valmax, valmin, nbins)



# Perform the bootstrapping ##

# Allocate arrays for SGS distributions
cp_obs_arr = np.empty((nbins,n_bts))
f_obs_arr = np.empty((nbins,n_bts))
cp_preind_arr = np.empty((nbins,n_mod,n_bts))
f_preind_arr = np.empty((nbins,n_mod,n_bts))
cp_target_arr = np.empty((nbins,n_mod,n_bts))
f_target_arr = np.empty((nbins,n_mod,n_bts))
cp_future_arr = np.empty((nbins,n_mod,n_bts))
f_future_arr = np.empty((nbins,n_mod,n_bts))

# The years which are resampled in the bootsrapping
the_list = obs_temp.loc[y1base:y2base].index

# bootstrapping loop
for b in np.arange(0, n_bts):
    print('Bootstrapping number', b+1)
    
    # select randomly 100 years
    res = random.choices(the_list, k=100)
    # select temperatures of those years
    temp = obs_temp.loc[res]

    # First, calculate the SGS distribution for bootstrapped actual observations
    f_obs_arr[:, b], cp_obs_arr[:, b] = subroutines.calculate_sgs(temp, valmax, valmin, nbins)

    # Then, calculate the SGS distribution for bootstrapped pseudo observations
    f_preind_arr[:,:,b], cp_preind_arr[:,:,b] = subroutines.calculate_sgs(pseudo_obs_preind_year_single_models.loc[res], valmax, valmin, nbins)
    
    f_target_arr[:,:,b], cp_target_arr[:,:,b] = subroutines.calculate_sgs(pseudo_obs_target_year_single_models.loc[res], valmax, valmin, nbins)

    f_future_arr[:,:,b], cp_future_arr[:,:,b] = subroutines.calculate_sgs(pseudo_obs_future_year_single_models.loc[res], valmax, valmin, nbins)

# Reshape 3D arrays into 2D arrays 
f_preind_arr = f_preind_arr.reshape(*f_preind_arr.shape[:-2], -1)
cp_preind_arr = cp_preind_arr.reshape(*cp_preind_arr.shape[:-2], -1)
f_target_arr = f_target_arr.reshape(*f_target_arr.shape[:-2], -1)
cp_target_arr = cp_target_arr.reshape(*cp_target_arr.shape[:-2], -1)
f_future_arr = f_future_arr.reshape(*f_future_arr.shape[:-2], -1)
cp_future_arr = cp_future_arr.reshape(*cp_future_arr.shape[:-2], -1)



"""
################################################################
Print the probabilities of warmer/colder temperatures
################################################################
"""
# The observed temperature (i.e. the target value)
target_value = obs_temp.tmean.loc[y_target].round(2)
# The corresponding index
i=int((nbins-1.)*(target_value-valmin)/(valmax-valmin))

# calculate probabilites and their upper + lower condidence intervals using 5-95 percentiles
if pwarm:
    prob_in_obs = 1- cp_obs[i]
    prob_in_obs_up = 1- np.nanpercentile(cp_obs_arr,95,1)[i]
    prob_in_obs_low = 1- np.nanpercentile(cp_obs_arr,5,1)[i]
    prob_in_preind= 1- cp_preind[int(i)]
    prob_in_preind_up = 1- np.nanpercentile(cp_preind_arr,95,1)[i]
    prob_in_preind_low = 1- np.nanpercentile(cp_preind_arr,5,1)[i]
    prob_in_target = 1- cp_target[int(i)]
    prob_in_target_up = 1- np.nanpercentile(cp_target_arr,95,1)[i]
    prob_in_target_low = 1- np.nanpercentile(cp_target_arr,5,1)[i]
    prob_in_future = 1- cp_future[int(i)]
    prob_in_future_up = 1- np.nanpercentile(cp_future_arr,95,1)[i]
    prob_in_future_low = 1- np.nanpercentile(cp_future_arr,5,1)[i]

else:
    prob_in_obs = cp_obs[int(i)]
    prob_in_obs_up = np.nanpercentile(cp_obs_arr,5,1)[i]
    prob_in_obs_low = np.nanpercentile(cp_obs_arr,95,1)[i]
    prob_in_preind = cp_preind[int(i)]
    prob_in_preind_up = np.nanpercentile(cp_preind_arr,5,1)[i]
    prob_in_preind_low = np.nanpercentile(cp_preind_arr,95,1)[i]
    prob_in_target = cp_target[int(i)]
    prob_in_target_up = np.nanpercentile(cp_target_arr,5,1)[i]
    prob_in_target_low = np.nanpercentile(cp_target_arr,95,1)[i]
    prob_in_future = cp_future[int(i)]
    prob_in_future_up = np.nanpercentile(cp_future_arr,5,1)[i]
    prob_in_future_low = np.nanpercentile(cp_future_arr,95,1)[i]

print("\nRESULTS\n")

print(str(y_preind)+':',np.round(prob_in_preind*100,2),'%',np.round(prob_in_preind_up*100,2),np.round(prob_in_preind_low*100,2),'%')

print(str(y_target)+':',np.round(prob_in_target*100,2),'%',np.round(prob_in_target_up*100,2),np.round(prob_in_target_low*100,2),'%')    

print(str(y_climate)+':',np.round(prob_in_future*100,2),'%',np.round(prob_in_future_up*100,2),np.round(prob_in_future_low*100,2),'%')

print('Probability ratio: ',np.round(prob_in_target / prob_in_preind, 1))

"""
################################################################
Finally, plot the results: first the time series plot and then the
distribution plot
################################################################
"""

# Define the default font
font = {'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)
# Define texts for the plots
place_text = subroutines.get_place_text(place)
target_text = subroutines.get_target_text(target_mon) + ' '+str(y_target)



### 1) Time series plot

## Show time series only up to target year
obs_temp_plot = obs_temp.loc[slice(str(y1base),str(y_target))]
target_mean = pseudo_obs_target_year_model_mean.loc[slice(str(y1base),str(y_target))]
target_up = pseudo_obs_target_year_single_models.quantile(0.95, axis=1).loc[slice(str(y1base),str(y_target))]
target_low = pseudo_obs_target_year_single_models.quantile(0.05, axis=1).loc[slice(str(y1base),str(y_target))]
preind_mean = pseudo_obs_preind_year_model_mean.loc[slice(str(y1base),str(y_target))]
preind_up = pseudo_obs_preind_year_single_models.quantile(0.95, axis=1).loc[slice(str(y1base),str(y_target))]
preind_low = pseudo_obs_preind_year_single_models.quantile(0.05, axis=1).loc[slice(str(y1base),str(y_target))]

y_l = np.array(target_mean) - np.array(target_low)
y_u =np.array(target_up) - np.array(target_mean)

errors = [y_l, y_u]


fig=plt.figure(figsize=(10,7), dpi=300); ax=plt.gca()

ax.plot(obs_temp_plot.index,obs_temp_plot, color='k', linewidth=1.5, label='Observations')

ax.scatter(target_mean.index, target_mean, label='Pseudo-observations')
ax.errorbar(target_mean.index, target_mean, yerr=errors, fmt='o', ecolor = 'red')

ax.axhline(y=obs_temp_plot['tmean'].loc[y_target], linestyle='--')

ax.set_ylabel('Temperature [°C]')
# ax.set_ylim(7,17.2)
ax.grid(True)
ax.tick_params(axis='both', which='major',)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.09),ncol=2,)
# ax.set_title('a) '+subroutines.get_target_text(target_mon)+' temperatures' , loc='left', 
#              fontsize=14)
ax.set_xlim(1900, 2024)

figurePath = '/Users/rantanem/Documents/python/cmip6-attribution/figures/'
figureName = 'timeser_plot.png'
   
plt.savefig(figurePath + figureName,dpi=300,bbox_inches='tight')



### 2) distribution plot
x = np.linspace(valmin, valmax,nbins)
hist, bin_edges = np.histogram(obs_temp.loc[y1base:y2base].values.squeeze(), density=True, 
                               bins=np.arange(valmin, valmax, 0.5)+0.25)

# fig=plt.figure(figsize=(7,7), dpi=300); ax=plt.gca()
fig, axlist=plt.subplots(nrows=1, ncols=2,figsize=(14,7), dpi=300, sharey=False)
ax1=axlist[0]
ax2=axlist[1]

ax1.plot(x,f_obs, color='blue', label=str(y1base)+'-'+str(y2base), zorder=7, linewidth=2)

ax2.plot(x,f_obs, color='blue',  zorder=7, linewidth=2)
ax2.plot(x,f_preind, color='grey',label="\'"+str(y_preind)+"\'", zorder=9, linewidth=2)
ax2.plot(x,f_target, color='orange',label="\'"+str(y_target)+"\'", zorder=8, linewidth=2)
ax2.plot(x,f_future, color='red',label="\'"+str(y_climate)+"\'", zorder=9, linewidth=2)



ax1.bar(bin_edges[:-1]+0.25,hist, 
        width=0.4, edgecolor='k', facecolor='skyblue', zorder=2, label=str(y1base)+'-'+str(y2base))

ax2.axvline(x=target_value, zorder=10, color='k')
ax2.annotate(f'{np.round(target_value,1):.1f}°C', xy=(target_value-0.1,ax2.get_ylim()[1]), xycoords='data',
            rotation=90, ha='right', va='top')

if pwarm:
    ax2.fill_between(x,f_obs,where = x>=target_value, color='skyblue', zorder=5, alpha=0.7)
    ax2.fill_between(x,f_target,where = x>=target_value, color='bisque', zorder=4, alpha=0.7)
    ax2.fill_between(x,f_future,where = x>=target_value, color='coral', zorder=3, alpha=0.7)
    probtext = 'Probability\ of\ T\ ≥\ '+f'{np.round(target_value,1):.1f}°C'
    returntext = 'Return\ period\ of\ T\ ≥\ '+f'{np.round(target_value,1):.1f}°C'
else:
    ax2.fill_between(x,f_obs,where = x<=target_value, color='lightblue', zorder=3, alpha=0.7)
    ax2.fill_between(x,f_preind,where = x<=target_value, color='coral', zorder=4, alpha=0.7)
    ax2.fill_between(x,f_target,where = x<=target_value, color='lightgrey', zorder=5, alpha=0.7)
    probtext = 'Probability\ of\ T\ ≤\ '+f'{np.round(target_value,1):.1f}°C'
    returntext = 'Return\ period\ of\ T\ ≤\ '+f'{np.round(target_value,1):.1f}°C'

# ax.tick_params(axis='both', which='major')
for ax in axlist:
    ax.set_xticks(np.arange(valmin, valmax, 2))
    ax.set_xlim(np.floor(np.nanmin(obs_temp.loc[y1base:y2base].values.squeeze()))-3,np.ceil(np.nanmax(obs_temp.tmean))+4)
    ax.set_xlabel('Temperature [°C]')
    ax.legend(loc='upper right', frameon=False, ncol=3, bbox_to_anchor=(1., 1.1))
    ax.grid(True, zorder=1)
    ax.set_ylim(0, 0.35)
ax1.set_ylabel('Relative frequency / probability density [1/°C]')

# Target value in 1901-2000 and 1900 climates
i = int((nbins-1.)*(target_value-valmin)/(valmax-valmin))
prob = cp_target[i]

t_1901 = np.round(np.squeeze(x[np.where(cp_obs == subroutines.find_nearest(cp_obs,prob))[0]]),1)
t_1900 = np.round(np.squeeze(x[np.where(cp_preind == subroutines.find_nearest(cp_preind,prob))[0]]),1)
t_2050 = np.round(np.squeeze(x[np.where(cp_future == subroutines.find_nearest(cp_future,prob))[0]]),1)

t1900_lower, t1900_upper,t1900_list = subroutines.find_intensity_interval(x, prob, cp_target_arr, cp_preind_arr, i)
t2050_lower, t2050_upper,_ = subroutines.find_intensity_interval(x, prob, cp_target_arr, cp_future_arr, i)
tdiff_lower, tdiff_upper = subroutines.find_difference_interval(x, prob, cp_target_arr,cp_preind_arr)
howtext = 'Change\ in\ intensity\  '

textstr = '\n'.join((
    place_text,
    target_text,
    '',
    r"$\bf{"+probtext+"}$",
    str(y1base)+'-'+str(y2base)+': '+f'{np.round(prob_in_obs*100,1):.1f} (' +\
    f'{np.round(prob_in_obs_up*100,1):.1f}-'+f'{np.round(prob_in_obs_low*100,1):.1f}) %',
    '',
    "\'"+str(y_preind)+"\'"+': '+f'{np.round(prob_in_preind*100,1):.1f} (' +\
    f'{np.round(prob_in_preind_up*100,1):.1f}-'+f'{np.round(prob_in_preind_low*100,1):.1f}) %',
    "\'"+str(y_target)+"\'"+': '+f'{np.round(prob_in_target*100,1):.1f} ('+\
    f'{np.round(prob_in_target_up*100,1):.1f}-'+f'{np.round(prob_in_target_low*100,1):.1f}) %',
    "\'"+str(y_climate)+"\'"+': '+f'{np.round(prob_in_future*100,1):.1f} ('+\
    f'{np.round(prob_in_future_up*100,1):.1f}-'+f'{np.round(prob_in_future_low*100,1):.1f}) %',
    '',
    r"$\bf{"+returntext+"}$",
    str(y1base)+'-'+str(y2base)+': '+f'{np.round(1/prob_in_obs,0):.0f} ('+\
    f'{np.round(1/prob_in_obs_low,0):.0f}-'+f'{np.round(1/prob_in_obs_up,0):.0f}) years',
    '',
    "\'"+str(y_preind)+"\'"+': '+f'{np.round(1/prob_in_preind,0):.0f} (' +\
    f'{np.round(1/prob_in_preind_low,0):.0f}-'+f'{np.round(1/prob_in_preind_up,0):.0f}) years',
    "\'"+str(y_target)+"\'"+': '+f'{np.round(1/prob_in_target,0):.0f} ('+\
    f'{np.round(1/prob_in_target_low,0):.0f}-'+f'{np.round(1/prob_in_target_up,0):.0f}) years',
    "\'"+str(y_climate)+"\'"+': '+f'{np.round(1/prob_in_future,0):.0f} ('+\
    f'{np.round(1/prob_in_future_low,0):.0f}-'+f'{np.round(1/prob_in_future_up,0):.0f}) years',
    '',
    r"$\bf{"+howtext+"}$",
    str(y1base)+'-'+str(y2base)+': '+f'{t_1901:.1f}°C',
    '',
    "\'"+str(y_preind)+"\'"+': '+f'{t_1900:.1f}°C ({t1900_lower:.1f}-{t1900_upper:.1f})°C',
    "\'"+str(y_target)+"\'"+': '+f'{target_value:.1f}°C',
    "\'"+str(y_climate)+"\'"+': '+f'{t_2050:.1f}°C ({t2050_lower:.1f}-{t2050_upper:.1f})°C',
    '',
    'Scenario for future climate: \n'+subroutines.get_scenario_text(ssp),
    ))


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax2.text(1.04, 0.99, textstr, transform=ax.transAxes, fontsize=13,
          verticalalignment='top', bbox=props)

# ax.set_title('b) '+subroutines.get_target_text(target_mon)+' temperature distributions' , loc='left', 
#               fontsize=14)

ax1.annotate('a', (0.03, 0.95), xycoords='axes fraction', fontweight='bold', fontsize=18)
ax2.annotate('b', (0.03, 0.95), xycoords='axes fraction', fontweight='bold', fontsize=18)

figurePath = '/Users/rantanem/Documents/python/cmip6-attribution/figures/'
figureName = 'dist_plot.png'
   
plt.savefig(figurePath + figureName,dpi=300,bbox_inches='tight')