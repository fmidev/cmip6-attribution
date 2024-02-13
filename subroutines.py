#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:55:35 2023

@author: rantanem
"""

import xarray as xr
import numpy as np
import pandas as pd

def find_nearest(array, value):
    array = np.asarray(array)
    diff = np.abs(array - value)
    idx = (diff).argmin()
    return array[idx]

def get_model_names(input_path):
    
    df = pd.read_excel(input_path + 'input_data/model_names.xlsx')
    
    df = df['Model']
    df.index= df.index+1
    
    return df
    

def get_place_text(place):
    
    place_text = {'kaisaniemi':'Helsinki Kaisaniemi',
                  'sodankylä':'Sodankylä Tähtelä',
                  'finland':'Finland national average'}
    return place_text[place]

def get_scenario_text(ssp):
    
    ssp_text = {'ssp119':'SSP1-1.9',
                'ssp126':'SSP1-2.6',
                'ssp245':'SSP2-4.5',
                'ssp370':'SSP3-7.0',
                'ssp585':'SSP5-8.5',}
    return ssp_text[ssp]

def get_target_text(target_mon):
    
    target_text = {1:'January',
                   2:'February',
                   3:'March',
                   4:'April',
                   5:'May',
                   6:'June',
                   7:'July',
                   8:'August',
                   9:'September',
                   10:'October',
                   11:'November',
                   12:'December',
                   13:'DJF',
                   14:'MAM',
                   15:'JJA',
                   16:'SON',
                   17:'Year'}
    return target_text[target_mon]

def read_obs_temp(input_path, place, target_mon):
    
    if place=='kaisaniemi':
        filename=input_path + 'input_data/HKI_T_R_ori-adj.xls'
        local_temp_ds = pd.read_excel(filename, sheet_name='HKI-Tadj', index_col=0)/10
        all_obs_months = local_temp_ds.stack(dropna=False)
        all_obs_months.index = pd.to_datetime(all_obs_months.index.get_level_values(0).astype(str)  + all_obs_months.index.get_level_values(1), format='%Y%b')
        
        all_obs_months = pd.DataFrame(data=all_obs_months, columns=['tmean'])
        
    elif place=='sodankylä':
        filename=input_path + 'input_data/sodankyla_monthly_temperatures.csv'
        local_temp_ds = pd.read_csv(filename, index_col=0)
        all_obs_months = local_temp_ds.stack(dropna=False)
        all_obs_months.index = pd.to_datetime(all_obs_months.index.get_level_values(0).astype(str)  + all_obs_months.index.get_level_values(1), format='%Y%b')
        
        all_obs_months = pd.DataFrame(data=all_obs_months, columns=['tmean'])
        
    elif place=='finland':
        filename='/Users/rantanem/Downloads/Suomi_hila_data.xlsx'
        local_temp_ds = pd.read_excel('/Users/rantanem/Downloads/Suomi_hiladata.xlsx', engine='openpyxl',
                                      index_col=0)
        all_obs_months = local_temp_ds.iloc[:,6:18].stack(dropna=False)
        all_obs_months.index = pd.to_datetime(all_obs_months.index.get_level_values(0).astype(str)  + all_obs_months.index.get_level_values(1), format='%Y%b')
        
        all_obs_months = pd.DataFrame(data=all_obs_months, columns=['tmean'])
    

    if target_mon<=12:
        obs_temp = all_obs_months[all_obs_months.index.month==target_mon].loc[slice('1850-01-01','2023-12-31')]
        obs_temp.index = obs_temp.index.year
    elif target_mon==13:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month==2].loc[slice('1850-01-01','2023-12-31')]
        obs_temp.index = obs_temp.index.year
    elif target_mon==14:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month==5].loc[slice('1850-01-01','2023-12-31')]
        obs_temp.index = obs_temp.index.year
    elif target_mon==15:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month==8].loc[slice('1850-01-01','2023-12-31')]
        obs_temp.index = obs_temp.index.year
    elif target_mon==16:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month==11].loc[slice('1850-01-01','2023-12-31')]
        obs_temp.index = obs_temp.index.year
    elif target_mon==17:
        obs_temp = all_obs_months.groupby(all_obs_months.index.year).apply(lambda g: g.mean(skipna=False)).loc[1850:2023]
    if target_mon>17:
        import sys;
        print('Target month is not valid. Please select between 1 and 17. Aborting.')
        sys.exit()
    
    # neglect the NAN values
    idx = np.isfinite(obs_temp).values
    obs_temp = obs_temp[idx]
        
    return obs_temp


def read_sim_temp_single_models(input_path, ssp, glob_obs_temp):
    
    import sys
    
    if ssp=='ssp119':
        filename = input_path + 'single_models/Tglob_'+ssp+'_14mod_1901-2100_minus_2000.nc'
    elif ssp=='ssp126':
        filename = input_path + 'single_models/Tglob_'+ssp+'_29mod_1901-2100_minus_2000.nc'
    elif ssp=='ssp245':
        filename = input_path + 'single_models/Tglob_'+ssp+'_29mod_1901-2100_minus_2000.nc'
    elif ssp=='ssp370':
        filename = input_path + 'single_models/Tglob_'+ssp+'_30mod_1901-2100_minus_2000.nc'
    elif ssp=='ssp585':
        filename = input_path + 'single_models/Tglob_'+ssp+'_30mod_1901-2100_minus_2000.nc'
    else:
        print('Scenario is not valid. Aborting...')
        sys.exit() 
    
    ## READ simulated temperature
    tglob_sim_ds = xr.open_dataset(filename)
    tglob_sim_temp = tglob_sim_ds.dt.squeeze().rename('t')
    
    # merge observed and simulated global temperature
    diff = glob_obs_temp[-1] - tglob_sim_temp.sel(time='2022-01-01')
    glob_temp = xr.concat([glob_obs_temp, tglob_sim_temp.sel(time=slice('2023-01-01','2100-12-31'))+diff], 
                          dim='time')


    # smooth with 11-year running average
    glob_temp_smooth = glob_temp.rolling(time=11, center=True, min_periods=1).mean()
    glob_temp_smooth['time'] = glob_temp_smooth.time.dt.year

    # Select year 2000 as baseline
    glob_temp_smooth = glob_temp_smooth - glob_temp_smooth.sel(time=2000)


    # convert to dataframe
    glob_temp_smooth = glob_temp_smooth.astype(float).drop_vars(('lat','lon')).to_pandas()
    
    return glob_temp_smooth


def read_sim_temp_model_mean(input_path, ssp, glob_obs_temp):
    
    import sys
    
    if ssp=='ssp119':
        filename = input_path + 'model_mean/Tglob_'+ssp+'_14mod_mean_1901-2100_minus_2000.nc'
    elif ssp=='ssp126':
        filename = input_path + 'model_mean/Tglob_'+ssp+'_29mod_mean_1901-2100_minus_2000.nc'
    elif ssp=='ssp245':
        filename = input_path + 'model_mean/Tglob_'+ssp+'_29mod_mean_1901-2100_minus_2000.nc'
    elif ssp=='ssp370':
        filename = input_path + 'model_mean/Tglob_'+ssp+'_30mod_mean_1901-2100_minus_2000.nc'
    elif ssp=='ssp585':
        filename = input_path + 'model_mean/Tglob_'+ssp+'_30mod_mean_1901-2100_minus_2000.nc'
    else:
        print('Scenario is not valid. Aborting...')
        sys.exit() 
    
    ## READ simulated temperature
    tglob_sim_ds = xr.open_dataset(filename)
    tglob_sim_temp = tglob_sim_ds.dt.squeeze().rename('t')
    
    # merge observed and simulated global temperature
    diff = glob_obs_temp[-1] - tglob_sim_temp.sel(time='2022-01-01')
    glob_temp = xr.concat([glob_obs_temp, tglob_sim_temp.sel(time=slice('2023-01-01','2100-12-31'))+diff], 
                          dim='time')


    # smooth with 11-year running average
    glob_temp_smooth = glob_temp.rolling(time=11, center=True, min_periods=1).mean()
    glob_temp_smooth['time'] = glob_temp_smooth.time.dt.year

    # Select year 2000 as baseline
    glob_temp_smooth = glob_temp_smooth - glob_temp_smooth.sel(time=2000)


    # convert to dataframe
    glob_temp_smooth = glob_temp_smooth.astype(float).drop_vars(('lat','lon')).to_pandas()
    
    return glob_temp_smooth

def read_coeffs_model_mean(input_path,ssp, target_mon, obs_lat, obs_lon):
    
    import sys
    
    if ssp=='ssp119':
        filename = input_path + 'model_mean/a_T_mean_variance_14mod_mean_'+ssp+'.nc'
    elif ssp=='ssp126':
        filename = input_path + 'model_mean/a_T_mean_variance_29mod_mean_'+ssp+'.nc'
    elif ssp=='ssp245':
        filename = input_path + 'model_mean/a_T_mean_variance_29mod_mean_'+ssp+'.nc'
    elif ssp=='ssp370':
        filename = input_path + 'model_mean/a_T_mean_variance_30mod_mean_'+ssp+'.nc'
    elif ssp=='ssp585':
        filename = input_path + 'model_mean/a_T_mean_variance_30mod_mean_'+ssp+'.nc'
    else:
        print('Scenario is not valid. Aborting...')
        sys.exit() 
    
    coeff_ds = xr.open_dataset(filename)
    coeffs =coeff_ds.sel(lat=obs_lat, lon=obs_lon, method='nearest').isel(time=target_mon-1).squeeze()

    return coeffs

def read_coeffs_single_models(input_path,ssp, target_mon, obs_lat, obs_lon):
    
    import sys
    
    if ssp=='ssp119':
        filename = input_path + 'single_models/a_T_mean_variance_14mod_'+ssp+'.nc'
    elif ssp=='ssp126':
        filename = input_path + 'single_models/a_T_mean_variance_29mod_'+ssp+'.nc'
    elif ssp=='ssp245':
        filename = input_path + 'single_models/a_T_mean_variance_29mod_'+ssp+'.nc'
    elif ssp=='ssp370':
        filename = input_path + 'single_models/a_T_mean_variance_30mod_'+ssp+'.nc'
    elif ssp=='ssp585':
        filename = input_path + 'single_models/a_T_mean_variance_30mod_'+ssp+'.nc'
    else:
        print('Scenario is not valid. Aborting...')
        sys.exit() 
    
    coeff_ds = xr.open_dataset(filename)
    coeffs =coeff_ds.sel(lat=obs_lat, lon=obs_lon, method='nearest').isel(time=target_mon-1).squeeze()
    

    return coeffs.rename({'am':'aam', 'av':'aav'})




def modify_obs(obs_temp, glob_temp, coeffs, y_target):
    
    # rmax and rmin define the range of accepted relative changes in the standard deviation
    rmax=2.5
    rmin=1./rmax
    # smoothed global mean temperature (relative to year 2000)
    g = glob_temp
    g.name=obs_temp.columns.values[0]
        
    # smoothed global mean T in target year
    gg = g.loc[y_target]
    
    # Calculation of the intermediate values, with changes is mean only
    mod3 = obs_temp.tmean + (coeffs.aam.values * (g.loc[y_target]-g.loc[obs_temp.index]))
    
    # Mean value, against which anomalies are defined
    mean_series = mod3.loc[slice('1901','2023')].mean()

    
    # Change in variability 
    # srat = (1.+gg*coeffs.aav.values)/(1.+g*coeffs.aav.values)
    ### EDIT 28 November 23 ###
    # if use the variance, take square root
    srat = np.sqrt((1.+gg*coeffs.aav.values)/(1.+g*coeffs.aav.values))
    srat =np.maximum(np.minimum(srat,rmax),rmin)
    fmod = mean_series + (mod3-mean_series)*srat     
    
    return fmod.loc[obs_temp.index[0]:obs_temp.index[-1]]



def frsgs(y, valmax, valmin, nbins,):
    
    # This function converts a sample of (original or modified) observations (y)
    # to a continuous SGS probability distribution (f). The corresponding
    # cumulative distribution (cub_prob) is also calculated.
    
        
    # Calculation of mean, standard deviation, skewness and excess kurtosis
    # (using wikipedia formulas; estimate for skewness is only unbiased
    # for symmetric distributions)  
    
    EPS=1e-3
    resol=(valmax-valmin)/(nbins-1)

        
    n = len(y)
    f = np.zeros((nbins))
    cum_prob = np.zeros((nbins))
    
    m1=0
    m2=0
    m3=0
    m4=0
    ndata=0
    for i in np.arange(0, n):
        if np.isfinite(y[i]):
           m1=m1+y[i]
           ndata=ndata+1
        
     
    m1=m1/ndata        
    for i in np.arange(0,n):
        if np.isfinite(y[i]):    
           m2=m2+((y[i]-m1)**2.)/ndata
           m3=m3+((y[i]-m1)**3.)/ndata
           m4=m4+((y[i]-m1)**4.)/ndata

    std=np.sqrt((ndata-0.)/(ndata-1.)*m2)
    skew=m3/(std**3.)
    variance = std**2
    kurt=(ndata+1.)*ndata/((ndata-1.)*(ndata-2.)*(ndata-3.))*ndata*m4/(std**4.) -3*(ndata-1.)*(ndata-1.)/((ndata-2.)*(ndata-3.))
    
    
    #  SGS parameters using Eqs. 8a-8c in Sardesmukh et al. 2015
    # (J. Climate, 28, 9166-9187)
    
    e2=np.maximum(2./3.*(kurt-3./2.*(skew**2.)/(kurt+2-(skew**2.))),1.-1./np.sqrt(1+(skew**2.)/4.)+EPS) 

    
    g=skew*std*(1.-e2)/(2*np.sqrt(e2))
    b2=2*(std**2.)*(1-e2/2.-((1-e2)**2.)/(8.*e2)*(skew**2.))
    
    if b2 < 0:
        f[:]=np.nan
        cum_prob[:]=np.nan    
        
        return f, cum_prob, (m1, variance, skew, kurt)
    
    # Calculation of the probability density function, first unnormalized.
    # Note that it is assumed that there is no probability mass beyond the range 
    # fmin...fmax -> these need to be put far enough in the tails.    
   
    
    for ind in np.arange(0, nbins):
        x=valmin+(ind-1.)/(nbins-1.)*(valmax-valmin)-m1
        f[ind]=np.log((np.sqrt(e2)*x+g)**2.+b2)*(-1.-1./e2) +(2*g/(e2*np.sqrt(b2))*np.arctan((np.sqrt(e2)*x+g)/np.sqrt(b2)))
    
    fmax=f[0]
    for ind in np.arange(1,nbins):
        if f[ind] > fmax:
            fmax=f[ind]
 
    
    sumf=0.
    for ind in np.arange(0,nbins):  
        f[ind]=np.exp(f[ind]-fmax)
  
    for ind in np.arange(0,nbins):
        sumf=sumf+resol*f[ind]
  
    for ind in np.arange(1,nbins):
        f[ind]=f[ind]/sumf

    
    cum_prob[0] = 0. 
    for ind in np.arange(1,nbins):
        cum_prob[ind]=cum_prob[ind-1]+resol*(f[ind]+f[ind-1])/2.
  
    for ind in np.arange(1,nbins):
        cum_prob[ind]=cum_prob[ind]/cum_prob[nbins-1] 
  
    
    return f, cum_prob, (m1, variance, skew, kurt)
    
def calculate_sgs_dist(obs_df, y1base, y2base, valmax, valmin, nbins, n_bts):
    
    import random

    
    obs_df = pd.DataFrame(obs_df)
    
    n_mod = obs_df.shape[1]
    
    resol=(valmax-valmin)/(nbins-1)
    index = np.arange(valmin, valmax+resol, resol).round(3)
    
    
    f_arr = np.zeros((len(index), n_mod, n_bts))
    cp_arr = np.zeros((len(index), n_mod, n_bts))
    

        
    # loop over all models (if there are many models)
    for m in np.arange(0,n_mod):
        
        # if there is only one realization
        if n_mod>1:
            the_list = list(obs_df[m+1].loc[y1base:y2base].values.squeeze())
        else:
            the_list = list(obs_df.loc[y1base:y2base].values.squeeze())
        
        # loop over all bootstrapping
        for I in np.arange(0, n_bts):
            
            # select randomly 100 temperatures
            temp = random.choices(the_list, k=100)
        
        
            f_arr[:,m,I], cp_arr[:,m,I] = frsgs(temp, y1base, y2base, valmax, valmin, nbins)
    
    return np.reshape(f_arr, (nbins, n_mod*n_bts)), np.reshape(cp_arr, (nbins, n_mod*n_bts))

def calculate_sgs(obs_df, valmax, valmin, nbins):

    
    obs_df = pd.DataFrame(obs_df)
    
    n_mod = obs_df.shape[1]
    
    resol=(valmax-valmin)/(nbins-1)
    index = np.arange(valmin, valmax+resol, resol).round(3)

    
    f_arr = np.zeros((len(index), n_mod))
    cp_arr = np.zeros((len(index), n_mod))
    

        
    # loop over all models (if there are many models)
    for m in np.arange(0,n_mod):
        
        # if there is only one realization
        if n_mod>1:
            temp = list(obs_df[m+1].values.squeeze())
        else:
            temp = list(obs_df.values.squeeze())
        
        
        f_arr[:,m], cp_arr[:,m],moments = frsgs(temp, valmax, valmin, nbins)
    
    return np.squeeze(f_arr), np.squeeze(cp_arr), moments
   
    



def find_intensity_interval(x, cp_arr0, cp_arr, i ):
        
    test_list =[]
    for I in np.arange(0, np.shape(cp_arr)[1]):
        PROB = cp_arr0[i,I]
        nearest = find_nearest(cp_arr[:,I],PROB)
        ind = np.where(cp_arr[:,I] == nearest)[0]
        TEMP = np.squeeze(x[ind])
        
        test_list.append(TEMP)

    
    return (np.percentile(test_list, 5), np.percentile(test_list, 95), test_list)

def find_difference_interval(x, cp_target_arr,cp_preind_arr, i):
    
    cp_df = pd.Series(index=np.arange(0, np.shape(cp_target_arr)[1]), dtype=float)
    for m in np.arange(0, np.shape(cp_target_arr)[1]):
        cp2 = cp_target_arr[:, m]
        # Calculate the probability in the present climate
        PROB = cp2[i]
        
        t2 = np.round(np.squeeze(x[np.where(cp2 == find_nearest(cp2,PROB))[0]]),1)
        t2 = np.squeeze(x[np.where(cp2 == find_nearest(cp2,PROB))[0]])
        cp4 = cp_preind_arr[:, m]
        
        t4 = np.round(np.squeeze(x[np.where(cp4 == find_nearest(cp4,PROB))[0]]),1)
        t4 = np.squeeze(x[np.where(cp4 == find_nearest(cp4,PROB))[0]])

        if np.sum(np.isfinite(cp2))>0:
            cp_df[m] = t2-t4
    
    
    
    return (np.percentile(cp_df, 5), np.percentile(cp_df, 95), cp_df)