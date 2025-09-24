#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:55:35 2023

@author: rantanem
"""

import sys

import numpy as np
import pandas as pd
import requests
import xarray as xr


def get_station_metadata(fmisid):

    params = ["time", "stationname", "longitude", "latitude", "tmon"]

    starttime = pd.to_datetime("2023-01-01").strftime("%Y%m%d")
    endtime = pd.to_datetime("2023-02-01").strftime("%Y%m%d")

    print("Downloading coordinates for", fmisid)
    url = (
        f"http://smartmet.fmi.fi/timeseries?format=ascii&starttime={starttime}0000&endtime={endtime}2359&"
        + f"fmisid={fmisid}&producer=opendata_daily&precision=double&"
        + f'param={",".join(params)}&separator=,&tz=UTC&groupareas=0'
    )
    try:
        df = pd.read_csv(url, names=params, parse_dates=[0], index_col="time").ffill(limit=6)
    except:
        print("FAILED")

    return df["latitude"][0], df["longitude"][0], df["stationname"][0]


def get_station_metadata_frost(metnosid: str, frost_client_id: str):
    """Get relevant metadata on station from Frost API.

    @author: Amalie Skålevåg (amalie.skalevag@met.no)

    Parameters
    ----------
    metnosid : str
        station identification number of Norwegian weather station, e.g. 'SN18700'
    frost_client_id : str
        client ID for Frost API, see https://frost.met.no/howto.html

    Returns
    -------
    int, int, str
        latitude, longitude, station name
    """
    # Define endpoint and parameters
    endpoint = "https://frost.met.no/sources/v0.jsonld"
    parameters = {
        "ids": metnosid,
    }

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(frost_client_id, ""))

    # Extract JSON data
    json = r.json()
    lon, lat = json["data"][0]["geometry"]["coordinates"]

    return lat, lon, json["data"][0]["shortName"]


def read_monthly_temps_from_smartmet(fmisid):

    params = ["time", "stationname", "tmon"]

    starttime = pd.to_datetime("1901-01-01").strftime("%Y%m%d")
    endtime = pd.to_datetime("today").normalize().strftime("%Y%m%d")

    print("Downloading data for", fmisid)
    url = (
        f"http://smartmet.fmi.fi/timeseries?format=ascii&starttime={starttime}0000&endtime={endtime}2359&"
        + f"fmisid={fmisid}&producer=opendata_daily&precision=double&"
        + f'param={",".join(params)}&separator=,&tz=UTC&groupareas=0'
    )
    try:
        df = pd.read_csv(url, names=params, parse_dates=[0], index_col="time")
    except:
        print("FAILED")

    return df["tmon"]


def read_monthly_temps_from_frost(metnosid: str, frost_client_id: str, homogenised=True):
    """Retrieve monthly temperature data from Frost API.

    @author: Amalie Skålevåg (amalie.skalevag@met.no), Herman F. Fuglestvedt

    Parameters
    ----------
    metnosid : str
        station identification number of Norwegian weather station, e.g. 'SN18700'
    frost_client_id : str
        client ID for Frost API, see https://frost.met.no/howto.html
    homogenised : bool, optional
        whether to use homogenised monthly temperatures or not, by default True
        homogenised timeseries tend to be longer

    Returns
    -------
    pandas.Series
        time series of monthly temperatures
    """

    # determine variable name
    if homogenised:
        var_name = "best_estimate_mean(air_temperature P1M)"
    else:
        var_name = "mean(air_temperature P1M)"

    # Define endpoint and parameters
    endpoint = "https://frost.met.no/observations/v0.jsonld"
    parameters = {
        "sources": metnosid,
        "elements": var_name,
        "referencetime": f"1850-01-01/{pd.Timestamp.utcnow().strftime('%Y-%m-%d')}",
        "timeoffsets": "default",
        "levels": "default",
        "qualities": "0,1,2,3,4",
    }

    r = requests.get(endpoint, parameters, auth=(frost_client_id, ""))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json["data"]
    else:
        print("Error! Returned status code %s" % r.status_code)
        print("Message: %s" % json["error"]["message"])
        print("Reason: %s" % json["error"]["reason"])
        raise RuntimeError(f'{json["error"]["message"]}. {json["error"]["reason"]}')

    # Create DataFrame from list of dictionaries
    df = pd.concat([pd.DataFrame(data[i]["observations"], index=[pd.to_datetime(data[i]["referenceTime"])]) for i in range(len(data))])
    # sort chronologically
    df.sort_index(inplace=True)
    # index to dates
    df.index = pd.to_datetime(df.index.date)

    return df.value.rename("tmon")


def find_nearest(array, value):
    array = np.asarray(array)
    diff = np.abs(array - value)
    idx = (diff).argmin()
    return array[idx]


# <<<<<<< Updated upstream
# def get_place_text(place):

#     place_text = {"kaisaniemi": "Helsinki Kaisaniemi", "sodankylä": "Sodankylä Tähtelä", "finland": "Finland national average"}
#     return place_text[place]
# =======
# def get_place_text(place):
    
#     place_text = {'kaisaniemi':'Helsinki Kaisaniemi',
#                   'sodankylä':'Sodankylä Tähtelä',
#                   'finland':'Finland national average'}
#     return place_text[place]
# >>>>>>> Stashed changes


def get_scenario_text(ssp):

    ssp_text = {
        "ssp119": "SSP1-1.9",
        "ssp126": "SSP1-2.6",
        "ssp245": "SSP2-4.5",
        "ssp370": "SSP3-7.0",
        "ssp585": "SSP5-8.5",
    }
    return ssp_text[ssp]


def get_target_text(target_mon):

    target_text = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
        13: "DJF",
        14: "MAM",
        15: "JJA",
        16: "SON",
        17: "Year",
    }
    return target_text[target_mon]


def read_obs_temp(input_path, fmisid, target_mon):

    # read raw observations from Smartmet
    all_obs_months = read_monthly_temps_from_smartmet(fmisid)

    # If Hel Kaisaniemi, use adjusted observations for the 20th century
    # and concat with raw observations
    if fmisid == 100971:
        print("Downloading adjusted observations...")
        filename = input_path + "input_data/HKI_T_R_ori-adj.xls"
        local_temp_ds = pd.read_excel(filename, sheet_name="HKI-Tadj", index_col=0) / 10
        all_obs_months_adj = local_temp_ds.stack(dropna=False)
        all_obs_months_adj.index = pd.to_datetime(all_obs_months_adj.index.get_level_values(0).astype(str) + all_obs_months_adj.index.get_level_values(1), format="%Y%b")

        # select only 20th century observations from adjusted values
        all_obs_months_adj = all_obs_months_adj.loc[slice("1901", "1999")]

        # concat with smartmet data
        all_obs_months = all_obs_months_adj.combine_first(all_obs_months)

    if target_mon <= 12:
        obs_temp = all_obs_months[all_obs_months.index.month == target_mon].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 13:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 2].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 14:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 5].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 15:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 8].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 16:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 11].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 17:
        obs_temp = all_obs_months.groupby(all_obs_months.index.year).apply(lambda g: g.mean(skipna=False)).loc[1850:]
    if target_mon > 17:
        import sys

        print("Target month is not valid. Please select between 1 and 17. Aborting.")
        sys.exit()

    # neglect the NAN values
    idx = np.isfinite(obs_temp).values
    obs_temp = obs_temp[idx]

    return obs_temp


def read_obs_temp_frost(frost_client_id, metnosid, target_mon):

    # read raw observations from Frost
    all_obs_months = read_monthly_temps_from_frost(metnosid, frost_client_id, homogenised=True)

    if target_mon <= 12:
        obs_temp = all_obs_months[all_obs_months.index.month == target_mon].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 13:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 2].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 14:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 5].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 15:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 8].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 16:
        obs_temp = all_obs_months.rolling(window=3).mean()[all_obs_months.index.month == 11].loc[slice("1850-01-01", None)]
        obs_temp.index = obs_temp.index.year
    elif target_mon == 17:
        obs_temp = all_obs_months.groupby(all_obs_months.index.year).apply(lambda g: g.mean(skipna=False)).loc[1850:]
    if target_mon > 17:
        import sys

        print("Target month is not valid. Please select between 1 and 17. Aborting.")
        sys.exit()

    # neglect the NAN values
    idx = np.isfinite(obs_temp).values
    obs_temp = obs_temp[idx]

    return obs_temp


def read_sim_temp_single_models(input_path, ssp, glob_obs_temp):

    filename = input_path + "single_models/" + ssp + "_g11_CMIP6_all_models_combined.nc"

    ## READ simulated temperature and smooth with 11-year rolling mean
    tglob_sim_ds = xr.open_dataset(filename)
    tglob_sim_temp = tglob_sim_ds.tas.squeeze().rename("t")
    tglob_sim_temp["time"] = tglob_sim_ds.time.dt.year
    tglob_sim_smooth = tglob_sim_temp.rolling(time=11, center=True, min_periods=1).mean()

    # smooth observed temperature with 11-year rolling mean
    glob_obs_smooth = glob_obs_temp.rolling(time=11, center=True, min_periods=1).mean()
    # find the last valid year. Minus 5 is because that's half of 11
    last_valid_idx = glob_obs_smooth.time.values[-1] - 5

    # merge observed and simulated global temperature at -5 year
    diff = glob_obs_smooth.sel(time=last_valid_idx) - tglob_sim_smooth.sel(time=last_valid_idx)
    glob_temp_smooth = xr.concat([glob_obs_smooth.sel(time=slice(None, last_valid_idx)), tglob_sim_smooth.sel(time=slice(last_valid_idx + 1, None)) + diff], dim="time")

    # Select year 2000 as baseline
    glob_temp_smooth = glob_temp_smooth - glob_temp_smooth.sel(time=2000)

    # convert to dataframe
    glob_temp_smooth = glob_temp_smooth.astype(float).drop_vars(("latitude", "longitude")).to_pandas()

    return glob_temp_smooth

# <<<<<<< Updated upstream
# =======
# def read_coeffs_model_mean(input_path,ssp, target_mon, obs_lat, obs_lon):
    
#     filename = input_path + 'model_mean/tas_'+ssp+'_regr_coeffs_CMIP6_modelmean.nc'
    
#     coeff_ds = xr.open_dataset(filename).load()
#     coeffs =coeff_ds.sel(lat=obs_lat, lon=obs_lon, method='nearest').isel(time=target_mon-1).squeeze()
# >>>>>>> Stashed changes

def read_sim_temp_model_mean(input_path, ssp, glob_obs_temp):

    filename = input_path + "model_mean/" + ssp + "_g11_CMIP6_modelmean.nc"

    ## READ simulated temperature and smooth with 11-year rolling mean
    tglob_sim_ds = xr.open_dataset(filename)
    tglob_sim_temp = tglob_sim_ds.tas.squeeze().rename("t")
    tglob_sim_temp["time"] = tglob_sim_ds.time.dt.year
    tglob_sim_smooth = tglob_sim_temp.rolling(time=11, center=True, min_periods=1).mean()

    # smooth observed temperature with 11-year rolling mean
    glob_obs_smooth = glob_obs_temp.rolling(time=11, center=True, min_periods=1).mean()
    # find the last valid year. Minus 5 is because that's half of 11
    last_valid_idx = glob_obs_smooth.time.values[-1] - 5

    # merge observed and simulated global temperature
    diff = glob_obs_smooth.sel(time=last_valid_idx) - tglob_sim_smooth.sel(time=last_valid_idx)
    glob_temp_smooth = xr.concat([glob_obs_smooth.sel(time=slice(None, last_valid_idx)), tglob_sim_smooth.sel(time=slice(last_valid_idx + 1, None)) + diff], dim="time")

    # Select year 2000 as baseline
    glob_temp_smooth = glob_temp_smooth - glob_temp_smooth.sel(time=2000)

    # convert to dataframe
    glob_temp_smooth = glob_temp_smooth.astype(float).drop_vars(("latitude", "longitude")).to_pandas()

    return glob_temp_smooth


def read_coeffs_model_mean(input_path, ssp, target_mon, obs_lat, obs_lon):

    filename = input_path + "model_mean/tas_" + ssp + "_regr_coeffs_CMIP6_modelmean.nc"

    coeff_ds = xr.open_dataset(filename)
    coeffs = coeff_ds.sel(lat=obs_lat, lon=obs_lon, method="nearest").isel(time=target_mon - 1).squeeze()
    
    return coeffs
    

def read_coeffs_single_models(input_path,ssp, target_mon, obs_lat, obs_lon):
       
    filename = input_path + 'single_models/tas_'+ssp+'_regr_coeffs_CMIP6_all_models_combined.nc'
    
    coeff_ds = xr.open_dataset(filename).load()
    coeffs =coeff_ds.sel(lat=obs_lat, lon=obs_lon, method='nearest').isel(time=target_mon-1).squeeze()

    return coeffs


def read_coeffs_single_models(input_path, ssp, target_mon, obs_lat, obs_lon):

    filename = input_path + "single_models/tas_" + ssp + "_regr_coeffs_CMIP6_all_models_combined.nc"

    coeff_ds = xr.open_dataset(filename)
    coeffs = coeff_ds.sel(lat=obs_lat, lon=obs_lon, method="nearest").isel(time=target_mon - 1).squeeze()

    return coeffs


def modify_obs(obs_temp, glob_temp, coeffs, y_target):

    # rmax and rmin define the range of accepted relative changes in the standard deviation
    rmax = 2.5
    rmin = 1.0 / rmax
    # smoothed global mean temperature (relative to year 2000)
    g = glob_temp
    g.name = obs_temp.name

    # smoothed global mean T in target year
    gg = g.loc[y_target]

    # Calculation of the intermediate values, with changes in mean only
    mod3 = obs_temp + (coeffs.aam.values * (g.loc[y_target] - g.loc[obs_temp.index]))

    # Mean value, against which anomalies are defined
    mean_series = mod3.loc[slice("1901", None)].mean()

    # Change in variability
    # srat = (1.+gg*coeffs.aav.values)/(1.+g*coeffs.aav.values)
    ### EDIT 28 November 23 ###
    # if use the variance, take square root
    srat = np.sqrt((1.0 + gg * coeffs.aav.values) / (1.0 + g * coeffs.aav.values))
    srat = np.maximum(np.minimum(srat, rmax), rmin)
    fmod = mean_series + (mod3 - mean_series) * srat

    return fmod.loc[obs_temp.index]


def frsgs(
    y,
    valmax,
    valmin,
    nbins,
):

    # This function converts a sample of (original or modified) observations (y)
    # to a continuous SGS probability distribution (f). The corresponding
    # cumulative distribution (cub_prob) is also calculated.

    # Calculation of mean, standard deviation, skewness and excess kurtosis
    # (using wikipedia formulas; estimate for skewness is only unbiased
    # for symmetric distributions)

    EPS = 1e-3
    resol = (valmax - valmin) / (nbins - 1)

    n = len(y)
    f = np.zeros((nbins))
    cum_prob = np.zeros((nbins))

    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    ndata = 0
    for i in np.arange(0, n):
        if np.isfinite(y[i]):
            m1 = m1 + y[i]
            ndata = ndata + 1

    m1 = m1 / ndata
    for i in np.arange(0, n):
        if np.isfinite(y[i]):
            m2 = m2 + ((y[i] - m1) ** 2.0) / ndata
            m3 = m3 + ((y[i] - m1) ** 3.0) / ndata
            m4 = m4 + ((y[i] - m1) ** 4.0) / ndata

    std = np.sqrt((ndata - 0.0) / (ndata - 1.0) * m2)
    skew = m3 / (std**3.0)
    variance = std**2
    kurt = (ndata + 1.0) * ndata / ((ndata - 1.0) * (ndata - 2.0) * (ndata - 3.0)) * ndata * m4 / (std**4.0) - 3 * (ndata - 1.0) * (ndata - 1.0) / ((ndata - 2.0) * (ndata - 3.0))

    if kurt < 3.0 / 2.0 * (skew**2.0):
        kurt = 3.0 / 2.0 * (skew**2.0) + EPS

    #  SGS parameters using Eqs. 8a-8c in Sardesmukh et al. 2015
    # (J. Climate, 28, 9166-9187)

    # e2=np.maximum(2./3.*(kurt-3./2.*(skew**2.)/(kurt+2-(skew**2.))),
    #               1.-1./np.sqrt(1+(skew**2.)/4.)+EPS)

    e2 = np.maximum(np.maximum(2.0 / 3.0 * (kurt - 3.0 / 2.0 * (skew**2.0) / (kurt + 2 - (skew**2.0))), 1.0 - 1.0 / np.sqrt(1 + (skew**2.0) / 4.0) + EPS), EPS)

    if e2 > 2.0 / 3.0:
        e2 = 2.0 / 3.0 * (1.0 - EPS)

    g = skew * std * (1.0 - e2) / (2 * np.sqrt(e2))
    b2 = 2 * (std**2.0) * (1 - e2 / 2.0 - ((1 - e2) ** 2.0) / (8.0 * e2) * (skew**2.0))

    if b2 < 0:
        print("asdfa")
        sys.exit()
        f[:] = np.nan
        cum_prob[:] = np.nan

        return f, cum_prob, (m1, variance, skew, kurt)

    # Calculation of the probability density function, first unnormalized.
    # Note that it is assumed that there is no probability mass beyond the range
    # fmin...fmax -> these need to be put far enough in the tails.

    for ind in np.arange(0, nbins):
        x = valmin + (ind - 1.0) / (nbins - 1.0) * (valmax - valmin) - m1
        f[ind] = np.log((np.sqrt(e2) * x + g) ** 2.0 + b2) * (-1.0 - 1.0 / e2) + (2 * g / (e2 * np.sqrt(b2)) * np.arctan((np.sqrt(e2) * x + g) / np.sqrt(b2)))

    fmax = f[0]
    for ind in np.arange(1, nbins):
        if f[ind] > fmax:
            fmax = f[ind]

    sumf = 0.0
    for ind in np.arange(0, nbins):
        f[ind] = np.exp(f[ind] - fmax)

    for ind in np.arange(0, nbins):
        sumf = sumf + resol * f[ind]

    for ind in np.arange(1, nbins):
        f[ind] = f[ind] / sumf

    cum_prob[0] = 0.0
    for ind in np.arange(1, nbins):
        cum_prob[ind] = cum_prob[ind - 1] + resol * (f[ind] + f[ind - 1]) / 2.0

    for ind in np.arange(1, nbins):
        cum_prob[ind] = cum_prob[ind] / cum_prob[nbins - 1]

    return f, cum_prob, (m1, variance, skew, kurt)


# def calculate_sgs_dist(obs_df, y1base, y2base, valmax, valmin, nbins, n_bts):

#     import random


#     obs_df = pd.DataFrame(obs_df)

#     n_mod = obs_df.shape[1]

#     resol=(valmax-valmin)/(nbins-1)
#     index = np.arange(valmin, valmax+resol, resol).round(3)


#     f_arr = np.zeros((len(index), n_mod, n_bts))
#     cp_arr = np.zeros((len(index), n_mod, n_bts))


#     # loop over all models (if there are many models)
#     for m in np.arange(0,n_mod):

#         # if there is only one realization
#         if n_mod>1:
#             the_list = list(obs_df[m+1].loc[y1base:y2base].values.squeeze())
#         else:
#             the_list = list(obs_df.loc[y1base:y2base].values.squeeze())

#         # loop over all bootstrapping
#         for I in np.arange(0, n_bts):

#             # select randomly 100 temperatures
#             temp = random.choices(the_list, k=100)


#             f_arr[:,m,I], cp_arr[:,m,I] = frsgs(temp, y1base, y2base, valmax, valmin, nbins)

#     return np.reshape(f_arr, (nbins, n_mod*n_bts)), np.reshape(cp_arr, (nbins, n_mod*n_bts))


def calculate_sgs(obs_df, valmax, valmin, nbins):

    obs_df = pd.DataFrame(obs_df)

    n_mod = obs_df.shape[1]

    resol = (valmax - valmin) / (nbins - 1)
    index = np.arange(valmin, valmax + resol, resol).round(3)

    f_arr = np.zeros((len(index), n_mod))
    cp_arr = np.zeros((len(index), n_mod))

    # loop over all models (if there are many models)
    for m in np.arange(0, n_mod):

        # if there is only one realization
        if n_mod > 1:
            temp = list(obs_df.iloc[:, m].values.squeeze())
        else:
            temp = list(obs_df.values.squeeze())

        f_arr[:, m], cp_arr[:, m], moments = frsgs(temp, valmax, valmin, nbins)

    return np.squeeze(f_arr), np.squeeze(cp_arr), moments


def find_intensity_interval(x, cp_arr0, cp_arr, i):

    test_list = []
    for I in np.arange(0, np.shape(cp_arr)[1]):
        PROB = cp_arr0[i, I]
        nearest = find_nearest(cp_arr[:, I], PROB)
        ind = np.where(cp_arr[:, I] == nearest)[0][0]
        TEMP = np.squeeze(x[ind])

        test_list.append(TEMP)

    return (np.percentile(test_list, 5), np.percentile(test_list, 95), test_list)


def find_difference_interval(x, cp_target_arr, cp_preind_arr, i):

    cp_df = pd.Series(index=np.arange(0, np.shape(cp_target_arr)[1]), dtype=float)
    for m in np.arange(0, np.shape(cp_target_arr)[1]):
        cp2 = cp_target_arr[:, m]
        # Calculate the probability in the present climate
        PROB = cp2[i]

        # t2 = np.round(np.squeeze(x[np.where(cp2 == find_nearest(cp2,PROB))[0]]),1)
        t2 = np.squeeze(x[np.where(cp2 == find_nearest(cp2, PROB))[0]][0])
        cp4 = cp_preind_arr[:, m]

        # t4 = np.round(np.squeeze(x[np.where(cp4 == find_nearest(cp4,PROB))[0]]),1)
        t4 = np.squeeze(x[np.where(cp4 == find_nearest(cp4, PROB))[0]][0])

        if np.sum(np.isfinite(cp2)) > 0:
            cp_df[m] = t2 - t4

    return (np.percentile(cp_df, 5), np.percentile(cp_df, 95), cp_df)


def obs_regression(target_mon):
    
    from scipy.stats import linregress
    
    # Start year of the linear regression
    start_year = 1900
    # End year of the linear regression
    end_year = 2020
    
    years = np.arange(int(start_year), int(end_year) + 1)
    # index of year 2000
    year_2000_idx = np.where(years == 2000)[0][0]
    
    # Read HadCRUT5 data
    ds = xr.open_dataset("/Users/rantanem/Downloads/HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc")
    da = ds["tas_mean"].squeeze()
    
    # Calculate annual mean temperature
    annual = da.resample(time="1YE").mean()
    
    # Calculate weighted global mean average
    weights = np.cos(np.deg2rad(da.latitude))
    global_mean = annual.weighted(weights).mean(dim=("latitude", "longitude"))
    
    # smooth with 11-year average
    ds_g11 = global_mean.rolling(time=11, center=True, min_periods=1).mean()
    
    # Extract data for the specific month, season or annual
    if target_mon <=12:
        monthly_data = da.sel(time=da.time.dt.month == target_mon)
    elif target_mon==13: #DJF
        d = da.resample(time='QS-DEC').mean(dim="time")
        monthly_data = d.sel(time=d.time.dt.month == 12)
        # add one year to select the year of Jan-Feb
        monthly_data['time'] = monthly_data.indexes['time'] + pd.DateOffset(years=1)
    elif target_mon==14: #MAM
        d = da.resample(time='QS-DEC').mean(dim="time")
        monthly_data = d.sel(time=d.time.dt.month == 3)
    elif target_mon==15: #JJA
        d = da.resample(time='QS-DEC').mean(dim="time")
        monthly_data = d.sel(time=d.time.dt.month == 6)
    elif target_mon==16: #SON
        d = da.resample(time='QS-DEC').mean(dim="time")
        monthly_data = d.sel(time=d.dt.month == 9)
    elif target_mon==17: #ANNUAL
        monthly_data = da.resample(time='AS').mean(dim="time")
    
    # select data accordingly
    monthly_data = monthly_data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds_g11 = ds_g11.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    
    # 11-year global mean time series
    g11 = ds_g11.values.squeeze()
    
    # Convert Kelvin to Celsius
    monthly_array = monthly_data.values

     # Initialize arrays for regression coefficients
    shape = monthly_array.shape[1:] # Get the shape for latitude/longitude
    A, B, C, D = np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape)

    for lat in range(shape[0]):
        for lon in range(shape[1]):
            slope, intercept, _, _, _ = linregress(g11, monthly_array[:, lat, lon])
            A[lat, lon] = intercept
            B[lat, lon] = slope

            regressed = intercept + slope * g11[:, None]
            residuals = monthly_array[:, lat, lon] - regressed.squeeze()
            tas_var = residuals ** 2

            slope_var, intercept_var, _, _, _ = linregress(g11, tas_var)
            C[lat, lon] = intercept_var
            D[lat, lon] = slope_var

    var_regression_fit = C + D * g11[:, None, None]
    var2000 = var_regression_fit[year_2000_idx, :, :]
    D_final = D / np.squeeze(var2000)
    
    # Convert A_dict and B_dict to xarray.DataArray objects
    B_da = xr.DataArray(
        data=B,
        dims=["lat", "lon"],
        coords={
            "lat": da.latitude.values,
            "lon": da.longitude.values
        },
    )
    
    # Convert A_dict and B_dict to xarray.DataArray objects
    D_da = xr.DataArray(
        data=D_final,
        dims=["lat", "lon"],
        coords={
            "lat": da.latitude.values,
            "lon": da.longitude.values
        },
    )
    
    return B_da, D_da


def get_obs_coeffs(target_mon, obs_lat, obs_lon):
    
    B_da, D_da = obs_regression(target_mon)
    
    # Create an xarray.Dataset to hold both A and B arrays
    ds_coeffs = xr.Dataset(
        {
            "aam": B_da,
            "aav": D_da,
        }
    )
    
    obs_coeffs = ds_coeffs.sel(lat=obs_lat, lon=obs_lon, method='nearest')
    
    return obs_coeffs
