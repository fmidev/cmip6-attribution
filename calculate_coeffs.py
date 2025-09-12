#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 08:53:54 2025

@author: mprantan
"""
import numpy as np
import xarray as xr
import os
from scipy.stats import linregress
import calendar
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import cftime
import glob
import pandas as pd


################################# Needed functions and arguments ######################################################################################################################
def model_mean_g11(input_dir, ssp, clim_var="tas",
                   output_file_mean=None, output_file_combined=None):
    """
    Compute the multi-model mean global mean temperature and also save
    the concatenated dataset of individual models.

    """
    # Build the search pattern for the files
    pattern = os.path.join(input_dir, f"*_g11_{ssp}*")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files found with pattern {pattern}")

    # Open all model coefficient files
    dsets = [xr.open_dataset(f) for f in files]

    # Concatenate along a new "model" dimension
    combined = xr.concat(dsets, dim="model")

    # Assign model names (based on the filename prefix)
    combined = combined.assign_coords(model=[os.path.basename(f).split("_")[0] for f in files])

    # Compute the mean across the models
    mean_ds = combined.mean(dim="model", skipna=True)

    # Save outputs if requested
    if output_file_mean is not None:
        mean_ds.to_netcdf(output_file_mean)

    if output_file_combined is not None:
        combined.to_netcdf(output_file_combined)


def model_mean_coeffs(input_dir, ssp, clim_var="tas", suffix="_regr_coeffs.nc",
                      output_file_mean=None, output_file_combined=None):
    """
    Compute the multi-model mean regression coefficients and also save
    the concatenated dataset of individual models.

    Parameters
    ----------
    input_dir : str
        Directory where the model-specific NetCDF files are stored.
    ssp : str
        Scenario name (e.g., "ssp245").
    clim_var : str, optional
        Variable name used in filenames (default: "tas").
    suffix : str, optional
        Filename suffix (default: "_regr_coeffs.nc").
    output_file_mean : str, optional
        Path to save the multi-model mean NetCDF file. If None, mean is not saved.
    output_file_combined : str, optional
        Path to save the concatenated NetCDF file with all models. If None, combined is not saved.

    Returns
    -------
    mean_ds : xr.Dataset
        Multi-model mean regression coefficients.
    combined : xr.Dataset
        Concatenated dataset of all models with a "model" dimension.
    """
    # Build the search pattern for the files
    pattern = os.path.join(input_dir, f"*_{clim_var}_{ssp}{suffix}")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No files found with pattern {pattern}")

    # Open all model coefficient files
    dsets = [xr.open_dataset(f) for f in files]

    # Concatenate along a new "model" dimension
    combined = xr.concat(dsets, dim="model")

    # Assign model names (based on the filename prefix)
    combined = combined.assign_coords(model=[os.path.basename(f).split("_")[0] for f in files])

    # Compute the mean across the models
    mean_ds = combined.mean(dim="model", skipna=True)

    # Save outputs if requested
    if output_file_mean is not None:
        mean_ds.to_netcdf(output_file_mean)

    if output_file_combined is not None:
        combined.to_netcdf(output_file_combined)


def load_cmip6_merged(scenario: str, base_path: str = "/projappl/project_2003992/cmip6/monthly_data_for_attribution/"):
    """
    Load and merge CMIP6 historical + scenario data for all available models.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g., "ssp245").
    base_path : str, optional
        Base path where the scenario folders are located.

    Returns
    -------
    xr.Dataset
        Combined dataset containing all models with full time coverage (18502100).
    """
    scen_dir = os.path.join(base_path, f"tas_{scenario}")
    hist_dir = os.path.join(base_path, "tas_historical")

    merged_datasets = []

    # Find all scenario files
    scen_files = sorted(glob.glob(os.path.join(scen_dir, "*.nc")))

    for scen_file in scen_files:
        fname = os.path.basename(scen_file)
        # Extract model name (e.g., NorESM2-MM)
        model = fname.split("_")[2]

        # Construct expected historical filename
        hist_pattern = f"tas_Amon_{model}_historical_*.nc"
        hist_files = glob.glob(os.path.join(hist_dir, hist_pattern))

        if len(hist_files) == 0:
            print(f"No historical file found for model {model}, skipping.")
            continue

        hist_file = hist_files[0]  # only one per model

        # Open datasets
        ds_hist = xr.open_dataset(hist_file)
        ds_scen = xr.open_dataset(scen_file)

        # Concatenate along time
        ds_merged = xr.concat([ds_hist, ds_scen], dim="time")

        # Add model as coordinate
        ds_merged = ds_merged.expand_dims({"model": [model]})

        merged_datasets.append(ds_merged)

    if not merged_datasets:
        raise ValueError("No matching historical-scenario pairs found!")

    # Combine all models into one dataset
    ds_all = xr.concat(merged_datasets, dim="model")

    return ds_all



def calculate_g11(path2results, ds_ts, clim_var, model_name):
    """
    Calculate 11-year running global mean temperature.

    Parameters
    ----------
    ds_ts : xr.Dataset
        Input dataset containing the climate variable.
    clim_var : str
        Name of the variable to process (e.g., 'tas').

    Returns
    -------
    xr.DataArray
        11-year running mean of the global mean temperature time series.
    """
    
    # Step 1: compute annual mean from monthly data
    annual = ds_ts.resample(time="1Y").mean()

    # Compute global mean (assumes latitude and longitude coords are 'lat' and 'lon')
    weights = np.cos(np.deg2rad(annual['lat']))
    weights /= weights.mean()  # normalize weights
    global_mean = annual.weighted(weights).mean(dim=("lat", "lon"))

    # Apply 11-year running mean (centered window)
    g11 = global_mean.rolling(time=11, center=True, min_periods=1).mean()
    
    # neglect the first and last five years
    g11 = g11.sel(time=slice("1855","2095")).drop_vars("height", errors='ignore')
    
    # save G11 to netcdf
    output_file_g11 = path2results + "/"+model_name+"_g11_"+ssp+".nc"
    g11.to_netcdf(output_file_g11)

    return g11

# A function for computing the regression coefficients A and B all coordinates during on a given day
def get_regression_coefficients(ds_var, ds_g11, day_str, clim_var):
    
    # Start year of the linear regression
    start_year = 1900
    # End year of the linear regression
    end_year = 2095
    
    years = np.arange(int(start_year), int(end_year) + 1)
    # index of year 2000
    year_2000_idx = np.where(years == 2000)[0][0]
        
    # day_str as integer
    day_int = int(day_str)

    # Extract data for the specific month, season or annual
    if day_int <=12:
        monthly_data = ds_var.sel(time=ds_var.time.dt.month == day_int)
    elif day_int==13: #DJF
        da = ds_var.resample(time='QS-DEC').mean(dim="time")
        monthly_data = da.sel(time=da.time.dt.month == 12)
        # add one year to select the year of Jan-Feb
        monthly_data['time'] = monthly_data.indexes['time'] + pd.DateOffset(years=1)
    elif day_int==14: #MAM
        da = ds_var.resample(time='QS-DEC').mean(dim="time")
        monthly_data = da.sel(time=da.time.dt.month == 3)
    elif day_int==15: #JJA
        da = ds_var.resample(time='QS-DEC').mean(dim="time")
        monthly_data = da.sel(time=da.time.dt.month == 6)
    elif day_int==16: #SON
        da = ds_var.resample(time='QS-DEC').mean(dim="time")
        monthly_data = da.sel(time=da.time.dt.month == 9)
    elif day_int==17: #ANNUAL
        monthly_data = ds_var.resample(time='AS').mean(dim="time")
    
    # select data accordingly
    monthly_data = monthly_data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds_g11 = ds_g11.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    
    # 11-year global mean time series
    g11 = ds_g11["tas"].values.squeeze()
    
    # Convert Kelvin to Celsius
    monthly_array = monthly_data[clim_var].values - 273.15

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

    return B, D_final

# A function for computing regression coefficients for the entire year
def get_regression_coefficients_year(ds_tas, ds_g11, clim_var):

    # Initialize dictionaries to store coefficients for each day of the year
    B_dict = {}
    D_dict = {}

    # Loop through each month (1 to 12), season (13 to 16) and annual (17)
    for month in range(1, 18):

        # Format the date as "MM" for the current month
        day_str = f"{month:02d}"
        print(f"Processing {day_str}")

        # Compute regression coefficients A and B for this specific day
        B, D = get_regression_coefficients(ds_tas, ds_g11, day_str, clim_var)

        # Store the coefficients in the dictionaries with day_str as the key
        B_dict[day_str] = B
        D_dict[day_str] = D

    return B_dict, D_dict


# A function for saving the regression coefficients to a netcdf file
def save_coeffs_to_netcdf(output_directory, model_name, lat, lon, B_dict, D_dict, clim_var, ssp):

     # Convert the date strings ("MM-DD") to datetime objects for the time coordinate
    start_date = datetime(2000,1,1)
    time_coords = [start_date + relativedelta(months=i) for i in range(len(B_dict))]

    # Output name
    output_name = f"{model_name}_{clim_var}_{ssp}_regr_coeffs.nc"
    output_path = os.path.join(output_directory,output_name)

    # Convert A_dict and B_dict to xarray.DataArray objects
    B_array = xr.DataArray(
        data=np.array([B_dict[f"{i:02d}"] for i in range(1, 18)]),
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lat,
            "lon": lon
        },
        attrs={"source": model_name}
    )

    D_array = xr.DataArray(
        data=np.array([D_dict[f"{i:02d}"] for i in range(1, 18)]),
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lat,
            "lon": lon
        },
        attrs={"source": model_name}
    )

    # Create an xarray.Dataset to hold both A and B arrays
    ds_coeffs = xr.Dataset(
        {
            "aam": B_array,
            "aav": D_array,
        }
    )
    # Set the model source attribute
    ds_coeffs.attrs["source"] = model_name

    # Save the dataset to a NetCDF file
    ds_coeffs.to_netcdf(output_path)
    


################################################################ Read model specific data #####################################################################################

# Emission scenario (ssp119, ssp126, ssp245, ssp370, ssp585)
ssp="ssp245"

# Climate variable (tas)
clim_var="tas"

# Paths to directories
# path where CMIP6 monthly mean temperatures are stored
path2tas_data = "/projappl/project_2003992/cmip6/monthly_data_for_attribution/"
# path to the regression coefficents files
path2results = f"/scratch/project_2005030/cmip6-attribution/regression_coefficients/{clim_var}/{ssp}"
#path to the G11 files
path2g11 = f"/scratch/project_2005030/cmip6-attribution/g11/{ssp}/"


# path to datafiles
scen_dir = os.path.join(path2tas_data, f"tas_{ssp}")
hist_dir = os.path.join(path2tas_data, "tas_historical")

# Find all scenario files
scen_files = sorted(glob.glob(os.path.join(scen_dir, "*.nc")))

# Filter out CIESM model files (due to strange precitation behaviour)
scen_files = [f for f in scen_files if "CIESM" not in os.path.basename(f)]


# Loop over the models
for scen_file in scen_files[:]:
    fname = os.path.basename(scen_file)
    # Extract model name (e.g., NorESM2-MM)
    model_name = fname.split("_")[2]

    # Construct expected historical filename
    hist_pattern = f"tas_Amon_{model_name}_historical_*.nc"
    hist_files = glob.glob(os.path.join(hist_dir, hist_pattern))

    if len(hist_files) == 0:
        print(f"No historical file found for model {model_name}, skipping.")
        continue

    hist_file = hist_files[0]  # only one per model
    
    # Open datasets
    ds_hist = xr.open_dataset(hist_file)
    ds_scen = xr.open_dataset(scen_file).sel(time=slice(None,"2100"))
    
    # Make the time axis same in hist and scenario runs
    ds_hist['time'] = pd.to_datetime(ds_hist.time.values)
    if isinstance(ds_scen.indexes["time"], xr.coding.cftimeindex.CFTimeIndex):
        datetimeindex = ds_scen.indexes['time'].to_datetimeindex()
        ds_scen['time'] = pd.to_datetime(datetimeindex)
    

    # Concatenate along time
    ds_tas = xr.concat([ds_hist, ds_scen], dim="time").drop_vars("time_bnds").squeeze()
    
    # Check that there are data until 2099
    if np.max(ds_tas.time.dt.year.values) < 2099:
        print(f"Data does not extend to the end of 21st century for model {model_name}, skipping.")
        continue


    # Calculate G11 (the covariate, 11-year global mean temperature)
    ds_g11 = calculate_g11(path2g11, ds_tas, clim_var, model_name)


    # Extract latitudes and longitudes
    lat = ds_tas["lat"].values
    lon = ds_tas["lon"].values

########################################################################################################################################################################################

################################################################### Run the regression fit code and save the resuts ####################################################################
    # Print model name to log
    print(model_name)

# Compute the coefficients B and D for months/seasons in a year
 
    B_fullyear, D_fullyear = get_regression_coefficients_year(ds_tas, ds_g11, clim_var)
    print("Regression coefficients computed!")
    print("Saving regression coefficients to a NetCDF file.")
    save_coeffs_to_netcdf(path2results, model_name, lat, lon, B_fullyear, D_fullyear, clim_var, ssp)


    print("Regression coefficients saved succesfully!")
#######################################################################################################################################################################################



# Calculate the CMIP6 multi-model mean G11 values
output_file_combined = path2g11 + "/"+ssp+"_g11_CMIP6_all_models_combined.nc"
output_file_mean = path2g11 + "/"+ssp+"_g11_CMIP6_modelmean.nc"

model_mean_g11(path2g11, ssp, clim_var="tas", output_file_mean=output_file_mean, output_file_combined = output_file_combined)


# Calculate the CMIP6 multi-model mean regression coefficiets
output_file_combined = path2results + "/tas_"+ssp+"_regr_coeffs_CMIP6_all_models_combined.nc"
output_file_mean = path2results + "/tas_"+ssp+"_regr_coeffs_CMIP6_modelmean.nc"

model_mean_coeffs(path2results, ssp, clim_var="tas", suffix="_regr_coeffs.nc",
                  output_file_mean=output_file_mean, output_file_combined = output_file_combined)