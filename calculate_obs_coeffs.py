
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
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

import pandas as pd


################################# Needed functions and arguments ######################################################################################################################
# Convert fractional year to datetime
def fractional_year_to_datetime(year_frac):
    year = int(year_frac)
    remainder = year_frac - year
    # approximate month by multiplying with 12
    month = int(np.floor(remainder * 12)) + 1
    # ensure month stays within 112
    month = min(month, 12)
    return pd.Timestamp(year=year, month=month, day=1)

def fix_fractional_year_time(ds):
    """Convert fractional-year time coordinate (e.g. Berkeley Earth) to datetime."""
    if not np.issubdtype(ds.time.dtype, np.floating):
        # Already datetime-like, nothing to do
        return ds
    
    t = ds.time.values

    # Conversion from fractional year  datetime
    def fractional_year_to_datetime(year_frac):
        year = int(year_frac)
        remainder = year_frac - year
        # month 112
        month = int(np.floor(remainder * 12)) + 1
        month = min(month, 12)
        # Berkeley Earth uses mid-month timestamps  set to day=15
        return pd.Timestamp(year=year, month=month, day=15)

    time_index = pd.to_datetime([fractional_year_to_datetime(v) for v in t])

    return ds.assign_coords(time=time_index)

def rename_lat_lon(ds):
    rename_dict = {}
    if "lat" in ds.dims or "lat" in ds.coords:
        rename_dict["lat"] = "latitude"
    if "lon" in ds.dims or "lon" in ds.coords:
        rename_dict["lon"] = "longitude"
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_dict["valid_time"] = "time"
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds

def calculate_g11(ds_tas, clim_var):
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
    annual = ds_tas[clim_var].resample(time="1YE").mean()
    
    # Last available observational year
    last_year = annual.time[-1].dt.year.item()
    
    # Number of future years needed for an 11-year centered running mean
    window = 11
    n_future = window // 2 + 1

    # Compute global mean (assumes latitude and longitude coords are 'lat' and 'lon')
    weights = np.cos(np.deg2rad(annual['latitude']))
    weights /= weights.mean()  # normalize weights
    global_mean = annual.weighted(weights).mean(dim=("latitude", "longitude"))
    
    # Linear extrapolation for the next 6 years using last 30 years
    last_years = global_mean.time[-30:].dt.year
    last_values = global_mean.values[-30:]
    coeff = np.polyfit(last_years, last_values, 1)  # linear fit
    # Automatically generate future years
    future_years = np.arange(last_year + 1,last_year + n_future + 1)

    extrapolated_values = np.polyval(coeff, future_years)
    
    # Create DataArray
    future_time = pd.to_datetime([f"{year}-12-31" for year in future_years])


    # Create new DataArray for 2025
    da_future = xr.DataArray(extrapolated_values,
                           dims=["time"],coords={"time": future_time})

    # Concatenate along the time dimension
    da_extended = xr.concat([global_mean, da_future], dim="time")

    # Apply 11-year running mean (centered window)
    g11 = da_extended.rolling(time=window, center=True, min_periods=window).mean()
    
    # neglect the first and last five years
    g11 = g11.drop_vars("height", errors='ignore')
    
    fig, ax = plt.subplots(1,1, dpi=300, figsize=(9,6))
    ax.plot(da_extended.time.dt.year, da_extended, linewidth=0.8)
    ax.plot(g11.time.dt.year, g11, color='k', linewidth=2)
    ax.set_ylabel('Global mean temperature\nrelative to 2000 [°C]', fontsize=14)
    ax.tick_params(labelsize=14); ax.grid(linestyle='--')
    plt.savefig("./global_temp.png",dpi=300, bbox_inches="tight")
    
    return g11

# A function for computing the regression coefficients A and B all coordinates during on a given day
def get_regression_coefficients(ds_var, ds_g11, day_str, clim_var):
    
    # Start year of the linear regression
    start_year = max(ds_g11.time.dt.year[0].values, 1895) + 5
    # End year of the linear regression: with 11-year rolling average, 
    # take the central year of the last 11-year period
    end_year = 2020
    
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
        monthly_data = ds_var.resample(time='YS').mean(dim="time")
    
    # select data accordingly
    monthly_data = monthly_data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    ds_g11 = ds_g11.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    
    # 11-year global mean time series
    g11 = ds_g11.values.squeeze()
    
    # Convert Kelvin to Celsius
    monthly_array = monthly_data[clim_var].values

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
    
    
    lat_idx = abs(monthly_data["latitude"]-67.4).argmin().item()
    lon_idx = abs(monthly_data["longitude"]-26.6).argmin().item()
    
    x = g11
    y=monthly_array[:, lat_idx, lon_idx]
    
    if day_str=="09":
        fig, ax = plt.subplots(1,1, dpi=300, figsize=(9,6))
        ax.scatter(x, y, )
        # lasketaan lineaarinen regressio
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # lasketaan trendiviiva
        y_trend = intercept + slope * x
        # piirretään trendiviiva
        ax.plot(x, y_trend, color='red', label=f'Trend: y={slope:.2f}x + {intercept:.2f}')
        ax.set_ylabel('September temperature anomaly [°C]', fontsize=14)
        ax.set_xlabel('11-year global mean temperature anomaly [°C]', fontsize=14)
        ax.tick_params(labelsize=14); ax.grid(linestyle='--'); ax.legend()
        plt.savefig("./linear_scatter.png",dpi=300, bbox_inches="tight")

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
def save_coeffs_to_netcdf(output_directory, lat, lon, B_dict, D_dict, clim_var, source):

     # Convert the date strings ("MM-DD") to datetime objects for the time coordinate
    start_date = datetime(2000,1,1)
    time_coords = [start_date + relativedelta(months=i) for i in range(len(B_dict))]

    # Output name
    output_name = "obs_regr_coeffs_"+source+".nc"
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
        attrs={"source": source}
    )

    D_array = xr.DataArray(
        data=np.array([D_dict[f"{i:02d}"] for i in range(1, 18)]),
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lat,
            "lon": lon
        },
        attrs={"source": source}
    )

    # Create an xarray.Dataset to hold both A and B arrays
    ds_coeffs = xr.Dataset(
        {
            "aam": B_array,
            "aav": D_array,
        }
    )
    # Set the model source attribute
    ds_coeffs.attrs["source"] = source

    # Save the dataset to a NetCDF file
    ds_coeffs.to_netcdf(output_path)
    


################################################################ Read model specific data #####################################################################################

source = "BEST"
# Climate variable (tas)
variables = {
    "HadCRUT5":"tas_mean",
    "GISTEMP":"tempanomaly",
    "BEST":"temperature",
    "ERA5":"t2m"
    }

clim_var=variables[source]

# Paths to directories
# path where HadCRUT5 monthly mean temperatures are stored
path2tas_data = "/Users/rantanem/Documents/python/cmip6-attribution/input_data/"
# path to the regression coefficents files
path2results =  "/Users/rantanem/Documents/python/cmip6-attribution/input_data/"


# open dataset
files = {
    "HadCRUT5":"HadCRUT.5.1.0.0.analysis.anomalies.ensemble_mean.nc",
    "GISTEMP":"gistemp1200_GHCNv4_ERSSTv5.nc",
    "BEST":"Global_TAVG_Gridded_1deg.nc",
    "ERA5":"era5_t2m_1940-2025.nc"}

ds_tas = xr.open_dataset(path2tas_data + files[source])

# rename coordinates consistently
ds_tas = rename_lat_lon(ds_tas)

# Fix time axis
ds_tas = fix_fractional_year_time(ds_tas)

# select data up to 2025
ds_tas = ds_tas.sel(time=slice(None, "2025"))

# Calculate G11 (the covariate, 11-year global mean temperature)
ds_g11 = calculate_g11(ds_tas, clim_var)


# Extract latitudes and longitudes
lat = ds_tas["latitude"].values
lon = ds_tas["longitude"].values

########################################################################################################################################################################################

################################################################### Run the regression fit code and save the resuts ####################################################################

# Compute the coefficients B and D for months/seasons in a year
 
B_fullyear, D_fullyear = get_regression_coefficients_year(ds_tas, ds_g11, clim_var)
print("Regression coefficients computed!")
print("Saving regression coefficients to a NetCDF file.")
save_coeffs_to_netcdf(path2results, lat, lon, B_fullyear, D_fullyear, clim_var, source)


print("Regression coefficients saved succesfully!")
#######################################################################################################################################################################################

# Output observation-based G11
output_file_mean = path2results + "/g11_"+source+".nc"
ds_g11.to_netcdf(output_file_mean)
