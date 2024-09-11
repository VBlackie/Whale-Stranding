import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import re
import xarray as xr

# Load the Natural Earth data from the downloaded shapefile
world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
# excel_data_df = pd.read_excel('LargeWhales-2005-2015.xlsx', sheet_name='Dataset')

# Use correct map column name
us_map = world[world['SOVEREIGNT'] == "United States of America"]

# Check the first few rows of the full world dataset
print(world.head())

# Check the shape of the dataset to confirm it's not empty
print(world.shape)

# Debugging us map
print(us_map.head())  # Check the first few rows of the US map
print(us_map.shape)   # Check the dimensions of the US map GeoDataFrame
print(world.columns)  # Check the available column names in the world dataset

fig, ax = plt.subplots(figsize=(10, 6))
us_map.plot(ax=ax, color='lightgray')
plt.show()


print(world['SOVEREIGNT'].unique())

import xarray as xr
import matplotlib.pyplot as plt

# # Open the .nc file
# dataset = xr.open_dataset('sst.ltm.1991-2020.nc')
#
# # Extract sea surface temperature (sst) variable
# sst_data = dataset['sst']
#
# # Take the mean over the time dimension to collapse it (since it's long-term mean data)
# sst_mean = sst_data.mean(dim='time')
#
# # Plot the SST mean data
# plt.figure(figsize=(10, 6))
# sst_mean.plot(cmap='coolwarm')  # Use a colormap that highlights temperature ranges
# plt.title('Long Term Mean Sea Surface Temperature (1991-2020)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.show()

import xarray as xr
import matplotlib.pyplot as plt

import xarray as xr
import matplotlib.pyplot as plt

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Open the .nc file (monthly SST data)
dataset = xr.open_dataset('sst.mnmean.nc')

# Extract the SST variable
sst_data = dataset['sst']

# Extract the year from the time coordinate
years = sst_data['time'].dt.year

# Group SST data by the extracted year and compute the mean for each year
sst_yearly_mean = sst_data.groupby(years).mean(dim=['lat', 'lon'])

# Select the range of years you're interested in (2005–2015)
sst_yearly_mean_2005_2015 = sst_yearly_mean.sel(time=slice('2005', '2015'))

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(sst_yearly_mean_2005_2015['time.year'], sst_yearly_mean_2005_2015, color='skyblue')

# Set custom Y-axis limits to zoom in on the range of values
plt.ylim([sst_yearly_mean_2005_2015.min() - 0.5, sst_yearly_mean_2005_2015.max() + 0.5])

# Set Y-axis ticks to scale by 0.1°C
min_temp = sst_yearly_mean_2005_2015.min().item()
max_temp = sst_yearly_mean_2005_2015.max().item()
plt.yticks(np.arange(min_temp, max_temp + 0.1, 0.1))

# Add title and labels
plt.title('Mean Sea Surface Temperature by Year (2005-2015)')
plt.xlabel('Year')
plt.ylabel('Mean SST (°C)')
plt.show()

#Example
# Whale strandings data (example, replace with your actual data)
whale_stranding_data = pd.DataFrame({
    'Year': [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015],
    'Strandings': [120, 130, 110, 140, 150, 160, 135, 145, 160, 155, 165]
})

# Assuming `sst_yearly_mean_2005_2015` is already calculated as shown earlier
# Convert to a pandas DataFrame
sst_data = pd.DataFrame({
    'Year': sst_yearly_mean_2005_2015['time.year'].values,
    'Mean_SST': sst_yearly_mean_2005_2015.values
})

# Merge whale strandings with SST data on Year
combined_data = pd.merge(whale_stranding_data, sst_data, on='Year')

# Calculate correlation
correlation = combined_data['Strandings'].corr(combined_data['Mean_SST'])
print(f"Correlation between Whale Strandings and SST: {correlation:.2f}")

# Scatter plot with trend line
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['Mean_SST'], combined_data['Strandings'], color='blue', label='Whale Strandings')

# Fit a linear regression line
z = np.polyfit(combined_data['Mean_SST'], combined_data['Strandings'], 1)
p = np.poly1d(z)
plt.plot(combined_data['Mean_SST'], p(combined_data['Mean_SST']), "r--", label='Trend Line')

# Labels and title
plt.title('Correlation between Whale Strandings and Sea Surface Temperature (SST)')
plt.xlabel('Mean Sea Surface Temperature (°C)')
plt.ylabel('Number of Whale Strandings')
plt.legend()
plt.show()
