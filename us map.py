import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import re

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
