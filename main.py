import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import re

# Load the Natural Earth data from the downloaded shapefile
world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
excel_data_df = pd.read_excel('LargeWhales-2005-2015.xlsx', sheet_name='Dataset')

# Filter the map for the United States
us_map = world[world['SOVEREIGNT'] == "United States of America"]

# Function to convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees
def dms_to_decimal(dms_str):
    """Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees."""
    dms_str = dms_str.replace('(', '').replace(')', '').replace(',', '').replace('/', ' ')
    parts = re.split(r'[^\d.]+', dms_str)
    parts = [p for p in parts if p]

    if len(parts) == 3:  # Degrees, minutes, seconds
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return degrees + (minutes / 60) + (seconds / 3600)
    elif len(parts) == 2:  # Degrees and minutes
        degrees = float(parts[0])
        minutes = float(parts[1])
        return degrees + (minutes / 60)
    elif len(parts) == 1:  # Decimal degrees
        return float(parts[0])
    else:
        return None

# Function to clean and convert coordinates
def clean_and_convert_coordinate(coord):
    if pd.isna(coord):
        return None
    coord = str(coord).replace('(', '').replace(')', '').replace(',', '').replace('/', ' ')
    if coord.count('.') > 1:
        return None
    try:
        return float(coord)
    except ValueError:
        return dms_to_decimal(coord)

# Clean and convert coordinates
excel_data_df['Cleaned Latitude'] = excel_data_df['Latitude'].apply(clean_and_convert_coordinate)
excel_data_df['Cleaned Longitude'] = excel_data_df['Longitude'].apply(clean_and_convert_coordinate)

# Drop rows where coordinates couldn't be cleaned and filter valid lat/long ranges
cleaned_data = excel_data_df.dropna(subset=['Cleaned Latitude', 'Cleaned Longitude'])
# Filter out invalid latitudes and longitudes, including Alaska
cleaned_data = cleaned_data[
    (cleaned_data['Cleaned Latitude'].between(24.396308, 71.538800)) &  # Include US and Alaska latitudes
    (cleaned_data['Cleaned Longitude'].between(-179.148909, -66.93457))  # Include US and Alaska longitudes
]

# Create the geometry column for GeoPandas
geometry = [Point(xy) for xy in zip(cleaned_data['Cleaned Longitude'], cleaned_data['Cleaned Latitude'])]
geo_df = gpd.GeoDataFrame(cleaned_data, geometry=geometry)

# Set the CRS to EPSG:4326
geo_df.set_crs(epsg=4326, inplace=True)

# Latitude range: including Alaska and mainland US
min_lat, max_lat = 24.396308, 71.538800  # 24.396308 is the southernmost point of the US mainland, 71.5 is for Alaska

# Longitude range: including Alaska (Aleutian Islands) and mainland US
min_lon, max_lon = -179.148909, -66.93457  # -179.1 for Alaska, -66.93457 for eastern US mainland

us_whale_data = geo_df[
    (geo_df['Cleaned Latitude'].between(min_lat, max_lat)) &
    (geo_df['Cleaned Longitude'].between(min_lon, max_lon))
]

# Plot the US map and whale strandings
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the US map
us_map.plot(ax=ax, color='lightgray')

# Plot the whale strandings on top of the US map
us_whale_data.plot(ax=ax, markersize=5, color='blue', marker='o', label="Whale Strandings")

# Add title and labels
plt.title('Whale Strandings (2005-2015) - NOAA Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Show the plot
plt.show()

