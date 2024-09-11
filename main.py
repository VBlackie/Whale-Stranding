import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import re
import numpy as np
import xarray as xr

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

# Grouping whale by type and counting them
# whale_type_counter = excel_data_df.groupby(['Common Name']).count()
# print(whale_type_counter)
# print(whale_type_counter.head())
# print(whale_type_counter.columns)
# Using value counts for whale type
whale_type_counts = excel_data_df['Common Name'].value_counts()

# Grouping whale death by year
whale_by_year = excel_data_df.groupby(['Year of Observation']).count()
print(whale_by_year)
print(whale_by_year.head())
print(whale_by_year.columns)



# Plot Whale Species Count
fig, ax1 = plt.subplots()
whale_type_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Whale Strandings by Species')
ax1.set_xlabel('Whale Species')
ax1.set_ylabel('Number of Strandings')

# Plot Whale Strandings by Year
fig, ax2 = plt.subplots()
whale_by_year['National Database Number'].plot(kind='bar', ax=ax2, color='orange')
ax2.set_title('Whale Strandings by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of Strandings')

plt.show()

# Function to display both count and percentage in the pie chart
def autopct_with_count(pct, allvalues):
    absolute = int(round(pct/100.*sum(allvalues)))
    return f"{pct:.1f}%\n({absolute} cases)"

# Count occurrences of different types of causes/interactions
boat_collision_counts = excel_data_df['Boat Collision'].value_counts()
fishery_interaction_counts = excel_data_df['Fishery Interaction'].value_counts()
other_human_interaction_counts = excel_data_df['Other Human Interaction'].value_counts()
human_interaction_findings = excel_data_df['Findings of Human Interaction'].value_counts()

# Create pie charts for each type of cause
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Boat collisions
boat_collision_counts.plot(kind='pie', ax=ax[0, 0], autopct=lambda pct: autopct_with_count(pct, boat_collision_counts),
                           colors=['lightblue', 'blue'], title='Boat Collisions')
ax[0, 0].set_ylabel('')  # Remove ylabel for pie charts

# Fishery interactions
fishery_interaction_counts.plot(kind='pie', ax=ax[0, 1], autopct=lambda pct: autopct_with_count(pct, fishery_interaction_counts),
                                colors=['lightgreen', 'green'], title='Fishery Interactions')
ax[0, 1].set_ylabel('')

# Other human interactions
other_human_interaction_counts.plot(kind='pie', ax=ax[1, 0], autopct=lambda pct: autopct_with_count(pct, other_human_interaction_counts),
                                    colors=['orange', 'darkorange'], title='Other Human Interactions')
ax[1, 0].set_ylabel('')

# Findings of human interaction
human_interaction_findings.plot(kind='pie', ax=ax[1, 1], autopct=lambda pct: autopct_with_count(pct, human_interaction_findings),
                                colors=['lightcoral', 'red'], title='Findings of Human Interaction')
ax[1, 1].set_ylabel('')

plt.tight_layout()
plt.show()

# Filter data to only include strandings with no human interaction
non_human_strandings = cleaned_data[
    (cleaned_data['Boat Collision'] == 'N') &
    (cleaned_data['Fishery Interaction'] == 'N') &
    (cleaned_data['Other Human Interaction'] == 'N')
]

# Count strandings with human interaction per year
human_strandings = cleaned_data[
    (cleaned_data['Boat Collision'] == 'Y') |
    (cleaned_data['Fishery Interaction'] == 'Y') |
    (cleaned_data['Other Human Interaction'] == 'Y')
]
human_strandings_by_year = human_strandings['Year of Observation'].value_counts().sort_index()

# Count strandings without human interaction per year (from previous filtering)
non_human_strandings_by_year = non_human_strandings['Year of Observation'].value_counts().sort_index()

# Ensure both Series have the same index (fill missing years with 0)
years = np.union1d(human_strandings_by_year.index, non_human_strandings_by_year.index)
human_strandings_by_year = human_strandings_by_year.reindex(years, fill_value=0)
non_human_strandings_by_year = non_human_strandings_by_year.reindex(years, fill_value=0)

# Create a stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(human_strandings_by_year.index, human_strandings_by_year, label='Human Interaction', color='orange')
ax.bar(non_human_strandings_by_year.index, non_human_strandings_by_year, bottom=human_strandings_by_year,
       label='Non-Human Interaction', color='purple')

# Add labels and title
ax.set_title('Whale Strandings by Year (Human vs Non-Human Interaction)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Strandings')
ax.legend()

plt.show()

# Count occurrences of each type of interaction
boat_collision = (excel_data_df['Boat Collision'] == 'Y').sum()
fishery_interaction = (excel_data_df['Fishery Interaction'] == 'Y').sum()
other_human_interaction = (excel_data_df['Other Human Interaction'] == 'Y').sum()
non_interaction = len(excel_data_df) - (boat_collision + fishery_interaction + other_human_interaction)

# Plot the stacked bar chart for interaction types
labels = ['Boat Collision', 'Fishery Interaction', 'Other Human Interaction', 'No Interaction']
counts = [boat_collision, fishery_interaction, other_human_interaction, non_interaction]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, counts, color=['blue', 'green', 'orange', 'purple'])

# Add labels and title
ax.set_title('Whale Strandings by Type of Interaction')
ax.set_ylabel('Number of Strandings')

plt.show()

# Extracting temperatures per year data
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

# Assuming `sst_yearly_mean_2005_2015` is already calculated as shown earlier
# Convert to a pandas DataFrame
sst_data = pd.DataFrame({
    'Year': sst_yearly_mean_2005_2015['time.year'].values,
    'Mean_SST': sst_yearly_mean_2005_2015.values
})

print(whale_by_year.columns)
# Rename 'Year of Observation' to 'Year' (for whale strandings data)
whale_by_year.rename(columns={'Year of Observation': 'Year'}, inplace=True)
print(whale_by_year.columns)

# Merge whale strandings with SST data on Year
combined_data = pd.merge(whale_by_year, sst_data, left_on='Year of Observation', right_on='Year')

# Calculate correlation
correlation = combined_data['National Database Number'].corr(combined_data['Mean_SST'])
print(f"Correlation between Whale Strandings and SST: {correlation:.2f}")

# Scatter plot with trend line
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['Mean_SST'], combined_data['National Database Number'], color='blue', label='Whale Strandings')

# Fit a linear regression line
z = np.polyfit(combined_data['Mean_SST'], combined_data['National Database Number'], 1)
p = np.poly1d(z)
plt.plot(combined_data['Mean_SST'], p(combined_data['Mean_SST']), "r--", label='Trend Line')

# Labels and title
plt.title('Correlation between Whale Strandings and Sea Surface Temperature (SST)')
plt.xlabel('Mean Sea Surface Temperature (°C)')
plt.ylabel('Number of Whale Strandings')
plt.legend()
plt.show()


