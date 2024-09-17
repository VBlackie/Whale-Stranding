## Importing Necessary Libraries
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import re
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D


# Load and Prepare Data

def load_data():
    # Load the Natural Earth data and whale stranding dataset
    world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
    whale_data = pd.read_excel('LargeWhales-2005-2015.xlsx', sheet_name='Dataset')
    return world, whale_data


def clean_coordinates(whale_data):
    # Function to convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees
    def dms_to_decimal(dms_str):
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
    whale_data['Cleaned Latitude'] = whale_data['Latitude'].apply(clean_and_convert_coordinate)
    whale_data['Cleaned Longitude'] = whale_data['Longitude'].apply(clean_and_convert_coordinate)

    # Filter valid lat/long ranges, including Alaska
    whale_data = whale_data.dropna(subset=['Cleaned Latitude', 'Cleaned Longitude'])
    whale_data = whale_data[
        (whale_data['Cleaned Latitude'].between(24.396308, 71.538800)) &  # US and Alaska latitudes
        (whale_data['Cleaned Longitude'].between(-179.148909, -66.93457))  # US and Alaska longitudes
        ]

    return whale_data


# Create and Plot GeoDataFrame
def plot_whale_strandings_map(world, whale_data):
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(whale_data['Cleaned Longitude'], whale_data['Cleaned Latitude'])]
    geo_df = gpd.GeoDataFrame(whale_data, geometry=geometry)
    geo_df.set_crs(epsg=4326, inplace=True)

    # Filter the map for the United States
    us_map = world[world['SOVEREIGNT'] == "United States of America"]

    # Plot the map
    fig, ax = plt.subplots(figsize=(10, 6))
    us_map.plot(ax=ax, color='lightgray')
    geo_df.plot(ax=ax, markersize=5, color='blue', marker='o', label="Whale Strandings")

    plt.title('Whale Strandings (2005-2015) - NOAA Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()


## Whale Strandings Analysis

def plot_whale_species_and_years(whale_data):
    # Whale Species Count
    whale_type_counts = whale_data['Common Name'].value_counts()
    fig, ax1 = plt.subplots()
    whale_type_counts.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Whale Strandings by Species')
    ax1.set_xlabel('Whale Species')
    ax1.set_ylabel('Number of Strandings')

    # Whale Strandings by Year
    whale_by_year = whale_data.groupby(['Year of Observation']).count()
    fig, ax2 = plt.subplots()
    whale_by_year['National Database Number'].plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title('Whale Strandings by Year')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Strandings')
    plt.show()


# Human Interaction Analysis
def plot_human_interaction_pie_charts(whale_data):
    def autopct_with_count(pct, allvalues):
        absolute = int(round(pct / 100. * sum(allvalues)))
        return f"{pct:.1f}%\\n({absolute} cases)"

    # Count occurrences of different types of causes/interactions
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    interaction_fields = {
        'Boat Collision': ('lightblue', 'blue'),
        'Fishery Interaction': ('lightgreen', 'green'),
        'Other Human Interaction': ('orange', 'darkorange'),
        'Findings of Human Interaction': ('lightcoral', 'red')
    }

    for i, (field, colors) in enumerate(interaction_fields.items()):
        counts = whale_data[field].value_counts()
        row, col = divmod(i, 2)
        counts.plot(kind='pie', ax=ax[row, col], autopct=lambda pct: autopct_with_count(pct, counts),
                    colors=colors, title=field)
        ax[row, col].set_ylabel('')

    plt.tight_layout()
    plt.show()


# Whale Interaction Analysis
def plot_interaction_analysis(whale_data):
    # Filter data to only include strandings with and without human interaction
    non_human_strandings = whale_data[
        (whale_data['Boat Collision'] == 'N') &
        (whale_data['Fishery Interaction'] == 'N') &
        (whale_data['Other Human Interaction'] == 'N')
        ]

    human_strandings = whale_data[
        (whale_data['Boat Collision'] == 'Y') |
        (whale_data['Fishery Interaction'] == 'Y') |
        (whale_data['Other Human Interaction'] == 'Y')
        ]

    # Count strandings with and without human interaction per year
    human_strandings_by_year = human_strandings['Year of Observation'].value_counts().sort_index()
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


# Interaction Types Count
def plot_interaction_types_count(whale_data):
    # Count occurrences of each type of interaction
    boat_collision = (whale_data['Boat Collision'] == 'Y').sum()
    fishery_interaction = (whale_data['Fishery Interaction'] == 'Y').sum()
    other_human_interaction = (whale_data['Other Human Interaction'] == 'Y').sum()
    non_interaction = len(whale_data) - (boat_collision + fishery_interaction + other_human_interaction)

    # Plot the stacked bar chart for interaction types
    labels = ['Boat Collision', 'Fishery Interaction', 'Other Human Interaction', 'No Interaction']
    counts = [boat_collision, fishery_interaction, other_human_interaction, non_interaction]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts, color=['blue', 'green', 'orange', 'purple'])

    # Add labels and title
    ax.set_title('Whale Strandings by Type of Interaction')
    ax.set_ylabel('Number of Strandings')

    plt.show()


# Sea Surface Temperature (SST) Analysis
def plot_sst_analysis():
    # Open the .nc file (monthly SST data)
    dataset = xr.open_dataset('sst.mnmean.nc')
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

    # Return the sst data for further analysis
    return sst_yearly_mean_2005_2015


# Correlation with SST and 3D Plot
def plot_correlation_with_sst(whale_data, sst_data):
    # Group by year for whale strandings
    whale_by_year = whale_data.groupby(['Year of Observation']).count().reset_index()

    # Rename 'Year of Observation' to 'Year' for consistency
    whale_by_year.rename(columns={'Year of Observation': 'Year'}, inplace=True)

    # Merge whale strandings with SST data on 'Year'
    combined_data = pd.merge(whale_by_year, sst_data, on='Year')

    # Calculate correlation
    correlation = combined_data['National Database Number'].corr(combined_data['Mean_SST'])
    print(f"Correlation between Whale Strandings and SST: {correlation:.2f}")

    # Scatter plot with trend line
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_data['Mean_SST'], combined_data['National Database Number'], color='blue',
                label='Whale Strandings')

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


def plot_3d_whale_strandings_sst(whale_data, sst_data):
    # Step 1: Re-create the whale_by_year DataFrame and ensure 'Year of Observation' is included
    whale_by_year = whale_data.groupby(['Year of Observation']).count().reset_index()

    # Rename 'Year of Observation' to 'Year' for consistency
    whale_by_year.rename(columns={'Year of Observation': 'Year'}, inplace=True)

    # Merge whale strandings with SST data on 'Year'
    combined_data = pd.merge(whale_by_year, sst_data, on='Year')

    # Step 3: Create the 3D plot with SST, Whale Strandings, and Year
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # X, Y, Z coordinates
    x = combined_data['Mean_SST']  # Sea Surface Temperature (SST) on X-axis
    y = combined_data['National Database Number']  # Whale Strandings on Y-axis
    z = combined_data['Year']  # Year on Z-axis

    # Scatter plot for the 3D graph
    ax.scatter(x, y, z, c='blue', marker='o')

    # Setting labels for each axis
    ax.set_xlabel('Mean Sea Surface Temperature (°C)')
    ax.set_ylabel('Number of Whale Strandings')
    ax.set_zlabel('Year')

    # Add a title
    ax.set_title('3D Plot of Whale Strandings, SST, and Year')

    plt.show()


# Whale Stranding Analysis - Main Function
def main():
    # Load data
    world, whale_data = load_data()

    # Clean data
    whale_data_cleaned = clean_coordinates(whale_data)

    # Plot GeoData
    plot_whale_strandings_map(world, whale_data_cleaned)
    plot_whale_species_and_years(whale_data_cleaned)
    plot_human_interaction_pie_charts(whale_data_cleaned)

    # Plot Interaction Analysis
    plot_interaction_analysis(whale_data_cleaned)
    plot_interaction_types_count(whale_data_cleaned)

    # Sea Surface Temperature (SST) Analysis
    sst_yearly_mean_2005_2015 = plot_sst_analysis()

    # Prepare SST DataFrame for further analysis
    sst_data = pd.DataFrame({
        'Year': sst_yearly_mean_2005_2015['time.year'].values,
        'Mean_SST': sst_yearly_mean_2005_2015.values
    })

    # Plot Correlation with SST
    plot_correlation_with_sst(whale_data_cleaned, sst_data)

    # 3D Plot of Whale Strandings, SST, and Year
    plot_3d_whale_strandings_sst(whale_data_cleaned, sst_data)


# Main execution
if __name__ == "__main__":
    main()
