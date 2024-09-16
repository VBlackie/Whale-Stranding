import pandas as pd
import xarray as xr
import numpy as np
import re


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


# 1. Process Whale Strandings Data
def clean_whale_data():
    # Read the Excel file
    df_whales = pd.read_excel('LargeWhales-2005-2015.xlsx', sheet_name='Dataset')

    # Clean Latitude and Longitude
    df_whales['Cleaned Latitude'] = df_whales['Latitude'].apply(clean_and_convert_coordinate)
    df_whales['Cleaned Longitude'] = df_whales['Longitude'].apply(clean_and_convert_coordinate)

    # Ensure a consistent Year column
    df_whales['Year'] = df_whales['Year of Observation']

    # Create the 'Has_Interaction' flag
    df_whales['Has_Interaction'] = df_whales.apply(
        lambda row: 'Y' if row['Boat Collision'] == 'Y' or row['Fishery Interaction'] == 'Y' or row[
            'Other Human Interaction'] == 'Y' else 'N',
        axis=1
    )

    # Save to a new Excel file
    df_whales.to_excel('CleanedWhalesData.xlsx', index=False)


# 2. Process Sea Surface Temperature (SST) Data
def process_sst_data():
    # Open the .nc file
    dataset = xr.open_dataset('sst.mnmean.nc')

    # Extract the SST variable
    sst_data = dataset['sst']

    # Extract the year from the time coordinate
    sst_df = sst_data.to_dataframe().reset_index()

    # Extract year for easier processing
    sst_df['Year'] = sst_df['time'].dt.year

    # Select the range of years you're interested in (2005–2015)
    sst_df = sst_df[(sst_df['Year'] >= 2005) & (sst_df['Year'] <= 2015)]

    # Calculate the mean SST for each year
    yearly_sst_df = sst_df.groupby('Year')['sst'].mean().reset_index()
    yearly_sst_df.columns = ['Year', 'Mean_SST']

    # Save only the yearly SST data to an Excel file
    with pd.ExcelWriter('sst_cleaned.xlsx') as writer:
        # Save only the yearly SST data
        yearly_sst_df.to_excel(writer, index=False, sheet_name='Yearly_SST')


# Execute the functions
# clean_whale_data()
process_sst_data()

