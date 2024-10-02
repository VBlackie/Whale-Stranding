# Whale Stranding Analysis Project 🐋📊

This project explores whale stranding events between 2005 and 2015 using data from NOAA. We delve into key factors that may influence strandings, such as human interaction and environmental changes like sea surface temperature (SST). The analysis combines data visualization, geographic mapping, and statistical methods to better understand patterns of whale strandings and their potential causes.
** You can read the Project here [a link](https://github.com/VBlackie/Whale-Stranding/blob/master/Whale%20Stranding.ipynb) **

## Project Structure 🗂️

The project is divided into several parts:

### 1. **Introduction & Motivation**  
This section explores the motivation behind the project. It starts with curiosity about animal behavior and the concept of animal suicide. Specifically, the question arose: **"Do whales intentionally strand themselves?"**  
Using descriptive statistics and data analysis, we seek to explore possible trends and connections that could help us better understand the phenomenon of whale strandings.

### 2. **Datasets**  
We use two primary datasets:
- **NOAA Whale Stranding Data (2005–2015):** Provides information about whale stranding incidents, whale species, and human interaction factors.
- **Sea Surface Temperature (SST) Data (2005–2015):** SST data was used to see if there is any correlation between temperature fluctuations and the number of whale strandings.

### 3. **Visualizations & Analysis**  
We generated several plots to analyze the data:
- **Geographical Map of Whale Strandings:** Visualizing the locations of strandings across the United States, including Alaska.
- **Whale Strandings by Species and Year:** Showing trends over time and highlighting which species are most affected.
- **Human Interaction Pie Charts:** Breaking down the percentage of strandings that involved boat collisions, fishery interactions, and other human impacts.
- **Whale Strandings vs Sea Surface Temperature:** Examining any potential correlations between rising temperatures and increased strandings.
- **3D Plot of Strandings, SST, and Year:** A dynamic 3D visualization to further explore the relationship between these variables.

### 4. **Findings**  
- **Species Insights:** Humpback and gray whales were the most affected species by strandings.
- **Geographic Patterns:** Strandings were concentrated along the Pacific coast and in Alaska, with a large number of incidents occurring along the eastern U.S. coast.
- **Temperature and Strandings:** There is a weak positive correlation between sea surface temperature and the number of whale strandings.
- **Human Interaction:** While human-related factors are present, most strandings do not appear to be directly caused by human activity.

## How to Run the Code 🖥️

1. Clone the repository:
   ```bash
   git clone https://github.com/VBlackie/Whale-Stranding-analysis.git
   cd whale-stranding-analysis

2. Set up the virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   .\venv\Scripts\activate  # For Windows

3. Install the requirements.
   ```bash
   pip install -r requirements.txt

4. Run the jupyter Notebook
   ```bash
   jupyter notebook Whale_Stranding.ipynb

## Requirements 📦
The project relies on the following libraries:

- pandas
- geopandas
- matplotlib
- xarray
- shapely
- mpl_toolkits.mplot3d
These are listed in the requirements.txt file, and they can be installed using pip.

## Conclusion
This project presents an exploratory data analysis (EDA) into the patterns of whale strandings over a decade. By incorporating both geographic and statistical tools, we uncover several insights, but also highlight the complexity of the issue. Further research could involve more data or looking into additional environmental factors to better understand this phenomenon.

## Acknowledgements 🙏
Special thanks to NOAA for making the whale stranding data available, and to the open-source community for the development of the tools that made this analysis possible.
