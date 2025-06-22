# import pandas as pd
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from collections import Counter as c
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pickle
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
#
# #Reading the dataset
# data=pd.read_csv('Indicators.csv')
# print(data.shape)
# print(data.head(10))
#
# #Check unique values in dataset
# countries=data['CountryName'].unique().tolist()
# print(len(countries))
# countryCodes=data['CountryCode'].unique().tolist()
# len(countryCodes)
# indicators=data['IndicatorName'].unique().tolist()
# len(indicators)
# years=data['Year'].unique().tolist()
# len(years)
# print(min(years), " to ", max(years))
#
#
# #CO2 emissions of countries
# # select CO2 emissions for the Arab
# hist_indicator1 = r'CO2 emissions \(metric'
# hist_country1   = 'ARB'
#
# mask11 = data['IndicatorName'].str.contains(hist_indicator1)
# mask22 = data['CountryCode'].str.contains(hist_country1)
#
# # stage is just those indicators matching the ARB for country code and CO2 emissions over time.
# stage1 = data[mask11 & mask22]
#
# print(stage1.head())
#
# # select CO2 emissions for the Barbados
# hist_indicator3 = r'CO2 emissions \(metric'
# hist_country3   = 'BRB'
#
# mask32 = data['IndicatorName'].str.contains(hist_indicator3)
# mask42 = data['CountryCode'].str.contains(hist_country3)
#
# # stage is just those indicators matching the BRB for country code and CO2 emissions over time.
# stage3 = data[mask32 & mask42]
#
# print(stage3.head())
#
# # select CO2 emissions for the India
# hist_indicator2 = r'CO2 emissions \(metric'
# hist_country2   = 'IND'
#
# mask22 = data['IndicatorName'].str.contains(hist_indicator2)
# mask32 = data['CountryCode'].str.contains(hist_country2)
#
# # stage is just those indicators matching the IND for country code and CO2 emissions over time
# stage2 = data[mask22 & mask32]
#
# print(stage2.head())
#
# # select CO2 emissions for the Singapore
# hist_indicator4 = r'CO2 emissions \(metric'
# hist_country4   = 'SGP'
#
# mask42 = data['IndicatorName'].str.contains(hist_indicator4)
# mask52 = data['CountryCode'].str.contains(hist_country4)
#
# # stage is just those indicators matching the SGP for country code and CO2 emissions over time
# stage4 = data[mask42 & mask52]
#
# print(stage4.head())
# # select CO2 emissions for the United States
# hist_indicator = r'CO2 emissions \(metric'
# hist_country   = 'USA'
#
# mask1 = data['IndicatorName'].str.contains(hist_indicator)
# mask2 = data['CountryCode'].str.contains(hist_country)
#
# # stage is just those indicators matching the USA for country code and CO2 emissions over time
# stage = data[mask1 & mask2]
#
# print(stage.head())
#
# #Understanding Data Type and Summary of features
# data.info()
# data.describe()
#
# #Observing target, numerical and atergorical columns
# np.unique(data.dtypes, return_counts=True)
# # fetching all the object or categorical type of columns from our data and we are storing it as set in a variable cat.
# cat = data.dtypes[data.dtypes == '0'].index.tolist()
# # detect which columns are categorical and which are not
#
# print(f"CAT: {cat}")
#
# for i in cat:
#     print('RESULT')
#     print("Column: ", i)
#     print("Count of classes:", data[i].nunique())
#     print(c(data[i]))
#     print('HI')
#
# #Taking care of missing data
# print(data.isnull().any())
# print(data.isnull().sum())
#
# #Label encoding
# data1=data.copy()
# x='*'
# for i in cat:
#     print("LABEL ENCODING OF:", i)
#     LE=LabelEncoder()
#     print(c[data[i]])
#     data[i]=LE.fit_transform(data[i])
#     print(x*100)
#
# #Data visualisation
#
# #Bar graph
# years=stage['Year'].values
# co2=stage['Value'].values
# plt.bar(years, co2)
# plt.show()
#
# #switch to line plot
# plt.plot(stage['Year'].values, stage['Value'].values)
#
# #Label the axes
# plt.xlabel('Year')
# plt.ylabel(stage['IndicatorName'].iloc[0])
#
# #Label the figure
# plt.title('CO2 Emissions in USA')
#
# #Staring the y axis at 0
# plt.show()
#
#
# #Histogram of the data
# plt.hist(hist_data, 10, normed=False, facecolor='red')
# plt.xlabel(stage['IndicatorName'].iloc[0])
# plt.ylabel('Number of years')
# plt.title('Histogram Example')
# plt.grid(True)
# plt.show()
#
# #Relationship between GDP and CO2 emissions in USA
# hist_indicator=r'GDP per capita\(constant 2005'
# hist_country='USA'
# mask1=data['IndicatorName'].str.contains(hist_indicator)
# mask2=data['CountryCode'].str.contains(hist_country)
#
# gdp_stage=data[mask1 & mask2]
# gdp_stage.head()
#
# #switch to a line plot
# plt.plot(gdp_stage['Year'].values, gdp_stage['Value'].values)
#
# #Label the axes
# plt.xlabel('Year')
# plt.ylabel(gdp_stage['IndicatorName'].iloc[0])
# plt.title('GDP Per Capita in USA')
# plt.show()
#
# #ScatterPlot
# fig, axis = plt.subplots()
# # Grid lines, Xticks, XLabel, YLabel
#
# axis.yaxis.grid(True)
# axis.set_title('CO2 Emissions vs. GDP \\(per capita\\)', fontsize=10)
# axis.set_xlabel(gdp_stage_trunc['IndicatorName'].iloc[10], fontsize=10)
# axis.set_ylabel(stage['IndicatorName'].iloc[0], fontsize=10)
#
# X = gdp_stage_trunc['Value']
# Y = stage['Value']
#
# axis.scatter(X, Y)
# plt.show()
#
# #Heatmap
# corr=data.corr()
# plt.subplots(figsize=(16,16))
# sns.heatmap(corr, annot=True, square=True)
# plt.title("Correlation matrix of numerical features")
# plt.tight_layout()
# plt.show()
#
# #Splitting the dataset into dependent and independent variables
# x=data.drop(['value', 'IndicatorCode'], axis=1)
# x=pd.Dataframe(x)
# y=data['Value']#dependent feature
# y=pd.DataFrame(y)
#
# #Splitting into train and test
# x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)
# print(x_train.shape)
# print(x_test.shape)




#GPT refined
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os # Import the os module for directory operations

# from sklearn.metrics import accuracy_score # Not used in the provided snippet, so commented out
# import pickle # Not used in the provided snippet, so commented out
# from sklearn.ensemble import RandomForestRegressor # Not used for actual training in snippet, so commented out

# --- Create the 'Graphs' directory if it doesn't exist ---
graphs_dir = 'Graphs'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)
    print(f"Created directory: {graphs_dir}")
else:
    print(f"Directory '{graphs_dir}' already exists.")

# --- 1. Data Loading and Initial Exploration ---
print("--- Loading Data ---")
data = pd.read_csv('Indicators.csv')
print(f"Data shape: {data.shape}")
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\n--- Unique Values in Key Columns ---")
countries = data['CountryName'].unique().tolist()
print(f"Number of unique countries: {len(countries)}")

country_codes = data['CountryCode'].unique().tolist()
print(f"Number of unique country codes: {len(country_codes)}")

indicators = data['IndicatorName'].unique().tolist()
print(f"Number of unique indicators: {len(indicators)}")

years = data['Year'].unique().tolist()
print(f"Year range: {min(years)} to {max(years)}")

print("\n--- Data Information and Summary Statistics ---")
data.info()
print("\nDescriptive statistics for numerical columns:")
print(data.describe())

# --- 2. Categorical Column Analysis and Missing Data ---
print("\n--- Categorical Columns Analysis ---")
# Detect categorical columns (object type)
categorical_cols = data.select_dtypes(include='object').columns.tolist()

if not categorical_cols:
    print("No object-type columns found in the dataset.")
else:
    print(f"Categorical columns identified: {categorical_cols}")
    for col in categorical_cols:
        print(f"\nColumn: '{col}'")
        print(f"Number of unique classes: {data[col].nunique()}")
        print("Value Counts:")
        print(Counter(data[col]))

print("\n--- Missing Data Check ---")
print("Are there any missing values in each column?")
print(data.isnull().any())
print("\nTotal missing values per column:")
print(data.isnull().sum())


# --- 3. Label Encoding (Prepare for potential model training) ---
# Create a copy of the dataframe if you plan to modify it for modeling
# and want to keep the original for plotting with original string values.
data_encoded = data.copy()

print("\n--- Applying Label Encoding to Categorical Columns (for modeling if needed) ---")
for col in categorical_cols:
    print(f"Label Encoding for column: '{col}'")
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col])
    # Optionally print value counts after encoding to verify
    # print(Counter(data_encoded[col]))
    print("-" * 50) # Separator

print("\nFirst 5 rows of the encoded dataset (for comparison):")
print(data_encoded.head())

# --- 4. Data Visualization ---

# Define the specific indicators we are interested in
CO2_INDICATOR_CODE = 'EN.ATM.CO2E.PC' # CO2 emissions (metric tons per capita)
GDP_INDICATOR_REGEX = r'GDP per capita \(constant 2005' # Regex for GDP per capita

def plot_co2_emissions_line(dataframe, country_code, indicator_code):
    """
    Plots CO2 emissions (metric tons per capita) for a given country code as a line plot.
    Saves the plot to the 'Graphs' folder.
    """
    mask_indicator = dataframe['IndicatorCode'] == indicator_code
    mask_country = dataframe['CountryCode'] == country_code

    country_data = dataframe[mask_indicator & mask_country]

    if country_data.empty:
        print(f"No CO2 data found for {country_code} with indicator {indicator_code}. Skipping line plot.")
        return

    country_name = country_data['CountryName'].iloc[0]
    indicator_name = country_data['IndicatorName'].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(country_data['Year'].values, country_data['Value'].values, marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel(indicator_name)
    plt.title(f'CO2 Emissions in {country_name} ({indicator_code})')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"{graphs_dir}/{country_code}_CO2_Emissions_Line_Plot.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close() # Close the plot to free memory

def plot_co2_emissions_bar(dataframe, country_code, indicator_code):
    """
    Plots CO2 emissions (metric tons per capita) for a given country code as a bar plot.
    Saves the plot to the 'Graphs' folder.
    """
    mask_indicator = dataframe['IndicatorCode'] == indicator_code
    mask_country = dataframe['CountryCode'] == country_code

    country_data = dataframe[mask_indicator & mask_country]

    if country_data.empty:
        print(f"No CO2 data found for {country_code} with indicator {indicator_code}. Skipping bar plot.")
        return

    country_name = country_data['CountryName'].iloc[0]
    indicator_name = country_data['IndicatorName'].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.bar(country_data['Year'].values, country_data['Value'].values, color='skyblue')
    plt.xlabel('Year')
    plt.ylabel(indicator_name)
    plt.title(f'CO2 Emissions in {country_name} ({indicator_code}) - Bar Graph')
    plt.grid(axis='y') # Only horizontal grid lines for bar graph
    plt.tight_layout()

    # Save the figure
    filename = f"{graphs_dir}/{country_code}_CO2_Emissions_Bar_Plot.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close() # Close the plot to free memory

def plot_co2_emissions_histogram(dataframe, country_code, indicator_code):
    """
    Plots a histogram of CO2 emissions (metric tons per capita) for a given country code.
    Saves the plot to the 'Graphs' folder.
    """
    mask_indicator = dataframe['IndicatorCode'] == indicator_code
    mask_country = dataframe['CountryCode'] == country_code

    country_data = dataframe[mask_indicator & mask_country]

    if country_data.empty:
        print(f"No CO2 data found for {country_code} with indicator {indicator_code}. Skipping histogram.")
        return

    country_name = country_data['CountryName'].iloc[0]
    indicator_name = country_data['IndicatorName'].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.hist(country_data['Value'].values, bins=10, density=False, facecolor='red', alpha=0.75)
    plt.xlabel(indicator_name)
    plt.ylabel('Number of Years')
    plt.title(f'Histogram of {indicator_name} in {country_name}')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    filename = f"{graphs_dir}/{country_code}_CO2_Emissions_Histogram.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close() # Close the plot to free memory

def plot_gdp_co2_scatterplot(dataframe, country_code, co2_indicator_code, gdp_indicator_regex):
    """
    Plots a scatter plot of CO2 emissions vs. GDP per capita for a given country code.
    Saves the plot to the 'Graphs' folder.
    """
    mask_co2_indicator = dataframe['IndicatorCode'] == co2_indicator_code
    mask_gdp_indicator = dataframe['IndicatorName'].str.contains(gdp_indicator_regex, regex=True)
    mask_country = dataframe['CountryCode'] == country_code

    co2_data = dataframe[mask_co2_indicator & mask_country][['Year', 'Value']].rename(columns={'Value': 'CO2_Value'})
    gdp_data = dataframe[mask_gdp_indicator & mask_country][['Year', 'Value']].rename(columns={'Value': 'GDP_Value'})

    merged_data = pd.merge(co2_data, gdp_data, on='Year', how='inner')

    if merged_data.empty:
        print(f"No overlapping CO2 and GDP data found for {country_code} for scatter plot. Skipping.")
        return

    country_name = dataframe[mask_country]['CountryName'].iloc[0] # Get country name from original data
    co2_name = dataframe[mask_co2_indicator]['IndicatorName'].iloc[0]
    gdp_name = dataframe[mask_gdp_indicator]['IndicatorName'].iloc[0]

    fig, axis = plt.subplots(figsize=(10, 7))
    axis.yaxis.grid(True)
    axis.set_title(f'CO2 Emissions vs. GDP Per Capita in {country_name} (Aligned by Year)', fontsize=14)
    axis.set_xlabel(gdp_name, fontsize=12)
    axis.set_ylabel(co2_name, fontsize=12)

    X = merged_data['GDP_Value']
    Y = merged_data['CO2_Value']

    axis.scatter(X, Y, alpha=0.6, edgecolors='w', s=50)
    plt.tight_layout()

    # Save the figure
    filename = f"{graphs_dir}/{country_code}_GDP_vs_CO2_Scatter_Plot.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close() # Close the plot to free memory

print("\n--- Plotting for the FIRST FIVE unique countries ---")

# Determine how many country codes to plot (first 5 or fewer if less than 5 exist)
# num_countries_to_plot = min(5, len(country_codes))
countries_to_plot = ['ARB', 'CAN', 'BGR', 'IND', 'USA']

for code in countries_to_plot:
    print(f"\n--- Generating plots for Country: {code} ---")
    plot_co2_emissions_line(data, code, CO2_INDICATOR_CODE)
    plot_co2_emissions_bar(data, code, CO2_INDICATOR_CODE)
    plot_co2_emissions_histogram(data, code, CO2_INDICATOR_CODE)
    plot_gdp_co2_scatterplot(data, code, CO2_INDICATOR_CODE, GDP_INDICATOR_REGEX)


print("\n--- General Visualizations (Not country-specific loops) ---")

# Heatmap (Correlation Matrix of All Features Including Encoded Categorical)
print("\n--- Heatmap of All Feature Correlations ---")
# Use data_encoded to include label-encoded categorical features in the correlation matrix
if not data_encoded.empty:
    corr = data_encoded.corr(numeric_only=True) # Ensure correlation only on numeric columns
    plt.figure(figsize=(12, 10)) # Adjusted figure size for better readability
    sns.heatmap(corr, annot=True, square=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
    plt.title("Correlation Matrix of All Features (Including Encoded Categorical)", fontsize=16)
    plt.tight_layout()

    # Save the figure
    filename = f"{graphs_dir}/Overall_Correlation_Heatmap.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close() # Close the plot to free memory
else:
    print("Data encoded dataframe is empty, cannot generate heatmap.")


# --- 5. Data Splitting for Machine Learning (using data_encoded) ---
print("\n--- Data Splitting for Machine Learning ---")
# Drop the 'Value' column from features (X) as it's the target (y)
# Also drop 'IndicatorCode' from features as requested, as it's likely encoded in the model input
# and might be redundant or less useful as a direct feature if other indicator details are present.
try:
    X = data_encoded.drop(['Value', 'IndicatorCode'], axis=1)
    y = data_encoded['Value'] # Dependent feature

    # Ensure X is a DataFrame (already is after drop, but keeping for robustness)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

except KeyError as e:
    print(f"Error during data splitting: {e}. Make sure 'Value' and 'IndicatorCode' columns exist.")
except Exception as e:
    print(f"An unexpected error occurred during data splitting: {e}")

print("\n--- Code Execution Completed ---")
