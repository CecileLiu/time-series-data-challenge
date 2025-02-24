import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot


''' 
The dataset is a time series dataset. It is numbers of pedestrian hourly of 10 blocks. 
'''
# Load dataset
file_path = "/mnt/data/pedestrian_time_series.xlsx"  # Update this if needed
df = pd.read_excel(file_path)

# Convert timestamp column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing values (optional: use interpolation)
df = df.interpolate()

'''
# To use seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Apply decomposition to one block (e.g., Block 1)
result = seasonal_decompose(df.iloc[:, 0], model="additive", period=24)  # 24-hour period for daily seasonality

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
result.observed.plot(ax=ax1, title="Observed")
result.trend.plot(ax=ax2, title="Trend")
result.seasonal.plot(ax=ax3, title="Seasonal")
result.resid.plot(ax=ax4, title="Residual")
plt.tight_layout()
plt.show()

'''

# Plot pedestrian count trends
plt.figure(figsize=(12, 6))
for col in df.columns:
    plt.plot(df.index, df[col], label=col)
plt.xlabel("Time")
plt.ylabel("Pedestrian Count")
plt.title("Pedestrian Count Over Time")
plt.legend()
plt.show()

# Rolling Mean and Standard Deviation (for stationarity analysis)
rolling_window = 24  # 24-hour window
rolling_mean = df.rolling(window=rolling_window).mean()
rolling_std = df.rolling(window=rolling_window).std()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df.iloc[:, 0], label="Original Data")
plt.plot(df.index, rolling_mean.iloc[:, 0], label="Rolling Mean", linestyle="dashed")
plt.plot(df.index, rolling_std.iloc[:, 0], label="Rolling Std", linestyle="dotted")
plt.xlabel("Time")
plt.ylabel("Pedestrian Count")
plt.title("Rolling Mean & Std of Block 1")
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller (ADF) Test for stationarity
def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    if result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is NOT stationary (consider differencing).")

# Apply ADF test to first block
adf_test(df.iloc[:, 0])

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 4))
autocorrelation_plot(df.iloc[:, 0])
plt.title("Autocorrelation Plot of Block 1")
plt.show()
