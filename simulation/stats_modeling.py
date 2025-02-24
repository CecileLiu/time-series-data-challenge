import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
        return False  # No differencing needed
    else:
        print("The data is NOT stationary (consider differencing).")
        return True

# Check stationarity for Block 1
needs_differencing = adf_test(df.iloc[:, 0])

# Apply differencing if needed
if needs_differencing:
    df_diff = df.diff().dropna()
else:
    df_diff = df

# Fit ARIMA model
arima_model = ARIMA(df_diff.iloc[:, 0], order=(1, 1, 1))  # p, d, q # one lag, one difference, one moving average component.
darima_results = arima_model.fit()
print(darima_results.summary())

# Fit SARIMA model
sarima_model = SARIMAX(df_diff.iloc[:, 0], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))  # p, d, q, seasonal period
sarima_results = sarima_model.fit()
print(sarima_results.summary())

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 4))
autocorrelation_plot(df.iloc[:, 0])
plt.title("Autocorrelation Plot of Block 1")
plt.show()