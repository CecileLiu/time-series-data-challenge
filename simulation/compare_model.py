import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

# Fit SARIMA model
sarima_model = SARIMAX(df_diff.iloc[:, 0], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))  # p, d, q, seasonal period
sarima_results = sarima_model.fit()
print(sarima_results.summary())

# Forecasting with SARIMA
sarima_forecast_1h = sarima_results.forecast(steps=1)
sarima_forecast_3h = sarima_results.forecast(steps=3)
print("SARIMA 1-hour forecast:", sarima_forecast_1h)
print("SARIMA 3-hour forecast:", sarima_forecast_3h)

# Prepare features for XGBoost
lags = 24  # Use past 24 hours as features
for i in range(1, lags+1):
    df[f'lag_{i}'] = df.iloc[:, 0].shift(i)
df.dropna(inplace=True)

# Split data
X = df.drop(columns=[df.columns[0]])
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Forecasting with XGBoost
xgb_forecast_1h = xgb_model.predict(X_test.iloc[-1:].values)
xgb_forecast_3h = xgb_model.predict(X_test.iloc[-3:].values)
print("XGBoost 1-hour forecast:", xgb_forecast_1h)
print("XGBoost 3-hour forecast:", xgb_forecast_3h)

# Evaluate XGBoost
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Model Performance Analysis
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} MAE: {mae:.4f}")
    print(f"{model_name} RMSE: {rmse:.4f}")

evaluate_model(y_test, y_pred, "XGBoost")

def predict_next(model, input_data, model_type="transformer"):
    if model_type == "transformer":
        model.eval()
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        return prediction
    elif model_type == "xgboost":
        return model.predict(input_data.reshape(1, -1))[0]

def model_performance_summary():
    print("SARIMA Performance:")
    evaluate_model(y_test, sarima_results.fittedvalues, "SARIMA")
    print("Transformer Performance:")
    evaluate_model(y_test_trans.numpy(), y_pred_trans, "Transformer")
    print("XGBoost Performance:")
    evaluate_model(y_test, y_pred, "XGBoost")

model_performance_summary()
