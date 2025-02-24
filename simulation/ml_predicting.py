import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

# Fit SARIMA model: baseline model
sarima_model = SARIMAX(df_diff.iloc[:, 0], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))  # p, d, q, seasonal period
sarima_results = sarima_model.fit()
print(sarima_results.summary())

# Prepare features for XGBoost: baseline model
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

# Evaluate XGBoost
print("XGBoost MAE:", mean_absolute_error(y_test, y_pred))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Transformer-based Model for Time Series Forecasting
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])  # Predict next time step
        return x

# Prepare Data for Transformer
sequence_length = 24
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_trans, y_trans = create_sequences(df.iloc[:, 0].values, sequence_length)
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train_trans = torch.tensor(X_train_trans, dtype=torch.float32).unsqueeze(-1)
y_train_trans = torch.tensor(y_train_trans, dtype=torch.float32)
X_test_trans = torch.tensor(X_test_trans, dtype=torch.float32).unsqueeze(-1)
y_test_trans = torch.tensor(y_test_trans, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_trans, y_train_trans)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize Transformer model
model = TimeSeriesTransformer(input_dim=1, model_dim=64, num_heads=2, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Transformer model
for epoch in range(10):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate Transformer Model
y_pred_trans = model(X_test_trans).detach().numpy().squeeze()
print("Transformer MAE:", mean_absolute_error(y_test_trans.numpy(), y_pred_trans))
print("Transformer RMSE:", np.sqrt(mean_squared_error(y_test_trans.numpy(), y_pred_trans)))

# Inferencing with Transformer Model
def predict_next_hour(model, input_sequence):
    model.eval()
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

# Example inference
example_input = X_test_trans[0].numpy()
predicted_value = predict_next_hour(model, example_input)
print("Predicted next hour pedestrian count:", predicted_value)