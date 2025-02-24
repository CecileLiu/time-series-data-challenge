## Simulation tasks  
  
```
Task 1. The dataset is a time series dataset. It is numbers of pedestrian hourly of 10 blocks. Please do EDA to this dataset.
Task 2. Please do the so-called modeling
Task 3. Please use a transformer-based model to predict the numbers of pedestrian of these 10 blocks in next 1 hour.
Task 4. Please use a transformer-based model to predict the numbers of pedestrian of these 10 blocks in next 3 hour.
Task 5. Please give me a report outline of how you'll summarize the above findings
Task 6. Please provide me how should we deploy a pipeline which using the 2 models we've just trained. 
```  
  
### Task 1 - EDA  
#### EDA Steps:
1. Load & Inspect the Data  
    - Check for missing values and data types.
    - Convert timestamps to a proper `datetime` format.
    - Ensure data is sorted chronologically.
2. Visualize Trends & Seasonality  
    - Plot pedestrian counts over time for each block.
    - Use rolling averages to smooth trends.
-    Analyze daily/weekly seasonal patterns.
3. Check for Stationarity  
    - Use the **Augmented Dickey-Fuller (ADF) test** to determine if differencing is needed.
4. Autocorrelation & Lag Analysis  
    - Plot **autocorrelation (ACF)** & **partial autocorrelation (PACF)** to assess dependencies.
5. Check Missing Data Handling  
    - Count missing values and decide whether to impute or interpolate.  
  

### Task 2 - Modeling  

#### Statistical Time Series Models  
These models are often used by statisticians:  
1. ARIMA (AutoRegressive Integrated Moving Average): Suitable for univariate series, capturing trends and seasonality.
2. SARIMA (Seasonal ARIMA): Extends ARIMA for seasonal patterns.
3. VAR (Vector AutoRegression): Used for multivariate time series, capturing dependencies between blocks.  
  
**If the result of ADF test is non-stationary, what should we do?** we need to transform the data before using ARIMA/SARIMA.  
- How to Make a Time Series Stationary?  
    - Differencing (df.diff()):Subtract the previous value from the current value (first-order differencing). If still non-stationary, apply second-order differencing.
    - Log Transformation (np.log()): Stabilizes variance if there are exponential trends.
    - Box-Cox Transformation (scipy.stats.boxcox()): More general transformation, making variance more stable.  
  
  
### Task 5: Report Outline  
1. Introduction  
    - Objective of the study
    - Overview of dataset (pedestrian flow time series)
    - Challenges in time series forecasting
2. Exploratory Data Analysis (EDA)  
    - Data distribution and summary statistics
    - Missing value treatment
    - Trend and seasonality analysis (using seasonal_decompose)
    - Stationarity check (ADF test)
    - Autocorrelation and Partial Autocorrelation Analysis
3. Baseline Models  
    - SARIMA: Model selection, training, and evaluation
    - XGBoost: Feature engineering, training, and performance comparison
4. Transformer-based Forecasting  
    - Model architecture and training strategy
    - Evaluation on 1-hour and 3-hour predictions
    - Comparison with baseline models
5. Model Performance Analysis  
    - MAE, RMSE across different time horizons
    - Error distribution
    - Strengths and weaknesses of each approach
  
  
### Task 6: Deployment Strategy
1. Preprocessing Pipeline  
    - Load new real-time pedestrian data
    - Apply necessary transformations (scaling, encoding)
    - Generate lag features if using XGBoost
2. Model Serving
    - Deploy SARIMA and XGBoost as batch inference jobs (e.g., scheduled predictions every hour)
    - Deploy Transformer model as a REST API (Flask/FastAPI) for real-time forecasting
    - Use a model registry to track different versions
3. Inference Pipeline
    - Accept real-time data via API or batch processing
    - Apply model inference
    - Return forecasts with confidence intervals
4. Monitoring & Logging
    - Log predictions and errors
    - Set up drift detection (if new patterns emerge)
    - Implement automated retraining triggers
