import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from tsfresh.utilities.dataframe_functions import roll_time_series
from pmdarima.preprocessing import FourierFeaturizer
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

df_price = pd.read_pickle('./dataset/price.pkl')

df_price.info()
df_price.describe()

def plot_time_series(df:pd.DataFrame):
    plt.figure(figsize=(15,6))
    plt.plot(df)
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel(str(df.columns[0]))
    plt.title(df.columns[0])
    plt.show()

plot_time_series(df_price[['Open']])

def fill_missing_dates(df: pd.DataFrame, impute_value=np.nan)->pd.DataFrame:
    min_date = min(df_price.index)
    max_date = max(df_price.index)
    datetime_index_range = pd.date_range(min_date, max_date, freq='D')

    return df.reindex(datetime_index_range, fill_value=impute_value)

df_price_imputed = fill_missing_dates(df_price[['Open']])
df_price_imputed['Open'] = df_price_imputed['Open'].bfill()

# Outlier check with isolation forest
# Check outliers
iso_forest = IsolationForest(contamination=.01)
df_price_imputed['anomaly'] = iso_forest.fit_predict(df_price_imputed[['Open']])

plt.figure(figsize=(15,6))
plt.plot(df_price_imputed['Open'], label='Actual')
df_anomalies = df_price_imputed[df_price_imputed['anomaly']==-1]
plt.scatter(df_anomalies.index, df_anomalies['Open'], color='red', label='Anomalies')
plt.legend()
plt.grid()
plt.show()

# Roll time series
df_price['Group'] = 'A'
df_price['Date'] = df_price.index
df_rolled_ts = roll_time_series(df_price[['Date','Group','Open']].reset_index(drop=True), column_id='Group', column_sort='Date')

# Test stationarity
def test_stationarity(df: pd.DataFrame, significance_level = 0.05):
    adf_test_results = adfuller(df)
    test_statistic = adf_test_results[0]
    p_val = adf_test_results[1]

    if p_val<=significance_level:
        conclusion = 'Reject H0, time series is stationary'
    else:
        conclusion = 'Accept H0, time series has a unit root therefore is not stationary'

    print(f'p value of the test: {p_val}, significance: {significance_level}, conclusion: {conclusion}')

df_price_stationary = df_price_imputed.diff(1).dropna()
test_stationarity(df_price_stationary)

# Plot ACF - PACF
def plot_acf_pacf(df: pd.DataFrame, lags: int, figsize: tuple):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    plot_acf(df, lags=lags, ax=ax[0])
    plot_pacf(df, lags=lags, ax=ax[1])
    plt.show()

plot_acf_pacf(df_price_stationary[['Open']], lags=10, figsize=(16,8))

# Granger causality
grangercausalitytests(df_price[['Open','High']], maxlag=5)

# Creating features
def create_additional_regressors(dataframe: pd.DataFrame, value_column: str, window_size: int):
 # Statistical measures
    dataframe[f'recent_mean_{window_size}'] = dataframe[value_column].rolling(window=window_size).mean()
    dataframe[f'recent_std_dev_{window_size}'] = dataframe[value_column].rolling(window=window_size).std()
    dataframe[f'recent_min_{window_size}'] = dataframe[value_column].rolling(window=window_size).min()
    dataframe[f'recent_max_{window_size}'] = dataframe[value_column].rolling(window=window_size).max()

    # 1 week percentage growth
    dataframe['1_week_pct_growth'] = dataframe[value_column].pct_change(periods=7)

    # Growth volatility past month
    dataframe['growth_volatility_past_month'] = dataframe[value_column].pct_change().rolling(window=30).std()

    return dataframe

def create_fourier_terms(df: pd.DataFrame, col_name: str, seasonal_period: int, number_of_sin_cos: int):

    four_terms = FourierFeaturizer(seasonal_period, number_of_sin_cos)
    _, exog = four_terms.fit_transform(df[col_name])
    exog.index = df.index

    return pd.concat([df, exog], axis=1)

def add_lag_values(df: pd.DataFrame, col_name: str, lag_indices: list):
    df = df.copy()
    
    for lag in lag_indices:
        df[f'{col_name}_lag_value_{str(lag)}'] = df[col_name].shift(lag)

    return df

# Modeling

# Create for forecast
df_with_exog = pd.concat([df_price_stationary, df_price[['High','Low']]], axis=1).drop(columns='anomaly')
df_with_exog = df_with_exog.bfill()

forecast_horizon = 30

df_futures = pd.DataFrame({
    'Open':forecast_horizon*[np.nan]
    ,'High':forecast_horizon*[np.nan]
    ,'Low':forecast_horizon*[np.nan]
}, index=pd.date_range((max(df_price.index)+timedelta(days=1)).date(), (max(df_price.index)+timedelta(days=forecast_horizon)).date()))

df_futures = pd.concat([df_with_exog, df_futures])
df_futures[['High','Low']] = df_futures[['High','Low']].shift(forecast_horizon)
df_futures = df_futures.iloc[forecast_horizon:]

# ARIMA
train_set = df_price[['Open','High','Low']].iloc[:-forecast_horizon]
test_set = df_price[['Open','High','Low']].iloc[-forecast_horizon:]

from itertools import product

p_orders = [1,2]
d_orders = [1]
q_orders = [1,2]

arima_orders = product(p_orders, d_orders, q_orders)
arima_orders = list(arima_orders)

rmse_list = []
aic_list = []
bic_list = []

for order in arima_orders[:2]:
    
    model = ARIMA(
        endog=train_set['Open']
        ,exog=train_set[['High','Low']]
        ,order=order
    ).fit()

    y_pred = model.get_forecast(len(test_set.index), exog=np.array(test_set[['High','Low']]))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1], exog=np.array(test_set[['High','Low']]))

    y_pred_df.index = test_set.index
    df_pred = pd.concat([test_set, y_pred_df], axis=1)

    rmse = mean_squared_error(df_pred['Open'], df_pred['Predictions'], squared=False)
    rmse_list.append(rmse)
    aic_list.append(model.aic)
    bic_list.append(model.aic)

    print('Order')
    print(model.summary())
    print('Residual diagnostics')
    model.plot_diagnostics()
    print(acorr_ljungbox(model.resid, lags=5))
    print('------------------')

# Final Forecast
model = ARIMA(
    endog=pd.concat([train_set['Open'],test_set['Open']])
    ,order=(1,1,1)
).fit()

y_pred = model.get_forecast(forecast_horizon)
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])




# AR
model = AutoReg(train_set[['Open']],lags=[1,2], exog=train_set[['High','Low']]).fit()
acorr_ljungbox(model.resid, lags=[1,2])

df_pred = pd.DataFrame(model.predict(start=len(train_set), end=len(train_set)+forecast_horizon, exog=pd.concat([
    train_set[['High','Low']]
    ,test_set[['High','Low']]
    ])), columns=['Open'])


######
ts_split = TimeSeriesSplit(n_splits=5, test_size=30)  
for p in p_orders:
    rmse_scores = []
    aic_scores = []
    
    for train_index, test_index in ts_split.split(df_price[['Open']]):
        #train_data, test_data = df_price[['Open']][train_index], df_price[['Open']][test_index]
        print(train_index, test_index)

        #...

def seasonality_plot(df, target_col_name, seasonality_type):
    # Monthly
    cols = ["month", "dayofmonth", str(target_col_name)]
    # Weekly
    cols = ["weekofyear", "dayofweek", str(target_col_name)]
    # Yearly
    cols = ["year", "month", str(target_col_name)]

    pivot_table = pd.pivot_table(
        df[cols],
        values=str(target_col_name),
        index=[cols[1]],
        columns=[cols[0]],
        aggfunc=np.mean,
    )

    plt.figure(figsize=(15, 6))
    for i, col in enumerate(pivot_table.columns):
        plt.plot(pivot_table.iloc[:, i], label=str(col))
    plt.title(f"{seasonality_type} seasonality plot")
    plt.xlabel(str(cols[1]))
    plt.ylabel(target_col_name)
    plt.grid()
    plt.legend()
    plt.show()