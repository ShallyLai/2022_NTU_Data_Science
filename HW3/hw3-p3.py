import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

# Load data
test = pd.read_csv("hw3_Data2/test.csv", sep = ",")
train = pd.read_csv("hw3_Data2/train.csv", sep = ",")

# Combine train and test data
data = pd.concat([train, test], axis = 0)
data['Date'] = pd.to_datetime(data.Date, format = '%Y-%m-%d')
data.index = data['Date']
data = data.drop('Date', axis = 1)
# print(data)

# Split Date as index
train['Date'] = pd.to_datetime(train.Date, format = '%Y-%m-%d')
train.index = train['Date']
train = train.drop('Date', axis = 1)

test['Date'] = pd.to_datetime(test.Date, format = '%Y-%m-%d')
test.index = test['Date']
test = test.drop('Date', axis = 1)

# Take "Close" values as training and testing data
train_close = train['Close']
test_close = test['Close']
data_close = data['Close']
# print(type(test_close))

# model_autoARIMA = auto_arima(train_close, start_p=0, start_q=0,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=3, max_q=3, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=False,   # No Seasonality
#                       start_P=0, 
#                       D=0, 
#                       trace=True,
#                       error_action='ignore',  
#                       suppress_warnings=True, 
#                       stepwise=True)
# print(model_autoARIMA.summary())
# model_autoARIMA.plot_diagnostics(figsize=(15,8))
# plt.show()
# print(model_autoARIMA.order)#(0, 1, 0)
# print(model_autoARIMA.seasonal_order)#(0, 0, 0, 0)
# model = model_autoARIMA  # seeded from the model we've already fit

# Estimated differencing term d
from pmdarima.arima import ndiffs
kpss_diffs = ndiffs(train_close, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(train_close, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)
# print(f"d is {n_diffs}") # d = 1

# Estimating the seasonal differencing term D
from pmdarima.arima.utils import nsdiffs
# estimate number of seasonal differences using a Canova-Hansen test
D = nsdiffs(train_close, m = 10, max_D = 12, test='ch') 
# print("D is {0}".format(0)) # D = 0

# Find p and q
# acf = pm.plot_acf(train_close) # find q for MA
# pacf = pm.plot_pacf(train_close) # find p for AR
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# fig, axes = plt.subplots(2, 1)
# plot_acf(train_close, ax = axes[0])
# plot_pacf(train_close, ax = axes[1])
# plt.show()

# Build Model
model = pm.ARIMA(order = (0, n_diffs, 0))
model.fit(train_close)  
# model.update(test_close)
# print(model.summary())

def forecast_one_step():
    fc, conf_int = model.predict(n_periods = 1, return_conf_int = True)
    return fc.tolist()[0], np.asarray(conf_int).tolist()[0]

forecasts = []
confidence_intervals = []

for new_ob in test_close:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
    # model.update(new_ob)
    model.update(fc)

print(f"Mean squared error: {mean_squared_error(test_close, forecasts)}")
print(f"SMAPE: {smape(test_close, forecasts)}")

# Plot actual data and predicted data
plt.plot(data_close, color = 'blue', label = 'Training Data')
plt.plot(test.index, forecasts, color = 'green', marker = 'o', label = 'Predicted Price')
plt.plot(test.index, test_close, color = 'red', marker = "X", label = 'Actual Price')
plt.title('Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
# conf_int = np.asarray(confidence_intervals)
# plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], alpha=0.9, color='orange', label="Confidence Intervals")
plt.legend()
plt.show()

