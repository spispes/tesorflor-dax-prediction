# Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from prepare_input import prepareInput
from prepare_input import getClose, getOpen
from loadRegresor import load_regressor


dataset = prepareInput('./daten/offset_2013.csv')
dataset_predict = prepareInput('./daten/predict.csv')

dataset_total_close = pd.concat((dataset['Close'], dataset_predict['Close']), axis = 0)

regressor_close = load_regressor('dax_regressor')
#regressor_close = load_regressor('dax_regressor')

sc = MinMaxScaler(feature_range = (0, 1), copy = False)
training_set_open = getClose(dataset)
training_set_open = sc.fit_transform(training_set_open.reshape(-1,1))

"""
input are the days to be predicted + 60 days from the past
e.g. dataset_predict contains 39 items --> input = 39 + 60
The last item in the list is the prediction for the next day
The last step in input preparation is to normalize the input range 0:1
"""
timestamp = 60

inputs = dataset_total_close[len(dataset_total_close) - len(dataset_predict) - timestamp:].values
#inputs = sc.fit_transform(inputs.reshape(-1,1))
inputs = sc.transform(inputs.reshape(-1,1))

"""
Prepare a prediction set where for each entry in dataset_predict
the last 60 days are added e.g.
[close day 0, close day -1, close day -2, ..., close day -60 ]
[close day 1, close day -1, close day -2, ..., close day -59 ]
[close day 2, close day -1, close day -2, ..., close day -58 ]
...
[close day max, close day -1, close day -2, ..., close day max - 60 ]
"""
# len + 1 is the last predicted day but there are no close values for this 
maxLen = len(inputs) +1
X_predict_close = []
for i in range(timestamp, maxLen):
    X_predict_close.append(inputs[i-timestamp:i, 0])
X_predict_close = np.array(X_predict_close)
X_predict_close = np.reshape(X_predict_close, (X_predict_close.shape[0], X_predict_close.shape[1], 1))

predicted_stock_price_close = regressor_close.predict(X_predict_close)
predicted_stock_price_close = sc.inverse_transform(predicted_stock_price_close)

real_stock_price_close = getClose(dataset_predict)
# Visualising the results
plt.plot(real_stock_price_close, color = 'red', label = 'Real DAX Stock Price')
plt.plot(predicted_stock_price_close, color = 'blue', label = 'Predicted DAX Stock Close Price')
plt.grid(b=None, which='major', axis='both')
plt.title('DAX Stock Close Price Prediction')
plt.legend()
plt.show()