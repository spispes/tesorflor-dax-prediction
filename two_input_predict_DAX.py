# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:48:37 2019

@author: peter
"""

# Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from prepare_input import prepareInput
from prepare_input import getClose
from loadRegresor import load_regressor


dataset = prepareInput('./daten/offset_2013.csv')
dataset_predict = prepareInput('./daten/predict.csv')
firstDataset_total = pd.concat((dataset['Open'], dataset_predict['Open']), axis = 0)
secondDataset_total = pd.concat((dataset['Close'], dataset_predict['Close']), axis = 0)

regressor_close = load_regressor('layers_5_units_60_algorithm_rmsprop_error_mean_squared_logarithmic_error_epochs_200_batchsize_32_dax_regressor')

"""
Not used for prediction just used to prepare scaler
"""
sc = MinMaxScaler(feature_range = (0, 1))
training_set_open = getClose(dataset)
sc.fit(training_set_open.reshape(-1,1))

"""
input are the days to be predicted + 60 days from the past
e.g. dataset_predict contains 39 items --> input = 39 + 60
The last item in the list is the prediction for the next day
The last step in input preparation is to normalize the input range 0:1
"""
timestamp = 60

firstInputs = firstDataset_total[len(firstDataset_total) - len(dataset_predict) - timestamp:].values
secondInputs = secondDataset_total[len(secondDataset_total) - len(dataset_predict) - timestamp:].values

firstInputs = sc.transform(firstInputs.reshape(-1,1))
secondInputs = sc.transform(secondInputs.reshape(-1,1))
overalInputs = np.column_stack((firstInputs,secondInputs))

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
maxLen = len(overalInputs) +1
X_1 = []

for i in range(timestamp, maxLen):
    X_1.append(overalInputs[i-timestamp:i, :])

X_1 = np.array(X_1)

X_predict = np.reshape(X_1, (X_1.shape[0], X_1.shape[1], 2))

predicted_stock_price_close = regressor_close.predict(X_predict)
predicted_stock_price_close = sc.inverse_transform(predicted_stock_price_close)

real_stock_price_close = getClose(dataset_predict)
# Visualising the results
plt.plot(real_stock_price_close, color = 'red', label = 'Real DAX Stock Price')
plt.plot(predicted_stock_price_close, color = 'blue', label = 'Predicted DAX Stock Close Price')
plt.grid(b=None, which='major', axis='both')
plt.title('DAX Stock Close Price Prediction')
plt.legend()
plt.show()