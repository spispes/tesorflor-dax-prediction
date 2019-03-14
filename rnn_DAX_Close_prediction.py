# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt


from prepare_input import prepareInput
from prepare_input import getOpen
from prepare_input import getClose

# Importing the training set
dataset_train = prepareInput('./daten/GDAXI_lerndaten.csv')
training_set_open = getOpen(dataset_train)
training_set_close = getClose(dataset_train)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_open_scaled = sc.fit_transform(training_set_open.reshape(-1,1))
training_set_close_scaled = sc.fit_transform(training_set_close.reshape(-1,1))

"""
 Creating a data structure for opening with 60 timesteps and 1 output
 X_train_open is the input file based on the last 60 close - values (starting at 0 up to 59 
 which is the first date to the date before the to be predicted day )
 predicting the next days (position 60) open
"""
X_train_close = []
y_train_close = []
for i in range(60, 1011):
    X_train_close.append(training_set_close_scaled[i-60:i, 0])
    y_train_close.append(training_set_close_scaled[i, 0])
X_train_close, y_train_close = np.array(X_train_close), np.array(y_train_close)

""" 
Reshaping adding a secon dimention to the network
Impup is X_train_close the prepared matrix and the connection is to
number of data sets = X_train_close.shape[0]
number of colums = X_train_close.shape[1] //60 in this case
"""
X_train_close = np.reshape(X_train_close, (X_train_close.shape[0], X_train_close.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor_close = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor_close.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_close.shape[1], 1)))
regressor_close.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor_close.add(LSTM(units = 50, return_sequences = True))
regressor_close.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor_close.add(LSTM(units = 50, return_sequences = True))
regressor_close.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor_close.add(LSTM(units = 50))
regressor_close.add(Dropout(0.2))

# Adding the output layer
regressor_close.add(Dense(units = 1))

# Compiling the RNN
regressor_close.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor_close.fit(X_train_close, y_train_close, epochs = 200, batch_size = 32)

"""
persist Model
"""
regressor_json = regressor_close.to_json()
with open("regressor.json", "w") as json_file:
    json_file.write(regressor_json)
regressor_close.save_weights("regressor_.h5")
"""

 Making the predictions and visualising the results
 the prediction is based on this years values only
 The prediction is compared to the real values included in GDAXI_test.csv

"""
# Getting the real stock price of 2019
dataset_predict = prepareInput('./daten/GDAXIto_be_predicted.csv')
real_stock_price_close = getClose(dataset_predict)

import pandas as pd
# Getting the predicted stock price of for opening
dataset_total_close = pd.concat((dataset_train['Close'], dataset_predict['Close']), axis = 0)
"""
input are the days to be predicted + 60 days from the past
e.g. dataset_predict contains 39 items --> input = 39 + 60
The last item in the list is the prediction for the next day
The last step in input preparation is to normalize the input range 0:1
"""
inputs = dataset_total_close[len(dataset_total_close) - len(dataset_predict) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
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
for i in range(60, maxLen):
    X_predict_close.append(inputs[i-60:i, 0])
X_predict_close = np.array(X_predict_close)
X_predict_close = np.reshape(X_predict_close, (X_predict_close.shape[0], X_predict_close.shape[1], 1))

predicted_stock_price_close = regressor_close.predict(X_predict_close)
predicted_stock_price_close = sc.inverse_transform(predicted_stock_price_close)


# Visualising the results
plt.plot(real_stock_price_close, color = 'red', label = 'Real DAX Stock Price')

plt.plot(predicted_stock_price_close, color = 'blue', label = 'Predicted DAX Stock Close Price')
plt.grid(b=None, which='major', axis='both')
plt.title('DAX Stock Close Price Prediction')
plt.inputlabel('Time')
plt.outputlabel('DAX Stock Price')
plt.legend()
plt.show()
