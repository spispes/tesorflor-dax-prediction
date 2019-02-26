# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt


from prepare_input import prepareInput
from prepare_input import getOpen
from prepare_input import getClose

# Importing the training set
dataset_train = prepareInput('GDAXI_train.csv')
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
for i in range(60, 2280):
    X_train_close.append(training_set_open_scaled[i-60:i, 0])
    y_train_close.append(training_set_close_scaled[i, 0])
X_train_close, y_train_close = np.array(X_train_close), np.array(y_train_close)

# Reshaping
X_train_close = np.reshape(X_train_close, (X_train_close.shape[0], X_train_close.shape[1], 1))

# Building the RNN for open

# Importing the Keras libraries and packages
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
regressor_close.fit(X_train_close, y_train_close, epochs = 100, batch_size = 32)
regressor_close.

"""

 Making the predictions and visualising the results
 the prediction is based on this years values only
 The prediction is compared to the real values included in GDAXI_test.csv

"""
# Getting the real stock price of 2019
dataset_test = prepareInput('GDAXI_test.csv')
real_stock_price_close = getClose(dataset_test)

import pandas as pd
# Getting the predicted stock price of for opening
dataset_total_close = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total_close[len(dataset_total_close) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

maxLen = len(inputs)
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
plt.xlabel('Time')
plt.ylabel('DAX Stock Price')
plt.legend()
plt.show()