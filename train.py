# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:40:30 2019

@author: peter
"""

import numpy as np

def train(filename, inputColumn, outputColumn):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from prepare_input import prepareInput
    trainSet = prepareInput(filename)    
    model = createModel(trainSet, inputColumn)
    reference = getReference(trainSet, outputColumn)
    regressor = Sequential()
    regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (model.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 60, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 60, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 60))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    # test_2019-04-18_07-14-07_dji_regressor schlecht regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_logarithmic_error')
    # test_2019-04-18_07-51-17_dji_regressor schlecht regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(model, reference, epochs = 120, batch_size = 32)
    persistModel(regressor)
    
    
def persistModel(regressor):
    import datetime
    import time
    regressor_json = regressor.to_json()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    with open("regressors/test_"+st+"_dji_regressor.json", "w") as json_file:
        json_file.write(regressor_json)
    regressor.save_weights("regressors/test_"+st+"_dji_regressor.h5")


def createModel(trainSet, inputColumn):
    inputSet = scale(trainSet[inputColumn].copy())
    trainingItems = len(trainSet)
    timestamps = 60
    X_train = []
    for i in range(timestamps, trainingItems):
        X_train.append(inputSet[i-timestamps:i, 0])
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train

def getReference(trainSet, outputColumn):
    outputSet = scale(trainSet[outputColumn].copy())
    trainingItems = len(trainSet)
    timestamps = 60
    y_train = []
    for i in range(timestamps, trainingItems):
        y_train.append(outputSet[i, 0])
    y_train = np.array(y_train)
    return y_train
    

def scale(unscaledSet):
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    scaledSet = sc.fit_transform(unscaledSet.reshape(-1,1))
    return scaledSet


train('./daten/DOW_JONES_lerndaten.csv', 'Close', 'Close')