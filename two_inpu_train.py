# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:06:48 2019

@author: peter
"""

import numpy as np

def train(filename, first_inputColumn, second_inputColumn, outputColumn):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from prepare_input import prepareInput
    trainSet = prepareInput(filename)    
    model = createModel(trainSet, first_inputColumn,second_inputColumn, 60)

    reference = getReference(trainSet, outputColumn)
    regressor = Sequential()
    regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (model.shape[1], 2)))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units = 60, return_sequences = True))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units = 60, return_sequences = True))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units = 60, return_sequences = True))
    regressor.add(Dropout(0.3))
    regressor.add(LSTM(units = 60))
    regressor.add(Dropout(0.3))
    regressor.add(Dense(units = 1))
    # test_2019-04-18_07-14-07_dji_regressor schlecht regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_logarithmic_error')
    # test_2019-04-18_07-51-17_dji_regressor schlecht regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_logarithmic_error')
    regressor.fit(model, reference, epochs = 200, batch_size = 32)
    persistModel(regressor)
    
    
def persistModel(regressor):
    import datetime
    import time
    regressor_json = regressor.to_json()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    with open("regressors/two_input_"+st+"_dax_regressor.json", "w") as json_file:
        json_file.write(regressor_json)
    regressor.save_weights("regressors/two_input_"+st+"_dax_regressor.h5")

"""
The train set is expected to have all required data
The firstColumn is one unrelated value e.g. open or volume or max
the secondColumn is the column to be predicted e.g. close these values are connected over time
"""
def createModel(trainSet, firstInputColumn, secontInputColumn, timestamps):
    firstInputSet = scale(trainSet[firstInputColumn].copy())
    secondInputSet = scale(trainSet[secontInputColumn].copy())
    training_set = np.column_stack((firstInputSet,secondInputSet))
    trainingItems = len(trainSet)
    X_1_train = []
    for i in range(timestamps, trainingItems):
        X_1_train.append(training_set[i-timestamps:i, :])
    X_1_train = np.array(X_1_train)
#    X_2_train = np.array(X_2_train)
    X_1_train = np.reshape(X_1_train, (X_1_train.shape[0], X_1_train.shape[1], 2))
    return X_1_train

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


train('./daten/train_dax.csv', 'Open', 'Close', 'Close')