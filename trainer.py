# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:06:21 2019

@author: peter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


class Trainer (object):
    timestamp = 60

    def __init__(self,inputfile, x_columns_names, y_column_name):
       self.sc = MinMaxScaler(feature_range = (0, 1)) 
       self.dataset = self.getDataset(inputfile)
       scalingRange = self.dataset[y_column_name].copy()
       self.sc.fit(self.dataset.values.reshape(-1,1))
       self.input = self.initX(x_columns_names)
       self.dimension = len(x_columns_names)
       self.reference = self.initY(y_column_name)
       self.model = self.createModel()      
       
    def initX(self, x_column_names):
       #x = self.dataset[x_column_names.copy()]
       training_set = self.dataset[x_column_names.copy()].values
       for i in range (0, len(x_column_names)):
           training_set = np.append(training_set, self.scale(training_set[:,i]), axis = 1)
       for i in range (0, len(x_column_names)):    
           training_set = np.delete(training_set,0,axis = 1)
       return training_set
   
    def initY(self, y_column_name):
        output_set = self.dataset[y_column_name].copy()
        output_set = self.scale(output_set)
        training_items = len(output_set)
        y = []
        for i in range(self.timestamp, training_items):
            y.append(output_set[i, 0])
        y = np.array(y)
        return y
   
    def getDataset(self,filename):
        dataset_in = pd.read_csv(filename, sep= ',', index_col = 0)
        dataset_in = dataset_in.drop(['Adj Close', 'Volume'], axis=1)
        dataset_in = dataset_in.dropna()  
        return dataset_in
    
    def createModel(self):
        training_set = self.input
        trainingItems = len(self.input)   
        X_1_train = []
        for j in range(self.timestamp, trainingItems):
            X_1_train.append(training_set[j-self.timestamp:j, :])
        X_1_train = np.array(X_1_train)
        X_1_train = np.reshape(X_1_train, (X_1_train.shape[0], X_1_train.shape[1], self.dimension))
        return X_1_train
    
    def scale(self, unscaled):
        scaled = self.sc.transform(unscaled.reshape(-1,1))
        return scaled
    
    def train(self, _layers, _units, _algorithm, _error, _epochs, _batch_size):
        regressor = Sequential()
        regressor.add(LSTM(units = _units, return_sequences = True, input_shape = (self.model.shape[1], self.dimension)))
        regressor.add(Dropout(0.2))
        for i in range (0, _layers):
            regressor.add(LSTM(units = _units, return_sequences = True))
            regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = _units))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = _algorithm, loss = _error)
        regressor.fit(self.model, self.reference, epochs = _epochs, batch_size = _batch_size)
        self.persistModel(regressor, self.dimension, _layers, _units, _algorithm, _error, _epochs, _batch_size)
        
    def persistModel(self, regressor,_dim, _layers, _units, _algorithm, _error, _epochs, _batch_size):
        regressor_json = regressor.to_json()
        name = 'dim_'+str(_dim)
        name = name + '_layers_'+str(_layers)
        name = name + '_units_' +str(_units)
        name = name + '_algorithm_' +str(_algorithm)
        name = name + '_error_' +str(_error)
        name = name + '_epochs_' +str(_epochs)
        name = name + '_batchsize_' +str(_batch_size)

        with open('regressors/'+name+'_dax_regressor.json', 'w') as json_file:
            json_file.write(regressor_json)
        regressor.save_weights('regressors/'+name+'_dax_regressor.h5')

       
tr = Trainer('./daten/offset_2013.csv',['Close'],'Close')
# tr.train(3, 60, 'adam', 'mean_squared_error', 200, 32) --> dax_down_regressor
# tr.train(3, 60, 'adam', 'mean_squared_logarithmic_error', 200, 32) --> 2019-05-07_19-49-44_1_dax_regressor forget it
# tr.train(3, 60, 'rmsprop', 'mean_squared_error', 200, 32) 2019-05-07_22-46-13_1_dax_regressor good
# tr.train(3, 60, 'rmsprop', 'mean_squared_logarithmic_error', 200, 32) --> 2019-05-07_23-09-19_1_dax_regressor better
for i in range(3,7):
    tr.train(i, 60, 'adam', 'mean_squared_logarithmic_error', 150, 32)

