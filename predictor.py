# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:06:21 2019

@author: peter
"""

import pandas as pd
import numpy as np
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

class Predictor (object):
    timestamp = 60
    layers = 3
    units = 50
    algorithm = 'adam' #rmsprop
    error = 'mean_squared_error' #mean_squared_logarithmic_error
    epochs = 200
    batch_size = 32

    def __init__(self,offset, predict, x_columns_names, y_column_name, regressor):
       self.offset = self.getDataset(offset)
       self.predict = self.getDataset(predict)
       self.regressor = self.loadPredictor(regressor)
       self.dataset = pd.concat((offset[x_columns_names], predict[x_columns_names]), axis = 0)      
       self.input = self.initX(x_columns_names)
       self.dimension = len(x_columns_names)
       self.reference = self.initY(y_column_name)
       self.model = self.createModel()

       

       
    def initX(self):
       inputs = self.offset[len(self.offset) - len(self.predict) - self.timestamp:].values
       for i in range (0, self.dimension):
           inputs = np.append(inputs, self.scale(inputs[:,i]), axis = 1)
       for i in range (0, len(self.dimension)):    
           inputs = np.delete(inputs,0,axis = 1)
       return inputs
   
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
        dataset_in.shape
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
        sc = MinMaxScaler(feature_range = (0, 1))
        unscaled = sc.fit_transform(unscaled.reshape(-1,1))
        return unscaled
    
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
        self.persistModel(regressor)
        
    def loadPredictor(self,name):
        reg = 'regressors/'+name+ '.json'
        weights = 'regressors/'+name+ '.h5'
        with open(reg, 'r') as f:
            regressor = model_from_json(f.read())
        regressor.load_weights(weights)
        return regressor
       
pr = Predictor('./daten/offset_2013.csv','./daten/predict.csv',['Open'],'Close','2019-05-05_19-17-34__dim_dax_regressor')
