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


class Trainer (object):
    timestamp = 60
    layers = 3
    units = 50
    algorithm = 'adam' #rmsprop
    error = 'mean_squared_error' #mean_squared_logarithmic_error
    epochs = 200
    batch_size = 32

    def __init__(self,inputfile, x_columns_names, y_column_name):
       self.dataset = self.getDataset(inputfile)
       self.input = self.initX(x_columns_names)
       self.reference = self.initY(y_column_name)
       self.model = self.createModel()
       self.train()
       
    def initX(self, x_column_names):
       x = [int,[]]
       for i in range(0, len(x_column_names)):
           x[i] = self.dataset[x_column_names[i]].copy()
           x[i] = self.scale(x[i])
       training_set = np.column_stack(x)
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
        X_1_train = np.reshape(X_1_train, (X_1_train.shape[0], X_1_train.shape[1], training_set.ndim))
        return X_1_train
    
    def scale(self, unscaled):
        scaled = []
        sc = MinMaxScaler(feature_range = (0, 1))
        scaled = sc.fit_transform(unscaled.reshape(-1,1))
        return scaled
    
    def train(self):
        regressor = Sequential()
        regressor.add(LSTM(units = self.units, return_sequences = True, input_shape = (self.model.shape[1], self.input.ndim)))
        regressor.add(Dropout(0.2))
        for i in range (0, self.layers):
            regressor.add(LSTM(units = self.units, return_sequences = True))
            regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = self.units))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = self.algorithm, loss = self.error)
        regressor.fit(self.model, self.reference, epochs = self.epochs, batch_size = self.batch_size)
        self.persistModel(regressor)
        
    def persistModel(self, regressor):
        regressor_json = regressor.to_json()
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        with open("regressors/two_input_"+st+"_dax_regressor.json", "w") as json_file:
            json_file.write(regressor_json)
        regressor.save_weights("regressors/two_input_"+st+"_dax_regressor.h5")
       
tr = Trainer('./daten/offset_2013.csv',['Open', 'Close'],'Close')
