# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:06:21 2019

@author: peter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json

class Predictor (object):
    timestamp = 60
    layers = 3
    units = 50
    algorithm = 'adam' #rmsprop
    error = 'mean_squared_error' #mean_squared_logarithmic_error
    epochs = 200
    batch_size = 32

    def __init__(self,offset, predict, x_columns_names, y_column_name):
       self.sc = MinMaxScaler(feature_range = (0, 1)) 
       self.offset = self.getDataset(offset, x_columns_names)
       self.predictionset = self.getDataset(predict, x_columns_names)
       self.dataset = self.offset.append(self.predictionset)
       self.reference = self.initY(y_column_name)
       self.sc.fit(self.reference.reshape(-1,1))
       self.dimension = len(x_columns_names)       
       self.input = self.initX()
       self.model = self.createModel()

    def predict(self, regressor):
        regressor = self.loadPredictor(regressor)
        predicted_stock_price_close = regressor.predict(self.model)
        predicted_stock_price_close = self.unscale(predicted_stock_price_close)
        print(predicted_stock_price_close)
        return predicted_stock_price_close
    """    
         dataset 1603 (2013-01-02 -- 2019-05-08)
         predictionset 88 (2019-01-02 -- 2019-05-08
         timestamp 60
         so we need the last 148 days from the concatenated dataset
    """      
        
    def initX(self):
       inputs = self.dataset[len(self.dataset) - len(self.predictionset) - self.timestamp:].values
       for i in range (0, self.dimension):
           inputs = np.append(inputs, self.scale(inputs[:,i]), axis = 1)
       for i in range (0, self.dimension):    
           inputs = np.delete(inputs,0,axis = 1)
       return inputs
   
    def initY(self, y_column_name):
        output_set = self.dataset[y_column_name].copy()
        return output_set
   
    def getDataset(self,file, x_columns_names):
        dataset = pd.read_csv(file, sep= ',', index_col = 0)
        dataset.shape
        dataset = dataset.drop(['Adj Close', 'Volume'], axis=1)
        dataset = dataset.dropna()
        offset_data = []
        offset_data = dataset[x_columns_names].copy()
        return offset_data
    
    def createModel(self):
        training_set = self.input
        maxLen = len(self.input)+1   
        X_1_train = []
        for j in range(self.timestamp, maxLen):
            X_1_train.append(training_set[j-self.timestamp:j, :])
        X_1_train = np.array(X_1_train)
        X_1_train = np.reshape(X_1_train, (X_1_train.shape[0], X_1_train.shape[1], self.dimension))
        return X_1_train
    
    def scale(self, unscaled):
        scaled = self.sc.transform(unscaled.reshape(-1,1))
        return scaled
    
    def unscale(self, scaled):
       unscaled = self.sc.inverse_transform(scaled)
       return unscaled
    
    def loadPredictor(self,name):
        reg = 'regressors/'+name+ '.json'
        weights = 'regressors/'+name+ '.h5'
        with open(reg, 'r') as f:
            regressor = model_from_json(f.read())
        regressor.load_weights(weights)
        return regressor
       
pr = Predictor('./daten/offset_2013.csv','./daten/predict.csv',['Close'],'Close')
result = pr.predict('dax_2_regressor')
