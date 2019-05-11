# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:06:21 2019

@author: peter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json

class Predictor (object):
    timestamp = 60

    def __init__(self,offset, predict, x_columns_names, y_column_name):
       self.offset = self.getDataset(offset, x_columns_names)
       self.predictionset = self.getDataset(predict, x_columns_names)
       self.dataset = self.offset.append(self.predictionset)
       self.sc = MinMaxScaler(feature_range = (0, 1))
       self.sc.fit(self.dataset.values.reshape(-1,1))
       self.reference = self.initY(y_column_name)
       self.dimension = len(x_columns_names)       
       self.input = self.initX()
       self.model = self.createModel()

    def predict(self, regressor):
        regressor = self.loadPredictor(regressor)
        predicted_stock_price_close = regressor.predict(self.model)
        predicted_stock_price_close = self.unscale(predicted_stock_price_close)
        return predicted_stock_price_close    
        
    def initX(self):
       inputs = self.dataset[len(self.dataset) - len(self.predictionset) - self.timestamp:].values
       for i in range (0, self.dimension):
           inputs = np.append(inputs, self.scale(inputs[:,i]), axis = 1)
       for i in range (0, self.dimension):    
           inputs = np.delete(inputs,0,axis = 1)
       return inputs
   
    def initY(self, y_column_name):
        outputset = self.dataset[y_column_name].copy()
        return outputset
   
    def getDataset(self,file, x_columns_names):
        dataset = pd.read_csv(file, sep= ',', index_col = 0)
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
    
    def plot(self, predicted):
        plt.plot(self.predictionset, color = 'red', label = 'Real DAX Stock Price')
        plt.plot(predicted, color = 'blue', label = 'Predicted DAX Stock Close Price')
        plt.grid(b=None, which='major', axis='both')
        plt.title('DAX Stock Close Price Prediction')
        plt.legend()
        plt.show()
       
pr = Predictor('./daten/offset_2013.csv','./daten/predict.csv',['Close'],'Close')
result = pr.predict('dim_1_layers_4_units_60_algorithm_adam_error_mean_squared_logarithmic_error_epochs_150_batchsize_32_dax_regressor')
pr.plot(result)
