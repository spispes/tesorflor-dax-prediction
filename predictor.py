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

"""
Predictor is initialized with
    offset_file:= values to get offset days (timestamp ammount days)
    predict_file:= values to be predicted
    x_columns_names:= column names to be used as input
    y_column_name:= output

class attributes are
    timestamp:= set to 60
    sc:= scaler the scaler is fitted with the first clumn of the input data
    dimension:= number of input columns
    model:= prediction madel
 

"""

class Predictor (object):
    timestamp = 60
    dataset = np.array
    sc = MinMaxScaler(feature_range = (0, 1))
    dimension = 0
    model = []

    def __init__(self,offset_file, predict_file, x_columns_names, y_column_name):
       offset = self.getDataset(offset_file, x_columns_names)
       prediction_set = self.getDataset(predict_file, x_columns_names)
       dataset = offset.append(prediction_set)
       inputs = dataset[len(dataset) - len(prediction_set) - self.timestamp:].values
       self.sc.fit(inputs[:,0].reshape(-1,1))       
       #self.sc.fit(self.dataset.values.reshape(-1,1))
       self.dimension = len(x_columns_names)       
       self.model = self.createModel(self.initX(inputs))

    def predict(self, regressor):
        regressor = self.loadPredictor(regressor)
        predicted_stock_price_close = regressor.predict(self.model)
        predicted_stock_price_close = self.unscale(predicted_stock_price_close)
        return predicted_stock_price_close    
        
    def initX(self,inputs):
       for i in range (0, self.dimension):
           inputs = np.append(inputs, self.scale(inputs[:,i]), axis = 1)
       for i in range (0, self.dimension):    
           inputs = np.delete(inputs,0,axis = 1)
       return inputs
   
    def getDataset(self,file, x_columns_names):
        dataset = pd.read_csv(file, sep= ',')
        # dataset = pd.read_csv(file, sep= ',', index_column = 0)
        dataset = dataset.drop(['Adj Close', 'Volume'], axis=1)
        dataset = dataset.dropna()
        offset_data = []
        offset_data = dataset[x_columns_names].copy()
        return offset_data
    
    def createModel(self, inputs):
        training_set = inputs
        maxLen = len(inputs)+1   
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
    
    def plot(self, predicted, reference):
        plt.plot(reference, color = 'red', label = 'Real DAX Stock Price', linewidth=0.5)
        plt.plot(predicted, color = 'blue', label = 'Predicted DAX Stock Close Price', linewidth=1.0)
        plt.grid(b=None, which='major', axis='both')
        plt.title('DAX Stock Close Price Prediction')
        plt.legend()
        plt.show()

offset_file = './daten/offset_2019.csv'
predict_file = './daten/predict_05.csv'
input_columns = ['Open','Close']
output_column = 'Close'
pr = Predictor(offset_file,predict_file,input_columns,output_column)
#name = 'dax_regressor'
#name = 'dax_2_regressor'
name = 'd_2_l_4_u_60_dax_regressor'
output = pr.predict(name)

inputs = pr.getDataset(predict_file,input_columns)
pr.plot(output, inputs)
inputs = pd.DataFrame(inputs)
output = pd.DataFrame(output)

"""
with pd.ExcelWriter('./prediction/result.xlsx', engine='openpyxl', mode='a') as writer:
    #inputs.to_excel(writer, sheet_name='input')
    output.to_excel(writer, sheet_name= name)
"""